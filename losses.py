"""Loss functions, co-training schedule, and EMA averaging."""

from __future__ import annotations

import math
from dataclasses import asdict

import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import CoTrainConfig


# ---------------------------------------------------------------------------
#  CTC helpers
# ---------------------------------------------------------------------------

def mask_logits_beyond_lens(logits: torch.Tensor, sub_lens: torch.Tensor,
                            blank_id: int = 0) -> torch.Tensor:
    """Force padded time-steps to predict *blank* with certainty.

    This prevents the CTC decoder from emitting spurious tokens after the
    valid sequence ends.
    """
    B, T, V = logits.shape
    pos = torch.arange(T, device=logits.device).unsqueeze(0)
    pad_mask = pos >= sub_lens.unsqueeze(1)
    m = logits.clone()
    m[pad_mask] = -1e9
    m[pad_mask, blank_id] = 1e9
    return m


def compute_ctc_loss(logits: torch.Tensor, sub_lens: torch.Tensor,
                     tokens: torch.Tensor, tok_lens: torch.Tensor,
                     blank_id: int = 0) -> torch.Tensor:
    """Compute CTC loss with blank-masking on padded positions."""
    ctc_fn = nn.CTCLoss(blank=blank_id, zero_infinity=True)
    masked = mask_logits_beyond_lens(logits, sub_lens, blank_id)
    lp = F.log_softmax(masked, dim=-1).permute(1, 0, 2)
    return ctc_fn(lp, tokens, sub_lens, tok_lens)


# ---------------------------------------------------------------------------
#  TTS regularizers
# ---------------------------------------------------------------------------

class TTSRegularizers(nn.Module):
    """Energy, smoothness, and band-limit penalties for TTS mel output.

    These keep the TTS output spectrogram well-behaved and prevent it from
    producing extreme or degenerate patterns.

    Args:
        lambda_energy: Weight for energy penalty.
        lambda_smoothness: Weight for temporal smoothness (total variation).
        lambda_bandlimit: Weight for band-limit penalty.
    """

    def __init__(self, lambda_energy: float = 0.01,
                 lambda_smoothness: float = 0.005,
                 lambda_bandlimit: float = 0.005):
        super().__init__()
        self.le = lambda_energy
        self.ls = lambda_smoothness
        self.lb = lambda_bandlimit
        # Band of interest (mel bins 2..30 out of 40)
        self.slo, self.shi = 2, 30

    def forward(self, mel_pred: torch.Tensor) -> dict:
        """Return a dict of named regularizer losses and ``reg_total``."""
        losses = {}
        fe = mel_pred.pow(2).mean(dim=1)
        losses["energy"] = self.le * (
            F.relu(fe - 5.0).mean() + F.relu(0.1 - fe).mean()
        )
        if mel_pred.shape[2] > 1:
            tv = (mel_pred[:, :, 1:] - mel_pred[:, :, :-1]).abs().mean()
        else:
            tv = torch.tensor(0.0, device=mel_pred.device)
        losses["smoothness"] = self.ls * tv
        te = mel_pred.pow(2).sum(dim=1).mean()
        se = mel_pred[:, self.slo : self.shi, :].pow(2).sum(dim=1).mean()
        losses["bandlimit"] = self.lb * (1.0 - se / (te + 1e-8))
        losses["reg_total"] = (
            losses["energy"] + losses["smoothness"] + losses["bandlimit"]
        )
        return losses


# ---------------------------------------------------------------------------
#  Co-training curriculum schedule
# ---------------------------------------------------------------------------

class CoTrainSchedule:
    """Three-phase curriculum for co-training TTS and ASR.

    Phase 0 (warmup): PS anchor trains ASR; CTC gradients do NOT flow to TTS.
    Phase 1 (ramp):   CTC-to-TTS weight ramps from 0 to 1; PS anchor decays.
    Phase 2 (full):   Pure co-training + Griffin-Lim roundtrip; no PS anchor.

    SNR is interpolated from easy (15, 30) dB to hard (-10, 15) dB across
    phases 0-2 to progressively increase augmentation difficulty.

    Args:
        cotrain_cfg: ``CoTrainConfig`` dataclass.
    """

    _SNR_EASY = (15.0, 30.0)
    _SNR_HARD = (-10.0, 15.0)

    def __init__(self, cotrain_cfg: CoTrainConfig):
        self.cfg = cotrain_cfg
        self.warmup_end = cotrain_cfg.warmup_steps
        self.ramp_end = cotrain_cfg.warmup_steps + cotrain_cfg.ramp_steps

    def get_phase(self, step: int) -> int:
        if step < self.warmup_end:
            return 0
        if step < self.ramp_end:
            return 1
        return 2

    def get_lambda_ctc_tts(self, step: int) -> float:
        """Weight for CTC loss that back-props through TTS."""
        if not self.cfg.ctc_to_tts:
            return 0.0
        p = self.get_phase(step)
        if p == 0:
            return 0.0
        if p == 1:
            progress = (step - self.warmup_end) / max(self.cfg.ramp_steps, 1)
            return self.cfg.lambda_ctc_tts * progress
        return self.cfg.lambda_ctc_tts

    def get_lambda_mel(self, step: int) -> float:
        """Weight for mel-reconstruction loss (PS anchor, decays to floor)."""
        p = self.get_phase(step)
        if p == 0:
            return self.cfg.lambda_mel
        if p == 1:
            progress = (step - self.warmup_end) / max(self.cfg.ramp_steps, 1)
            return (
                self.cfg.lambda_mel
                + (self.cfg.lambda_mel_floor - self.cfg.lambda_mel) * progress
            )
        return self.cfg.lambda_mel_floor

    def should_detach_tts(self, step: int) -> bool:
        """Whether TTS output should be detached (Phase 0)."""
        return self.get_phase(step) == 0

    def _interp_snr(self, step: int):
        p = self.get_phase(step)
        if p == 0:
            return self._SNR_EASY
        if p == 2:
            return self._SNR_HARD
        progress = (step - self.warmup_end) / max(self.cfg.ramp_steps, 1)
        lo = self._SNR_EASY[0] + (self._SNR_HARD[0] - self._SNR_EASY[0]) * progress
        hi = self._SNR_EASY[1] + (self._SNR_HARD[1] - self._SNR_EASY[1]) * progress
        return (lo, hi)

    def get_mel_snr_range(self, step: int):
        return self._interp_snr(step)

    def get_wav_snr_range(self, step: int):
        return self._interp_snr(step)

    def get_lambda_roundtrip(self, step: int) -> float:
        return self.cfg.lambda_roundtrip

    def status(self, step: int) -> str:
        """Human-readable status string for logging."""
        snr = self.get_mel_snr_range(step)
        return (
            f"Phase {self.get_phase(step)} | "
            f"lam_ctc={self.get_lambda_ctc_tts(step):.3f} | "
            f"lam_mel={self.get_lambda_mel(step):.3f} | "
            f"lam_rt={self.get_lambda_roundtrip(step):.3f} | "
            f"snr=[{snr[0]:+.0f},{snr[1]:+.0f}]"
        )

    def state_dict(self) -> dict:
        return asdict(self.cfg)


# ---------------------------------------------------------------------------
#  Exponential Moving Average
# ---------------------------------------------------------------------------

class EMA:
    """Exponential moving average of model parameters.

    After ``update()`` is called each step, ``apply_shadow()`` temporarily
    replaces model weights with the averaged values for validation.
    ``restore()`` puts the original weights back.

    Args:
        model: The ``nn.Module`` whose parameters to track.
        decay: Averaging coefficient (e.g. 0.999).
    """

    def __init__(self, model: nn.Module, decay: float = 0.999):
        self.decay = decay
        self.shadow: dict[str, torch.Tensor] = {}
        self.backup: dict[str, torch.Tensor] = {}
        for name, param in model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = param.data.clone()

    def update(self, model: nn.Module):
        for name, param in model.named_parameters():
            if param.requires_grad and name in self.shadow:
                self.shadow[name].mul_(self.decay).add_(
                    param.data, alpha=1.0 - self.decay
                )

    def apply_shadow(self, model: nn.Module):
        """Replace model weights with shadow (averaged) weights."""
        for name, param in model.named_parameters():
            if param.requires_grad and name in self.shadow:
                self.backup[name] = param.data.clone()
                param.data.copy_(self.shadow[name])

    def restore(self, model: nn.Module):
        """Restore original weights after ``apply_shadow()``."""
        for name, param in model.named_parameters():
            if param.requires_grad and name in self.backup:
                param.data.copy_(self.backup[name])
        self.backup = {}

    def state_dict(self) -> dict:
        return {k: v.cpu() for k, v in self.shadow.items()}

    def load_state_dict(self, state_dict: dict):
        for k, v in state_dict.items():
            if k in self.shadow:
                self.shadow[k].copy_(v.to(self.shadow[k].device))
