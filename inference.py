"""Inference utilities: checkpoint loading, encoding, and decoding.

Usage::

    from ttr.inference import load_model, encode, decode

    tts, asr, cfg = load_model("checkpoints/best_model.pt")
    mel, wav = encode("move forward 5 meters", tts)
    text = decode(mel, asr)
"""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F
import torchaudio.transforms as T

from .audio import build_gl_transform, build_mel_transform, wav_to_logmel
from .config import Config, MelConfig
from .models import ASRModel, TTSModel
from .tokenizer import RobotTokenizer


def load_model(checkpoint_path: str, device: str | torch.device = "cpu"):
    """Load TTS and ASR models from a checkpoint file.

    Args:
        checkpoint_path: Path to a ``.pt`` checkpoint.
        device: Target device.

    Returns:
        ``(tts_model, asr_model, cfg)`` tuple.
    """
    device = torch.device(device)
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    cfg_dict = ckpt.get("cfg", {})
    cfg = Config(
        d_model=cfg_dict.get("d_model", 128),
        d_ff=cfg_dict.get("d_ff", 256),
        attention_heads=cfg_dict.get("attention_heads", 4),
        encoder_layers=cfg_dict.get("encoder_layers", 4),
        decoder_layers=cfg_dict.get("decoder_layers", 4),
        n_mels=cfg_dict.get("n_mels", 40),
        max_mel_len=cfg_dict.get("max_mel_len", 400),
    )

    tts = TTSModel(cfg).to(device)
    asr = ASRModel(cfg).to(device)

    # Strip torch.compile prefix if present
    def _strip(sd):
        return {k.replace("_orig_mod.", ""): v for k, v in sd.items()}

    tts.load_state_dict(_strip(ckpt["tts_state"]))
    asr.load_state_dict(_strip(ckpt["asr_state"]))
    tts.eval()
    asr.eval()
    return tts, asr, cfg


_tokenizer = RobotTokenizer()


def encode(text: str, tts_model: TTSModel, *,
           device: str | torch.device | None = None,
           return_waveform: bool = True,
           mel_cfg: MelConfig | None = None,
           gl_iters: int = 32) -> tuple[torch.Tensor, torch.Tensor | None]:
    """Encode text to a mel spectrogram (and optionally a waveform).

    Args:
        text: Input text (may contain robot command tokens).
        tts_model: A loaded ``TTSModel``.
        device: Override device.
        return_waveform: If ``True``, also run Griffin-Lim to produce audio.
        mel_cfg: Mel-spectrogram config for waveform reconstruction.
        gl_iters: Number of Griffin-Lim iterations.

    Returns:
        ``(mel, waveform)`` where ``waveform`` is ``None`` when
        ``return_waveform=False``.
    """
    if device is None:
        device = next(tts_model.parameters()).device

    tokens = _tokenizer.encode(text)
    if not tokens:
        return torch.zeros(1, 40, 1, device=device), None

    tok_t = torch.tensor([tokens], dtype=torch.long, device=device)
    tok_l = torch.tensor([len(tokens)], dtype=torch.long, device=device)

    tts_model.eval()
    with torch.no_grad():
        mel, _, dur_used = tts_model.infer(tok_t, tok_l)
        mel_len = dur_used.sum(dim=1).clamp(min=1, max=mel.size(2))
        mel = mel[:, :, : int(mel_len.item())]

    wav = None
    if return_waveform:
        if mel_cfg is None:
            mel_cfg = MelConfig()
        mel_transform = build_mel_transform(mel_cfg, target_device=device)
        mel_fb_pinv = torch.linalg.pinv(mel_transform.mel_scale.fb.T).to(device)
        gl = build_gl_transform(mel_cfg, gl_iters).to(device)
        with torch.no_grad(), torch.amp.autocast("cuda", enabled=False):
            x = mel.float()
            mel_lin = torch.expm1(x.clamp(max=10.0))
            power_spec = torch.matmul(mel_fb_pinv, mel_lin).clamp(min=0.0)
            mag_spec = power_spec.sqrt()
            wav = gl(mag_spec).squeeze(0)

    return mel.squeeze(0), wav


def decode(mel: torch.Tensor, asr_model: ASRModel, *,
           device: str | torch.device | None = None) -> str:
    """Decode a mel spectrogram to text.

    Args:
        mel: Mel spectrogram of shape ``(n_mels, T)`` or ``(1, n_mels, T)``.
        asr_model: A loaded ``ASRModel``.
        device: Override device.

    Returns:
        Decoded text string.
    """
    if device is None:
        device = next(asr_model.parameters()).device

    if mel.dim() == 2:
        mel = mel.unsqueeze(0)
    mel = mel.to(device)
    mel_len = torch.tensor([mel.size(2)], dtype=torch.long, device=device)

    asr_model.eval()
    with torch.no_grad():
        logits, sub_lens = asr_model(mel, mel_len)
    ids = asr_model.decode_greedy(logits, sub_lens)[0]
    return _tokenizer.decode(ids)


def decode_waveform(waveform: torch.Tensor, asr_model: ASRModel, *,
                    device: str | torch.device | None = None,
                    mel_cfg: MelConfig | None = None) -> str:
    """Decode a raw waveform to text.

    Args:
        waveform: 1-D float tensor of audio samples.
        asr_model: A loaded ``ASRModel``.
        device: Override device.
        mel_cfg: Mel-spectrogram config.

    Returns:
        Decoded text string.
    """
    if device is None:
        device = next(asr_model.parameters()).device

    if mel_cfg is None:
        mel_cfg = MelConfig()
    mel_transform = build_mel_transform(mel_cfg, target_device=device)
    mel = wav_to_logmel(waveform.to(device), mel_transform)
    return decode(mel, asr_model, device=device)


def end_to_end(text: str, tts_model: TTSModel, asr_model: ASRModel, *,
               device: str | torch.device | None = None) -> str:
    """Full encode-decode pipeline: text -> TTS mel -> ASR -> text.

    This is the primary test of the co-trained system.
    """
    mel, _ = encode(text, tts_model, device=device, return_waveform=False)
    return decode(mel, asr_model, device=device)
