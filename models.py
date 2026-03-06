"""TTS and ASR model architectures.

TTS: FastSpeech2-Nano (~1.18M params)
    tokens -> encoder -> duration predictor -> length regulator -> decoder -> mel

ASR: Conformer-Tiny (~938K params)
    mel -> conv subsampling -> conformer blocks -> CTC projection
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
#  Shared components
# ---------------------------------------------------------------------------

class PositionalEncoding(nn.Module):
    """Sinusoidal positional encoding."""

    def __init__(self, d_model: int, max_len: int = 2000, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(max_len).unsqueeze(1).float()
        div = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.dropout(x + self.pe[:, : x.size(1)])


# ---------------------------------------------------------------------------
#  ASR components
# ---------------------------------------------------------------------------

class ConvSubsampling(nn.Module):
    """Two-layer strided convolution that reduces the time dimension by 4x."""

    def __init__(self, n_mels: int, d_model: int):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(1, 32, 3, stride=2, padding=1), nn.GELU(),
            nn.Conv2d(32, 32, 3, stride=2, padding=1), nn.GELU(),
        )
        self.proj = nn.Linear(32 * (n_mels // 4), d_model)

    def forward(self, mel: torch.Tensor) -> torch.Tensor:
        x = self.conv(mel.unsqueeze(1))
        B, C, Fr, T_ = x.shape
        return self.proj(x.permute(0, 3, 1, 2).reshape(B, T_, C * Fr))


class DepthwiseSeparableConv(nn.Module):
    """Depthwise-separable convolution used inside Conformer blocks."""

    def __init__(self, d: int, k: int = 15, drop: float = 0.1):
        super().__init__()
        self.depthwise = nn.Conv1d(d, d, k, padding=k // 2, groups=d)
        self.pointwise = nn.Conv1d(d, d, 1)
        self.norm = nn.LayerNorm(d)
        self.act = nn.GELU()
        self.drop = nn.Dropout(drop)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        r = x
        x = self.pointwise(self.depthwise(x.transpose(1, 2))).transpose(1, 2)
        return self.drop(self.act(self.norm(x))) + r


class ConformerBlock(nn.Module):
    """Single Conformer block: FF -> MHSA -> Conv -> FF."""

    def __init__(self, d: int, nh: int, dff: int, k: int = 15):
        super().__init__()
        self.ff1 = nn.Sequential(
            nn.LayerNorm(d), nn.Linear(d, dff), nn.GELU(),
            nn.Dropout(0.1), nn.Linear(dff, d), nn.Dropout(0.1),
        )
        self.ff2 = nn.Sequential(
            nn.LayerNorm(d), nn.Linear(d, dff), nn.GELU(),
            nn.Dropout(0.1), nn.Linear(dff, d), nn.Dropout(0.1),
        )
        self.attn_norm = nn.LayerNorm(d)
        self.attn = nn.MultiheadAttention(d, nh, dropout=0.1, batch_first=True)
        self.ad = nn.Dropout(0.1)
        self.conv = DepthwiseSeparableConv(d, k)
        self.final_norm = nn.LayerNorm(d)

    def forward(self, x: torch.Tensor, mask=None) -> torch.Tensor:
        x = x + 0.5 * self.ff1(x)
        r = x
        xn = self.attn_norm(x)
        xa, _ = self.attn(xn, xn, xn, key_padding_mask=mask)
        x = r + self.ad(xa)
        x = self.conv(x)
        x = x + 0.5 * self.ff2(x)
        return self.final_norm(x)


class ASRModel(nn.Module):
    """Conformer-Tiny ASR encoder with CTC projection.

    Args:
        cfg: A Config (or any object) with attributes ``n_mels``, ``d_model``,
             ``attention_heads``, ``d_ff``, ``decoder_layers``, ``vocab_size``,
             and ``blank_id``.
    """

    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.subsampling = ConvSubsampling(cfg.n_mels, cfg.d_model)
        self.pos_enc = PositionalEncoding(cfg.d_model)
        self.blocks = nn.ModuleList(
            [ConformerBlock(cfg.d_model, cfg.attention_heads, cfg.d_ff)
             for _ in range(cfg.decoder_layers)]
        )
        self.proj = nn.Linear(cfg.d_model, cfg.vocab_size)

    def forward(self, mel, mel_lens=None):
        x = self.pos_enc(self.subsampling(mel))
        T_ = x.size(1)
        mask = sub_lens = None
        if mel_lens is not None:
            sub_lens = (mel_lens // 4).clamp(min=1)
            mask = torch.arange(T_, device=x.device).unsqueeze(0) >= sub_lens.unsqueeze(1)
        for block in self.blocks:
            x = block(x, mask)
        return self.proj(x), sub_lens

    def decode_greedy(self, logits, lengths=None):
        """CTC greedy decoding: collapse repeated tokens and remove blanks."""
        preds = logits.argmax(-1)
        results = []
        for b in range(preds.size(0)):
            sl = lengths[b].item() if lengths is not None else preds.size(1)
            collapsed = []
            prev = None
            for tok in preds[b, :sl].tolist():
                if tok != prev:
                    if tok != self.cfg.blank_id:
                        collapsed.append(tok)
                    prev = tok
            results.append(collapsed)
        return results


# ---------------------------------------------------------------------------
#  TTS components
# ---------------------------------------------------------------------------

class DurationPredictor(nn.Module):
    """Two-layer 1-D convolution that predicts log-duration per token."""

    def __init__(self, d: int):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(d, d, 3, padding=1), nn.GELU(), nn.BatchNorm1d(d),
            nn.Conv1d(d, d, 3, padding=1), nn.GELU(), nn.BatchNorm1d(d),
        )
        self.proj = nn.Linear(d, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.proj(self.conv(x.transpose(1, 2)).transpose(1, 2)).squeeze(-1)


class LengthRegulator(nn.Module):
    """Expand encoder hidden states according to predicted durations."""

    @torch.compiler.disable
    def forward(self, x, durations, max_len=None):
        B, T, D = x.shape
        dur = durations.long()
        if max_len is None:
            max_len = int(dur.sum(dim=1).max().item())
        dur_clamped = dur.clamp(min=0)
        out = torch.zeros(B, max_len, D, device=x.device, dtype=x.dtype)
        for b in range(B):
            d = dur_clamped[b]
            total = d.sum().item()
            if total == 0:
                continue
            idx = torch.repeat_interleave(torch.arange(T, device=x.device), d)
            usable = min(idx.size(0), max_len)
            out[b, :usable] = x[b, idx[:usable]]
        return out


class TTSModel(nn.Module):
    """FastSpeech2-Nano TTS: tokens -> log-mel spectrogram.

    Args:
        cfg: A Config (or any object) with attributes ``vocab_size``, ``d_model``,
             ``pad_id``, ``attention_heads``, ``d_ff``, ``encoder_layers``,
             ``n_mels``, and ``max_mel_len``.
    """

    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.tok_emb = nn.Embedding(cfg.vocab_size, cfg.d_model, padding_idx=cfg.pad_id)
        self.pos_enc = PositionalEncoding(cfg.d_model)
        el = nn.TransformerEncoderLayer(
            d_model=cfg.d_model, nhead=cfg.attention_heads,
            dim_feedforward=cfg.d_ff, activation="gelu",
            dropout=0.1, batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(el, num_layers=cfg.encoder_layers)
        self.dur_pred = DurationPredictor(cfg.d_model)
        self.length_reg = LengthRegulator()
        dl = nn.TransformerEncoderLayer(
            d_model=cfg.d_model, nhead=cfg.attention_heads,
            dim_feedforward=cfg.d_ff, activation="gelu",
            dropout=0.1, batch_first=True,
        )
        self.decoder = nn.TransformerEncoder(dl, num_layers=cfg.encoder_layers)
        self.mel_proj = nn.Linear(cfg.d_model, cfg.n_mels)

    def forward(self, tokens, tok_lens, target_durations=None, max_mel_len=None):
        B, T = tokens.shape
        tok_mask = (
            torch.arange(T, device=tokens.device).unsqueeze(0) >= tok_lens.unsqueeze(1)
        )
        x = self.pos_enc(self.tok_emb(tokens))
        x = self.encoder(x, src_key_padding_mask=tok_mask)
        ldp = self.dur_pred(x)
        if target_durations is not None:
            du = target_durations
        else:
            du = torch.clamp(torch.round(torch.exp(ldp) - 1), min=1).long()
            du = du * (~tok_mask).long()
        xr = self.pos_enc(self.length_reg(x, du, max_len=max_mel_len))
        return self.mel_proj(self.decoder(xr)).transpose(1, 2), ldp, du

    def infer(self, tokens, tok_lens):
        """Run inference with predicted durations (no ground-truth)."""
        return self.forward(tokens, tok_lens, max_mel_len=self.cfg.max_mel_len)
