"""Audio processing: mel frontend, Griffin-Lim, waveform and mel augmentation.

Provides:
    - ``build_mel_transform``: create a torchaudio MelSpectrogram transform
    - ``wav_to_logmel``: waveform -> log-mel spectrogram
    - ``mel_roundtrip``: log-mel -> GL -> waveform -> log-mel (non-differentiable)
    - ``WaveformChannelAugmentor``: GPU-batched waveform-domain channel simulation
    - ``MelAugmentor``: differentiable mel-domain augmentation
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio.transforms as T

from .config import MelConfig, AugConfig


# ---------------------------------------------------------------------------
#  Mel frontend
# ---------------------------------------------------------------------------

def build_mel_transform(mel_cfg: MelConfig, target_device=None):
    """Create a MelSpectrogram transform from config."""
    xform = T.MelSpectrogram(
        sample_rate=mel_cfg.sample_rate,
        n_fft=mel_cfg.n_fft,
        win_length=mel_cfg.win_length,
        hop_length=mel_cfg.hop_length,
        f_min=mel_cfg.f_min,
        f_max=mel_cfg.f_max,
        n_mels=mel_cfg.n_mels,
        power=mel_cfg.power,
        center=mel_cfg.center,
    )
    if target_device is not None:
        xform = xform.to(target_device)
    return xform


def wav_to_logmel(wav: torch.Tensor, mel_transform) -> torch.Tensor:
    """Convert waveform to log-mel spectrogram using ``log1p`` compression."""
    squeeze = False
    if wav.dim() == 1:
        wav = wav.unsqueeze(0)
        squeeze = True
    mel = mel_transform(wav.to(mel_transform.mel_scale.fb.device))
    logmel = torch.log1p(mel)
    if squeeze:
        logmel = logmel.squeeze(0)
    return logmel


# ---------------------------------------------------------------------------
#  Griffin-Lim round-trip
# ---------------------------------------------------------------------------

def build_gl_transform(mel_cfg: MelConfig, n_iter: int = 32, device=None):
    """Create a GriffinLim transform from config."""
    gl = T.GriffinLim(
        n_fft=mel_cfg.n_fft,
        win_length=mel_cfg.win_length,
        hop_length=mel_cfg.hop_length,
        power=1.0,
        n_iter=n_iter,
    )
    if device is not None:
        gl = gl.to(device)
    return gl


@torch.no_grad()
def mel_roundtrip(logmel: torch.Tensor, mel_fb_pinv: torch.Tensor,
                  gl_transform, wav_to_logmel_fn) -> torch.Tensor:
    """Non-differentiable Griffin-Lim round-trip: logmel -> GL -> wav -> logmel.

    Used as a self-consistency target so TTS output stays GL-invertible.

    Args:
        logmel: Log-mel spectrogram ``[B, n_mels, T]``.
        mel_fb_pinv: Pseudo-inverse of the mel filter bank.
        gl_transform: GriffinLim transform instance.
        wav_to_logmel_fn: Callable ``wav -> logmel``.

    Returns:
        Reconstructed log-mel spectrogram.
    """
    with torch.amp.autocast("cuda", enabled=False):
        x = logmel.float()
        mel_linear = torch.expm1(x.clamp(max=10.0))
        power_spec = torch.matmul(mel_fb_pinv, mel_linear).clamp(min=0.0)
        mag_spec = power_spec.sqrt()
        wav = gl_transform(mag_spec)
        return wav_to_logmel_fn(wav)


# ---------------------------------------------------------------------------
#  Waveform channel augmentor (GPU-batched, non-differentiable)
# ---------------------------------------------------------------------------

class WaveformChannelAugmentor(nn.Module):
    """Simulates a noisy acoustic channel on batched waveforms.

    Effects applied (each with independent coin-flip probability):
        1. Random gain
        2. Additive noise (white + pink + brown)
        3. Parametric EQ via FFT
        4. Synthetic reverb via FFT convolution
        5. Hard clipping
        6. Resample drift (sample-rate jitter)
        7. Automatic gain control (AGC)
    """

    def __init__(self, aug_cfg: AugConfig, mel_cfg: MelConfig):
        super().__init__()
        self.aug = aug_cfg
        self.sr = mel_cfg.sample_rate
        bc = torch.tensor([0.049922035, -0.095993537, 0.050612699, -0.004709510])
        self.register_buffer("pink_kernel", bc.flip(0).view(1, 1, -1))

    @torch.no_grad()
    def augment_batch(self, wav_padded, wav_lens, seed=0,
                      snr_range_override=None):
        """Apply channel augmentation to a batch of waveforms.

        Args:
            wav_padded: Padded waveforms ``[B, L]``.
            wav_lens: Actual lengths ``[B]``.
            seed: RNG seed for reproducibility.
            snr_range_override: Override the default SNR range ``(lo, hi)`` in dB.

        Returns:
            Tuple of ``(augmented_wav, wav_lens)``.
        """
        dev = wav_padded.device
        torch.manual_seed(seed)
        if dev.type == "cuda":
            torch.cuda.manual_seed(seed)

        wav = wav_padded.clone()
        B, L = wav.shape
        mask = torch.arange(L, device=dev).unsqueeze(0) < wav_lens.unsqueeze(1)
        mask_f = mask.float()

        # 1. Random gain
        lo, hi = self.aug.gain_db_range
        gain_db = lo + (hi - lo) * torch.rand(B, 1, device=dev)
        wav = wav * (10 ** (gain_db / 20.0))

        # 2. Additive noise
        snr_lo, snr_hi = snr_range_override or self.aug.noise_snr_range
        snr_db = snr_lo + (snr_hi - snr_lo) * torch.rand(B, 1, device=dev)
        sig_power = ((wav ** 2) * mask_f).sum(1, keepdim=True) / wav_lens.unsqueeze(1).float().clamp(min=1)
        sig_power = sig_power.clamp(min=1e-10)
        noise_power = sig_power / (10 ** (snr_db / 10))
        noise = torch.randn(B, L, device=dev) * noise_power.sqrt()

        # Pink noise component
        pink_coin = torch.rand(B, device=dev)
        if (pink_coin < self.aug.pink_noise_prob).any():
            pink_mask = (pink_coin < self.aug.pink_noise_prob).float().unsqueeze(1)
            white = torch.randn(B, L, device=dev)
            pink = F.conv1d(
                white.unsqueeze(1), self.pink_kernel,
                padding=self.pink_kernel.shape[-1] - 1,
            ).squeeze(1)[:, :L]
            pink_rms = (pink ** 2).mean(1, keepdim=True).sqrt().clamp(min=1e-8)
            pink = pink / pink_rms * noise_power.sqrt() * 0.5
            noise = noise + pink * pink_mask

        # Brown noise component
        brown_coin = torch.rand(B, device=dev)
        if (brown_coin < self.aug.brown_noise_prob).any():
            brown_mask = (brown_coin < self.aug.brown_noise_prob).float().unsqueeze(1)
            brown = torch.randn(B, L, device=dev).cumsum(dim=-1)
            brown_rms = (brown ** 2).mean(1, keepdim=True).sqrt().clamp(min=1e-8)
            brown = brown / brown_rms * noise_power.sqrt() * 0.3
            noise = noise + brown * brown_mask

        wav = wav + noise * mask_f

        # 3. Parametric EQ via FFT
        eq_coin = torch.rand(B, device=dev)
        if (eq_coin < self.aug.eq_prob).any():
            eq_mask = (eq_coin < self.aug.eq_prob).float().unsqueeze(1)
            W = torch.fft.rfft(wav, n=L, dim=1)
            n_bins = W.shape[1]
            freqs = torch.linspace(0, self.sr / 2, n_bins, device=dev)
            tilt_db = -4.0 + 8.0 * torch.rand(B, 1, device=dev)
            log_ratio = torch.log2((freqs + 1.0) / 1000.0).unsqueeze(0)
            tilt_gain = 10 ** (tilt_db * log_ratio / 20.0)
            peak_freq = 200.0 + 5800.0 * torch.rand(B, 1, device=dev)
            peak_gain_db = -6.0 + 12.0 * torch.rand(B, 1, device=dev)
            peak_q = 1.0 + 3.0 * torch.rand(B, 1, device=dev)
            rel_freq = (freqs.unsqueeze(0) - peak_freq) / (peak_freq / peak_q)
            bell = 10 ** (peak_gain_db / 20.0 * torch.exp(-0.5 * rel_freq ** 2))
            eq_filter = tilt_gain * bell
            combined = eq_filter * eq_mask + (1.0 - eq_mask)
            W = W * combined
            wav = torch.fft.irfft(W, n=L, dim=1)[:, :L] * mask_f

        # 4. Synthetic reverb via FFT convolution
        reverb_coin = torch.rand(B, device=dev)
        if (reverb_coin < self.aug.reverb_prob).any():
            reverb_mask = (reverb_coin < self.aug.reverb_prob).float().unsqueeze(1)
            ir_len = int(self.sr * 0.05)
            dlo, dhi = self.aug.reverb_decay_range
            decay = dlo + (dhi - dlo) * torch.rand(B, 1, device=dev)
            ir_noise = torch.randn(B, ir_len, device=dev)
            t_axis = torch.linspace(0, 1, ir_len, device=dev).unsqueeze(0)
            ir = ir_noise * torch.exp(-5.0 * decay * t_axis)
            ir[:, 0] = 1.0
            ir = ir / ir.abs().sum(1, keepdim=True).clamp(min=1e-8)
            n_fft_conv = 1
            while n_fft_conv < L + ir_len - 1:
                n_fft_conv *= 2
            W_wav = torch.fft.rfft(wav, n=n_fft_conv, dim=1)
            W_ir = torch.fft.rfft(ir, n=n_fft_conv, dim=1)
            wav_reverbed = torch.fft.irfft(W_wav * W_ir, n=n_fft_conv, dim=1)[:, :L]
            mix = 0.1 + 0.3 * torch.rand(B, 1, device=dev)
            wav_mixed = (1 - mix) * wav + mix * wav_reverbed
            wav = wav * (1 - reverb_mask) + wav_mixed * reverb_mask
            wav = wav * mask_f

        # 5. Hard clipping
        clip_coin = torch.rand(B, device=dev)
        if (clip_coin < self.aug.clip_prob).any():
            clip_mask = (clip_coin < self.aug.clip_prob).float().unsqueeze(1)
            clo, chi = self.aug.clip_threshold_range
            thresh = clo + (chi - clo) * torch.rand(B, 1, device=dev)
            peak = wav.abs().amax(1, keepdim=True).clamp(min=1e-8)
            wn_clipped = (wav / peak).clamp(-thresh, thresh) * peak
            wav = wav * (1 - clip_mask) + wn_clipped * clip_mask

        # 6. Resample drift
        drift_coin = torch.rand(B, device=dev)
        if (drift_coin < self.aug.resample_drift_prob).any():
            drift_mask = drift_coin < self.aug.resample_drift_prob
            for b in range(B):
                if drift_mask[b]:
                    dlo, dhi = self.aug.resample_drift_range
                    ratio = dlo + (dhi - dlo) * torch.rand(1, device=dev).item()
                    new_len = int(wav_lens[b].item() * ratio)
                    if 1 < new_len <= L:
                        resampled = F.interpolate(
                            wav[b : b + 1, : wav_lens[b]].unsqueeze(0),
                            size=new_len, mode="linear", align_corners=False,
                        ).squeeze(0).squeeze(0)
                        usable = min(new_len, L)
                        wav[b, :] = 0
                        wav[b, :usable] = resampled[:usable]

        # 7. AGC
        agc_coin = torch.rand(B, device=dev)
        if (agc_coin < self.aug.agc_prob).any():
            agc_mask = (agc_coin < self.aug.agc_prob).float().unsqueeze(1)
            target = 0.05 + 0.25 * torch.rand(B, 1, device=dev)
            rms = ((wav ** 2) * mask_f).sum(1, keepdim=True) / wav_lens.unsqueeze(1).float().clamp(min=1)
            rms = rms.sqrt().clamp(min=1e-8)
            wav_agc = wav * (target / rms)
            wav = wav * (1 - agc_mask) + wav_agc * agc_mask

        wav = wav * mask_f
        return wav, wav_lens.clone()


# ---------------------------------------------------------------------------
#  Mel-domain augmentor (differentiable)
# ---------------------------------------------------------------------------

class MelAugmentor(nn.Module):
    """Differentiable mel-spectrogram augmentation.

    Applied to TTS-generated mels before ASR to simulate acoustic variability:
        1. Additive mel noise
        2. SpecAugment (frequency + time masking)
        3. EQ tilt
        4. Impulsive burst
        5. Time shift
        6. Temporal blur
    """

    def __init__(self, aug_cfg: AugConfig):
        super().__init__()
        self.aug = aug_cfg

    def forward(self, X, seed=None, snr_override=None, snr_range_override=None):
        gen = torch.Generator()
        gen.manual_seed(seed if seed is not None else 0)
        B, Fr, T = X.shape

        # 1. Additive noise
        if snr_override is not None:
            snr_db = snr_override
        elif snr_range_override is not None:
            lo, hi = snr_range_override
            snr_db = lo + (hi - lo) * torch.rand(1, generator=gen).item()
        else:
            lo, hi = self.aug.mel_noise_snr_range
            snr_db = lo + (hi - lo) * torch.rand(1, generator=gen).item()
        sig_power = X.detach().pow(2).mean().clamp(min=1e-10)
        noise_std = (sig_power / (10 ** (snr_db / 10))).sqrt()
        noise = torch.randn(B, Fr, T, generator=gen).to(X.device) * noise_std
        X = X + noise

        # 2. SpecAugment
        if torch.rand(1, generator=gen).item() < self.aug.spec_augment_prob:
            fm_w = int(torch.randint(1, self.aug.freq_mask_max + 1, (1,), generator=gen).item())
            fm_s = int(torch.randint(0, max(Fr - fm_w, 1), (1,), generator=gen).item())
            f_mask = torch.ones(1, Fr, 1, device=X.device)
            f_mask[:, fm_s : fm_s + fm_w, :] = 0.0
            X = X * f_mask
            tm_w = int(torch.randint(1, min(self.aug.time_mask_max + 1, T), (1,), generator=gen).item())
            tm_s = int(torch.randint(0, max(T - tm_w, 1), (1,), generator=gen).item())
            t_mask = torch.ones(1, 1, T, device=X.device)
            t_mask[:, :, tm_s : tm_s + tm_w] = 0.0
            X = X * t_mask

        # 3. EQ tilt
        if torch.rand(1, generator=gen).item() < self.aug.mel_eq_tilt_prob:
            slope = -0.3 + 0.6 * torch.rand(1, generator=gen).item()
            offset = -0.1 + 0.2 * torch.rand(1, generator=gen).item()
            curve = torch.linspace(offset - slope / 2, offset + slope / 2, Fr, device=X.device)
            X = X * (1.0 + curve).view(1, Fr, 1)

        # 4. Impulsive burst
        if torch.rand(1, generator=gen).item() < self.aug.mel_burst_prob:
            bw = max(1, int(1 + 4 * torch.rand(1, generator=gen).item()))
            bs = int(torch.randint(0, max(T - bw, 1), (1,), generator=gen).item())
            burst_scale = 2.0 + 6.0 * torch.rand(1, generator=gen).item()
            burst = torch.zeros(B, Fr, T, device=X.device)
            burst[:, :, bs : bs + bw] = (
                torch.randn(B, Fr, bw, generator=gen).to(X.device) * burst_scale * noise_std
            )
            X = X + burst

        # 5. Time shift
        shift = int(
            torch.randint(
                -self.aug.mel_time_shift_max, self.aug.mel_time_shift_max + 1,
                (1,), generator=gen,
            ).item()
        )
        if shift != 0:
            X = torch.roll(X, shifts=shift, dims=-1)

        # 6. Temporal blur
        if torch.rand(1, generator=gen).item() < self.aug.mel_blur_prob:
            k = 3
            blur_k = torch.ones(1, 1, k, device=X.device) / k
            Xp = F.pad(X, (k // 2, k // 2), mode="replicate")
            Xf = Xp.reshape(B * Fr, 1, -1)
            X = F.conv1d(Xf, blur_k).reshape(B, Fr, T)

        return X
