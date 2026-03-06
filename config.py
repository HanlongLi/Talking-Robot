"""Configuration dataclasses for the TTR system."""

from dataclasses import dataclass, field, asdict
from typing import Tuple


@dataclass
class MelConfig:
    """Mel-spectrogram extraction parameters."""
    sample_rate: int = 16_000
    n_fft: int = 512
    hop_length: int = 160
    win_length: int = 512
    n_mels: int = 40
    f_min: float = 0.0
    f_max: float = 8000.0
    power: float = 2.0
    center: bool = True


@dataclass
class CoTrainConfig:
    """Co-training schedule and loss weights.

    Three-phase curriculum:
        Phase 0 (warmup):  CTC->TTS off, PS anchor on, GL roundtrip on.
        Phase 1 (ramp):    CTC->TTS ramps 0->1, PS decays 1->0.
        Phase 2 (full):    Pure co-training + roundtrip, no PS.
    """
    ctc_to_tts: bool = True
    warmup_steps: int = 1000
    ramp_steps: int = 3000
    lambda_ctc_tts: float = 1.0
    lambda_ctc_ref: float = 0.25
    lambda_mel: float = 1.0
    lambda_mel_floor: float = 0.0
    lambda_roundtrip: float = 0.5
    roundtrip_gl_iters: int = 32
    lambda_dur: float = 0.1
    lambda_energy: float = 0.01
    lambda_smoothness: float = 0.005
    lambda_bandlimit: float = 0.005


@dataclass
class DataConfig:
    """Dataset generation parameters."""
    use_ps_ref: bool = True
    dataset_size: int = 25_000
    train_ratio: float = 0.85
    val_ratio: float = 0.075
    test_ratio: float = 0.075
    min_seq_len: int = 3
    max_seq_len: int = 50


@dataclass
class AugConfig:
    """Channel augmentation parameters (waveform + mel domain)."""
    # Waveform-domain
    use_waveform_aug: bool = True
    wav_aug_prob: float = 0.9
    gain_db_range: Tuple[float, float] = (-15.0, 8.0)
    noise_snr_range: Tuple[float, float] = (-10.0, 30.0)
    pink_noise_prob: float = 0.5
    brown_noise_prob: float = 0.4
    reverb_prob: float = 0.5
    reverb_decay_range: Tuple[float, float] = (0.1, 0.8)
    clip_prob: float = 0.3
    clip_threshold_range: Tuple[float, float] = (0.2, 0.9)
    time_offset_ms_range: Tuple[float, float] = (0.0, 300.0)
    resample_drift_range: Tuple[float, float] = (0.96, 1.04)
    resample_drift_prob: float = 0.3
    eq_prob: float = 0.5
    agc_prob: float = 0.4
    # Mel-domain
    mel_noise_snr_range: Tuple[float, float] = (3.0, 30.0)
    spec_augment_prob: float = 0.6
    freq_mask_max: int = 7
    time_mask_max: int = 20
    mel_time_shift_max: int = 5
    mel_blur_prob: float = 0.3
    mel_eq_tilt_prob: float = 0.4
    mel_burst_prob: float = 0.2


@dataclass
class Config:
    """Top-level configuration for TTR training and inference."""
    mel: MelConfig = None
    cotrain: CoTrainConfig = None
    data: DataConfig = None
    aug: AugConfig = None

    # Vocabulary
    vocab_size: int = 128
    blank_id: int = 0
    pad_id: int = 1
    sos_id: int = 2
    eos_id: int = 3
    num_special: int = 6

    # Model dimensions
    latent_dim: int = 64
    d_model: int = 128
    d_ff: int = 256
    attention_heads: int = 4
    encoder_layers: int = 4
    decoder_layers: int = 4

    # Audio (derived from MelConfig at __post_init__)
    sample_rate: int = 16_000
    n_fft: int = 512
    hop_length: int = 160
    n_mels: int = 40
    max_mel_len: int = 400

    # Training
    batch_size: int = 128
    lr_tts: float = 8e-5
    lr_asr: float = 2e-4
    weight_decay: float = 1e-2
    num_epochs: int = 150
    warmup_lr_steps: int = 500
    grad_clip: float = 1.0
    seed: int = 42

    # EMA
    ema_decay: float = 0.999
    ema_start_epoch: int = 20

    # Early stopping
    early_stop_acc: float = 0.995
    early_stop_patience: int = 10

    def __post_init__(self):
        if self.mel is None:
            self.mel = MelConfig()
        if self.cotrain is None:
            self.cotrain = CoTrainConfig()
        if self.data is None:
            self.data = DataConfig()
        if self.aug is None:
            self.aug = AugConfig()
        self.sample_rate = self.mel.sample_rate
        self.n_fft = self.mel.n_fft
        self.hop_length = self.mel.hop_length
        self.n_mels = self.mel.n_mels

    def to_dict(self):
        return asdict(self)
