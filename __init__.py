"""TTR -- Text-to-Radio: Co-trained TTS + ASR for acoustic robot communication.

A co-trained TTS + ASR system for reliable robot-to-robot text messaging
over acoustic channels.  See the README for usage instructions.
"""

__version__ = "1.0.0"

from .config import AugConfig, Config, CoTrainConfig, DataConfig, MelConfig
from .tokenizer import RobotTokenizer
from .models import ASRModel, TTSModel
from .inference import load_model, encode, decode, end_to_end

__all__ = [
    "Config", "MelConfig", "CoTrainConfig", "DataConfig", "AugConfig",
    "RobotTokenizer",
    "ASRModel", "TTSModel",
    "load_model", "encode", "decode", "end_to_end",
]
