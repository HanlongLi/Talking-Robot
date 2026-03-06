"""Data generation, procedural synthesizer, and dataset utilities."""

import math
import random

import numpy as np
import torch
from torch.utils.data import Dataset

from .tokenizer import (
    BLANK_ID, PAD_ID, SPACE_ID, UNK_ID, NUM_SPECIAL,
    VOCAB_SIZE, LETTER_OFFSET, DIGIT_OFFSET, PUNCT_OFFSET,
    RESERVED_OFFSET, CMD_OFFSET,
    LETTERS, DIGITS, PUNCTUATION, ROBOT_COMMANDS,
)

# ---------------------------------------------------------------------------
#  Corpus templates
# ---------------------------------------------------------------------------

ROBOT_COMM_TEMPLATES = [
    "move forward {n} meters", "turn left {n} degrees",
    "turn right {n} degrees", "go to position {n},{m}",
    "navigate to waypoint {n}", "return to base",
    "stop immediately", "proceed to target", "follow the path",
    "avoid obstacle ahead", "move backward {n} steps",
    "rotate {n} degrees", "heading {n} degrees north",
    "speed {n} meters per second", "battery level {n} percent",
    "sensor {n} online", "distance to target {n} meters",
    "motor {n} temperature {m}", "obstacle detected at {n} meters",
    "signal strength {n} percent", "task {n} completed",
    "error code {n}", "position x {n} y {m}", "heading {n} degrees",
    "robot {n} ready", "robot {n} waiting",
    "formation alpha", "formation beta", "synchronize clocks",
    "begin patrol route {n}", "switch to channel {n}",
    "acknowledge message {n}", "request assistance",
    "area {n} clear", "target acquired", "mission complete",
    "<GO> <FWD> {n}", "<STOP> <WAIT>", "<ACK> message received",
    "<ALERT> obstacle at {n}", "<STATUS> battery {n}",
    "<READY> for task {n}", "<GRAB> object at {n},{m}",
    "<SCAN> sector {n}", "<FOLLOW> robot {n}",
    "<FORM> line formation", "<SYNC> start in {n}",
    "<REPORT> area {n} status",
    "waypoint {n} reached", "hold position for {n} seconds",
    "engage target at bearing {n}", "requesting backup at sector {n}",
    "all units report status", "switch to frequency {n}",
    "abort mission {n}", "commence landing sequence",
    "deploy sensor array {n}", "initiate scan pattern {n}",
    "<ROTATE> {n} degrees", "<LIFT> object {n}",
    "<LOWER> to position {n}", "<PUSH> forward {n}",
    "<PULL> backward {n}", "<OPEN> hatch {n}", "<CLOSE> valve {n}",
    "<LEAD> formation {n}", "<FOLLOW> leader",
    "<CHARGE> at station {n}", "<IDLE> until signal",
]

GENERAL_ENGLISH = [
    "hello", "yes", "no", "ok", "help", "the quick brown fox",
    "all systems nominal", "ready to proceed", "waiting for input",
    "check sensors", "update firmware", "low power mode",
    "high priority alert", "data received", "connection established",
    "link quality good", "calibration complete", "test passed",
    "reboot system", "standby mode active", "emergency shutdown",
    "clear the area", "hold position", "confirm receipt",
    "negative contact", "all units report", "green light go",
    "target locked", "signal lost", "retry connection",
    "mission parameters loaded", "coordinates locked in",
    "perimeter secure", "requesting permission to proceed",
    "all clear on channel {n}", "interference detected",
    "switching to backup frequency", "telemetry nominal",
    "payload delivered", "rendezvous at checkpoint {n}",
]


def generate_training_corpus(n_samples: int = 25000, seed: int = 42) -> list:
    """Generate a mixed corpus of robot commands and general English.

    Distribution: 55% templates, 20% general English, 15% pure command
    tokens, 10% random character sequences.

    Args:
        n_samples: Total number of text samples.
        seed: RNG seed.

    Returns:
        List of text strings.
    """
    rng = random.Random(seed)
    corpus = []
    nt = int(n_samples * 0.55)
    ne = int(n_samples * 0.20)
    nc = int(n_samples * 0.15)
    nr = n_samples - nt - ne - nc

    for _ in range(nt):
        t = rng.choice(ROBOT_COMM_TEMPLATES)
        corpus.append(
            t.replace("{n}", str(rng.randint(0, 999)))
             .replace("{m}", str(rng.randint(0, 999)))
        )
    for _ in range(ne):
        p = rng.choice(GENERAL_ENGLISH)
        p = p.replace("{n}", str(rng.randint(0, 99)))
        if rng.random() < 0.3:
            p += f" {rng.randint(0, 99)}"
        corpus.append(p)
    cl = list(ROBOT_COMMANDS.keys())
    for _ in range(nc):
        corpus.append(" ".join(rng.choice(cl) for _ in range(rng.randint(1, 5))))
    ac = LETTERS + DIGITS + PUNCTUATION
    for _ in range(nr):
        s = "".join(rng.choice(ac) for _ in range(rng.randint(3, 25)))
        words = []
        i = 0
        while i < len(s):
            wl = rng.randint(1, 5)
            words.append(s[i : i + wl])
            i += wl
        corpus.append(" ".join(words))

    rng.shuffle(corpus)
    return corpus


# ---------------------------------------------------------------------------
#  Procedural synthesizer (PS)
# ---------------------------------------------------------------------------

class StructuredSynthesizer:
    """Deterministic procedural waveform synthesizer.

    Maps each token to a unique three-harmonic tone using a fixed frequency
    assignment. Vowels, consonants, digits, punctuation, and robot commands
    each have distinct frequency bands so the resulting spectrograms are
    separable by an ASR model.
    """

    VOWELS = set("aeiou")
    CONSONANTS = set("bcdfghjklmnpqrstvwxyz")

    def __init__(self, sample_rate: int = 16000):
        self.sr = sample_rate
        self.dur = 0.06
        self.n_samp = int(self.dur * self.sr)
        rng = np.random.RandomState(42)
        self.freqs = np.zeros((VOCAB_SIZE, 3))
        self.amps = np.zeros((VOCAB_SIZE, 3))
        self.phases = np.zeros((VOCAB_SIZE, 3))
        for t in range(VOCAB_SIZE):
            self._assign(t, rng)

    def _assign(self, tid, rng):
        if tid < NUM_SPECIAL:
            self.freqs[tid] = [100, 200, 300]
            self.amps[tid] = [0.01, 0.01, 0.01]
            if tid == SPACE_ID:
                self.freqs[tid] = [150, 300, 450]
                self.amps[tid] = [0.05, 0.03, 0.02]
            if tid == UNK_ID:
                self.freqs[tid] = [500, 1000, 1500]
                self.amps[tid] = [0.3, 0.2, 0.1]
            return
        if LETTER_OFFSET <= tid < LETTER_OFFSET + 26:
            ch = LETTERS[tid - LETTER_OFFSET]
            if ch in self.VOWELS:
                vi = "aeiou".index(ch)
                b = 250 + vi * 120
                self.freqs[tid] = [b, b * 2.2, b * 3.1]
                self.amps[tid] = [0.9, 0.5, 0.25]
            else:
                ci = sorted(self.CONSONANTS).index(ch)
                b = 800 + ci * 80
                self.freqs[tid] = [b, b * 1.5, b * 2.3]
                self.amps[tid] = [0.7, 0.4, 0.2]
            self.freqs[tid] += rng.uniform(-20, 20, 3)
            self.phases[tid] = rng.uniform(0, 2 * np.pi, 3)
            return
        if DIGIT_OFFSET <= tid < DIGIT_OFFSET + 10:
            d = tid - DIGIT_OFFSET
            b = 2500 + d * 100
            self.freqs[tid] = [b, b * 1.3, b * 1.7]
            self.amps[tid] = [0.8, 0.45, 0.2]
            self.phases[tid] = rng.uniform(0, 2 * np.pi, 3)
            return
        if PUNCT_OFFSET <= tid < PUNCT_OFFSET + 16:
            idx = tid - PUNCT_OFFSET
            b = 3500 + idx * 30
            self.freqs[tid] = [b, b * 0.5, b * 1.5]
            self.amps[tid] = [0.6, 0.3, 0.15]
            self.phases[tid] = rng.uniform(0, 2 * np.pi, 3)
            return
        if RESERVED_OFFSET <= tid < CMD_OFFSET:
            idx = tid - RESERVED_OFFSET
            b = 1500 + idx * 50
            self.freqs[tid] = [b, b * 1.4, b * 2.0]
            self.amps[tid] = [0.5, 0.3, 0.15]
            self.phases[tid] = rng.uniform(0, 2 * np.pi, 3)
            return
        if CMD_OFFSET <= tid < VOCAB_SIZE:
            ci = tid - CMD_OFFSET
            g = (1 + np.sqrt(5)) / 2
            self.freqs[tid] = [
                300 + (ci * g * 83) % 3500,
                500 + ((ci + 7) * g * 97) % 3200,
                200 + ((ci + 13) * g * 71) % 3600,
            ]
            self.amps[tid] = [0.85, 0.55, 0.35]
            self.phases[tid] = [ci * np.pi / 22, ci * np.pi / 11, ci * np.pi / 7]

    def synthesize(self, token_seq: list) -> torch.Tensor:
        """Synthesize a waveform from a sequence of token IDs.

        Args:
            token_seq: List of integer token IDs.

        Returns:
            1-D float tensor of audio samples at ``self.sr`` Hz.
        """
        t = torch.linspace(0, self.dur, self.n_samp)
        fade = min(int(0.005 * self.sr), self.n_samp // 4)
        fi = torch.linspace(0, 1, fade)
        fo = torch.linspace(1, 0, fade)
        chunks = []
        for tok in token_seq:
            w = sum(
                float(self.amps[tok, h])
                * torch.sin(
                    2 * math.pi * float(self.freqs[tok, h]) * t
                    + float(self.phases[tok, h])
                )
                for h in range(3)
            )
            w[:fade] *= fi
            w[-fade:] *= fo
            chunks.append(w)
        return torch.cat(chunks) if chunks else torch.zeros(1)


# ---------------------------------------------------------------------------
#  Dataset and collation
# ---------------------------------------------------------------------------

class RobotCommDataset(Dataset):
    """Dataset of (tokens, waveform) pairs.

    Optionally pre-caches PS waveforms for all samples at construction time.

    Args:
        text_token_pairs: List of ``(text, token_ids)`` tuples.
        synthesizer: Optional ``StructuredSynthesizer`` for waveform generation.
    """

    def __init__(self, text_token_pairs, synthesizer=None):
        self.pairs = text_token_pairs
        self._wav_cache = {}
        if synthesizer is not None:
            for i, (_, tokens) in enumerate(text_token_pairs):
                self._wav_cache[i] = synthesizer.synthesize(tokens)

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        _, tokens = self.pairs[idx]
        token_tensor = torch.tensor(tokens, dtype=torch.long)
        if idx in self._wav_cache:
            wav = self._wav_cache[idx]
            return token_tensor, wav, wav.shape[0]
        return token_tensor, torch.zeros(1), 0


def collate_fn(batch):
    """Collate a batch of ``(tokens, wav, wav_len)`` tuples into padded tensors."""
    tokens_list, wavs, wav_lens_raw = zip(*batch)
    tok_lens = [len(t) for t in tokens_list]
    max_tok = max(tok_lens)
    tokens_padded = torch.full((len(tokens_list), max_tok), PAD_ID, dtype=torch.long)
    for i, t in enumerate(tokens_list):
        tokens_padded[i, : len(t)] = t
    tok_lens_t = torch.tensor(tok_lens, dtype=torch.long)
    wav_lens = torch.tensor(wav_lens_raw, dtype=torch.long)
    if wav_lens.max().item() > 0:
        max_wav = wav_lens.max().item()
        wav_padded = torch.zeros(len(wavs), max_wav)
        for i, w in enumerate(wavs):
            L = wav_lens[i].item()
            if L > 0:
                wav_padded[i, :L] = w[:L]
    else:
        wav_padded = None
        wav_lens = None
    return tokens_padded, tok_lens_t, wav_padded, wav_lens


def compute_target_durations(tok_lens, mel_lens, max_tok_len):
    """Compute integer duration targets from token and mel lengths.

    Distributes mel frames evenly across tokens with remainder to the first
    few tokens.
    """
    B = tok_lens.size(0)
    dev = tok_lens.device
    durations = torch.zeros(B, max_tok_len, dtype=torch.long, device=dev)
    safe_tok = tok_lens.clamp(min=1).float()
    base = (mel_lens.float() / safe_tok).long()
    rem = (mel_lens - base * tok_lens.clamp(min=1)).clamp(min=0)
    pos = torch.arange(max_tok_len, device=dev).unsqueeze(0)
    tok_mask = pos < tok_lens.unsqueeze(1)
    durations[tok_mask] = base.unsqueeze(1).expand_as(durations)[tok_mask]
    rem_mask = tok_mask & (pos < rem.unsqueeze(1))
    durations[rem_mask] += 1
    return durations
