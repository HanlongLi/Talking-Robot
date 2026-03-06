"""Co-training loop for TTS and ASR.

Usage::

    python -m ttr.train [--epochs N] [--batch-size B] [--save-dir DIR]
"""

from __future__ import annotations

import argparse
import math
import os
import time
from collections import defaultdict
from dataclasses import asdict

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, random_split
from tqdm.auto import tqdm

from .audio import (
    MelAugmentor,
    WaveformChannelAugmentor,
    build_gl_transform,
    build_mel_transform,
    mel_roundtrip,
    wav_to_logmel,
)
from .config import Config
from .data import (
    RobotCommDataset,
    StructuredSynthesizer,
    collate_fn,
    compute_target_durations,
    generate_training_corpus,
)
from .losses import (
    CoTrainSchedule,
    EMA,
    TTSRegularizers,
    compute_ctc_loss,
)
from .models import ASRModel, TTSModel
from .tokenizer import RobotTokenizer


# ---------------------------------------------------------------------------
#  Validation helpers
# ---------------------------------------------------------------------------


@torch.no_grad()
def validate(asr_model, tts_model, val_loader, device, *,
             use_amp=True, amp_dtype=torch.bfloat16,
             use_ema=False, ema_asr=None, ema_tts=None):
    """TTS -> mel -> ASR accuracy on the validation set."""
    asr_base = getattr(asr_model, "_orig_mod", asr_model)
    tts_base = getattr(tts_model, "_orig_mod", tts_model)
    if use_ema and ema_asr is not None:
        ema_asr.apply_shadow(asr_base)
        ema_tts.apply_shadow(tts_base)
    asr_model.eval()
    tts_model.eval()
    correct = total = exact = n = 0
    for tokens, tok_lens, wav_ref, wav_lens in val_loader:
        tokens = tokens.to(device, non_blocking=True)
        tok_lens = tok_lens.to(device, non_blocking=True)
        with torch.amp.autocast("cuda", enabled=use_amp, dtype=amp_dtype):
            mel_pred, _, dur_used = tts_model.infer(tokens, tok_lens)
            mel_lens = dur_used.sum(dim=1).clamp(min=1, max=mel_pred.size(2))
            logits, sub_lens = asr_model(mel_pred, mel_lens)
        decoded = asr_model.decode_greedy(logits, sub_lens)
        for b in range(len(decoded)):
            ref = tokens[b, : tok_lens[b]].tolist()
            hyp = decoded[b]
            correct += sum(1 for r, h in zip(ref, hyp) if r == h)
            total += len(ref)
            exact += int(ref == hyp)
            n += 1
    if use_ema and ema_asr is not None:
        ema_asr.restore(asr_base)
        ema_tts.restore(tts_base)
    return correct / max(total, 1), exact / max(n, 1)


@torch.no_grad()
def validate_ps(asr_model, val_loader, device, wav_to_logmel_fn, *,
                use_amp=True, amp_dtype=torch.bfloat16,
                use_ema=False, ema_asr=None,
                wav_augmentor=None, noisy=False):
    """PS -> wav -> mel -> ASR (ceiling measurement)."""
    asr_base = getattr(asr_model, "_orig_mod", asr_model)
    if use_ema and ema_asr is not None:
        ema_asr.apply_shadow(asr_base)
    asr_model.eval()
    correct = total = exact = n = 0
    for tokens, tok_lens, wav_ref, wav_lens in val_loader:
        if wav_ref is None:
            continue
        tokens = tokens.to(device, non_blocking=True)
        tok_lens = tok_lens.to(device, non_blocking=True)
        wav_gpu = wav_ref.to(device, non_blocking=True)
        wav_lens_gpu = wav_lens.to(device, non_blocking=True)
        if noisy and wav_augmentor is not None:
            wav_gpu, wav_lens_gpu = wav_augmentor.augment_batch(
                wav_gpu, wav_lens_gpu, seed=12345,
                snr_range_override=(0.0, 15.0),
            )
        hop = 160  # default hop_length
        with torch.amp.autocast("cuda", enabled=use_amp, dtype=amp_dtype):
            mels = wav_to_logmel_fn(wav_gpu)
            mel_lens = (wav_lens_gpu // hop + 1).clamp(max=mels.size(2))
            logits, sub_lens = asr_model(mels, mel_lens)
        decoded = asr_model.decode_greedy(logits, sub_lens)
        for b in range(len(decoded)):
            ref = tokens[b, : tok_lens[b]].tolist()
            hyp = decoded[b]
            correct += sum(1 for r, h in zip(ref, hyp) if r == h)
            total += len(ref)
            exact += int(ref == hyp)
            n += 1
    if use_ema and ema_asr is not None:
        ema_asr.restore(asr_base)
    return correct / max(total, 1), exact / max(n, 1)


@torch.no_grad()
def validate_roundtrip(asr_model, tts_model, val_loader, device,
                       mel_roundtrip_fn, *,
                       use_amp=True, amp_dtype=torch.bfloat16,
                       use_ema=False, ema_asr=None, ema_tts=None):
    """TTS -> mel -> GL -> wav -> mel -> ASR (deployment path)."""
    asr_base = getattr(asr_model, "_orig_mod", asr_model)
    tts_base = getattr(tts_model, "_orig_mod", tts_model)
    if use_ema and ema_asr is not None:
        ema_asr.apply_shadow(asr_base)
        ema_tts.apply_shadow(tts_base)
    asr_model.eval()
    tts_model.eval()
    correct = total = exact = n = 0
    rt_l1_sum = 0.0
    rt_n = 0
    for tokens, tok_lens, wav_ref, wav_lens in val_loader:
        tokens = tokens.to(device, non_blocking=True)
        tok_lens = tok_lens.to(device, non_blocking=True)
        with torch.amp.autocast("cuda", enabled=use_amp, dtype=amp_dtype):
            mel_pred, _, dur_used = tts_model.infer(tokens, tok_lens)
            mel_lens_pred = dur_used.sum(dim=1).clamp(min=1, max=mel_pred.size(2))
        mel_rt = mel_roundtrip_fn(mel_pred)
        min_t = min(mel_pred.shape[2], mel_rt.shape[2])
        rt_l1 = F.l1_loss(
            mel_pred[:, :, :min_t].float(), mel_rt[:, :, :min_t].float()
        ).item()
        rt_l1_sum += rt_l1
        rt_n += 1
        mel_rt_lens = mel_lens_pred.clamp(max=mel_rt.size(2))
        with torch.amp.autocast("cuda", enabled=use_amp, dtype=amp_dtype):
            logits, sub_lens = asr_model(mel_rt, mel_rt_lens)
        decoded = asr_model.decode_greedy(logits, sub_lens)
        for b in range(len(decoded)):
            ref = tokens[b, : tok_lens[b]].tolist()
            hyp = decoded[b]
            correct += sum(1 for r, h in zip(ref, hyp) if r == h)
            total += len(ref)
            exact += int(ref == hyp)
            n += 1
    if use_ema and ema_asr is not None:
        ema_asr.restore(asr_base)
        ema_tts.restore(tts_base)
    return correct / max(total, 1), exact / max(n, 1), rt_l1_sum / max(rt_n, 1)


# ---------------------------------------------------------------------------
#  Checkpoint helpers
# ---------------------------------------------------------------------------


def _strip_compile_prefix(sd: dict) -> dict:
    return {k.replace("_orig_mod.", ""): v for k, v in sd.items()}


def load_checkpoint(path, asr_model, tts_model, device, *,
                    optimizer_tts=None, optimizer_asr=None,
                    ema_asr=None, ema_tts=None):
    """Load a checkpoint; returns ``(global_step, epoch, val_acc)``."""
    if not os.path.exists(path):
        return 0, 0, 0.0
    ckpt = torch.load(path, map_location=device, weights_only=False)
    asr_base = getattr(asr_model, "_orig_mod", asr_model)
    tts_base = getattr(tts_model, "_orig_mod", tts_model)
    asr_base.load_state_dict(_strip_compile_prefix(ckpt.get("asr_state", {})))
    tts_base.load_state_dict(_strip_compile_prefix(ckpt.get("tts_state", {})))
    if optimizer_tts and "opt_tts_state" in ckpt:
        optimizer_tts.load_state_dict(ckpt["opt_tts_state"])
    if optimizer_asr and "opt_asr_state" in ckpt:
        optimizer_asr.load_state_dict(ckpt["opt_asr_state"])
    if ema_asr and "ema_asr_state" in ckpt:
        ema_asr.load_state_dict(ckpt["ema_asr_state"])
    if ema_tts and "ema_tts_state" in ckpt:
        ema_tts.load_state_dict(ckpt["ema_tts_state"])
    step = ckpt.get("step", 0)
    epoch = ckpt.get("epoch", 0)
    val_acc = ckpt.get("val_acc", 0.0)
    return step, epoch, val_acc


# ---------------------------------------------------------------------------
#  Main training entry point
# ---------------------------------------------------------------------------


def train(cfg: Config | None = None, save_dir: str = "./checkpoints",
          resume_path: str | None = None):
    """Run the full co-training pipeline.

    Args:
        cfg: Configuration object.  Uses defaults if ``None``.
        save_dir: Directory for checkpoint files.
        resume_path: Path to a checkpoint to resume from.
    """
    if cfg is None:
        cfg = Config()

    # ── Device & AMP ──
    assert torch.cuda.is_available(), "CUDA is required for training."
    device = torch.device("cuda")
    compute_cap = torch.cuda.get_device_capability(0)
    if compute_cap[0] >= 8:
        amp_dtype = torch.bfloat16
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
    else:
        amp_dtype = torch.float16
    use_amp = True

    os.makedirs(save_dir, exist_ok=True)
    torch.manual_seed(cfg.seed)
    torch.cuda.manual_seed_all(cfg.seed)
    np.random.seed(cfg.seed)
    import random as _random
    _random.seed(cfg.seed)

    # ── Tokenizer + data ──
    tokenizer = RobotTokenizer()
    corpus = generate_training_corpus(cfg.data.dataset_size)
    tokenized = [tokenizer.encode(msg) for msg in corpus]
    valid_pairs = [
        (msg, toks)
        for msg, toks in zip(corpus, tokenized)
        if cfg.data.min_seq_len <= len(toks) <= cfg.data.max_seq_len
    ]
    synth = StructuredSynthesizer(sample_rate=cfg.sample_rate)
    full_ds = RobotCommDataset(valid_pairs, synthesizer=synth)

    n_train = int(len(full_ds) * cfg.data.train_ratio)
    n_val = int(len(full_ds) * (1 - cfg.data.train_ratio) * 0.5)
    n_test = len(full_ds) - n_train - n_val
    train_ds, val_ds, _ = random_split(
        full_ds, [n_train, n_val, n_test],
        generator=torch.Generator().manual_seed(cfg.seed),
    )
    train_loader = DataLoader(
        train_ds, batch_size=cfg.batch_size, shuffle=True,
        collate_fn=collate_fn, num_workers=0, pin_memory=True,
    )
    val_loader = DataLoader(
        val_ds, batch_size=cfg.batch_size, shuffle=False,
        collate_fn=collate_fn, num_workers=0, pin_memory=True,
    )
    print(f"Data: {len(valid_pairs)} samples, "
          f"train={n_train}, val={n_val}, test={n_test}")
    print(f"Batches/epoch: {len(train_loader)} (BS={cfg.batch_size})")

    # ── Audio transforms ──
    mel_transform = build_mel_transform(cfg.mel, target_device=device)

    mel_fb_pinv = torch.linalg.pinv(mel_transform.mel_scale.fb.T).to(device)
    gl_transform = build_gl_transform(cfg.mel, cfg.cotrain.roundtrip_gl_iters).to(device)

    def _wav_to_logmel(wav):
        return wav_to_logmel(wav, mel_transform)

    def _mel_roundtrip(logmel):
        return mel_roundtrip(logmel, mel_fb_pinv, gl_transform, _wav_to_logmel)

    wav_augmentor = WaveformChannelAugmentor(cfg.aug, cfg.mel).to(device)
    mel_augmentor = MelAugmentor(cfg.aug)

    # ── Models ──
    asr_model = ASRModel(cfg).to(device)
    tts_model = TTSModel(cfg).to(device)
    n_asr = sum(p.numel() for p in asr_model.parameters())
    n_tts = sum(p.numel() for p in tts_model.parameters())
    print(f"ASR: {n_asr:,} params | TTS: {n_tts:,} params")

    if hasattr(torch, "compile"):
        try:
            asr_model = torch.compile(asr_model, mode="default")
            tts_model = torch.compile(tts_model, mode="default")
            print("torch.compile enabled")
        except Exception:
            pass

    # ── Optimizers & LR schedule ──
    optimizer_tts = torch.optim.AdamW(
        tts_model.parameters(), lr=cfg.lr_tts, weight_decay=cfg.weight_decay,
    )
    optimizer_asr = torch.optim.AdamW(
        asr_model.parameters(), lr=cfg.lr_asr, weight_decay=cfg.weight_decay,
    )
    grad_scaler = torch.amp.GradScaler(
        "cuda", enabled=(use_amp and amp_dtype == torch.float16),
    )
    total_steps_est = cfg.num_epochs * len(train_loader)

    def lr_lambda_fn(step):
        if step < cfg.warmup_lr_steps:
            return step / max(cfg.warmup_lr_steps, 1)
        progress = (step - cfg.warmup_lr_steps) / max(
            total_steps_est - cfg.warmup_lr_steps, 1
        )
        return max(0.05, 0.5 * (1 + math.cos(math.pi * progress)))

    sched_tts = torch.optim.lr_scheduler.LambdaLR(optimizer_tts, lr_lambda_fn)
    sched_asr = torch.optim.lr_scheduler.LambdaLR(optimizer_asr, lr_lambda_fn)

    # ── EMA ──
    asr_base = getattr(asr_model, "_orig_mod", asr_model)
    tts_base = getattr(tts_model, "_orig_mod", tts_model)
    ema_asr = EMA(asr_base, cfg.ema_decay)
    ema_tts = EMA(tts_base, cfg.ema_decay)

    # ── Schedule & losses ──
    schedule = CoTrainSchedule(cfg.cotrain)
    tts_regs = TTSRegularizers(
        cfg.cotrain.lambda_energy,
        cfg.cotrain.lambda_smoothness,
        cfg.cotrain.lambda_bandlimit,
    ).to(device)

    # ── Resume ──
    global_step = 0
    start_epoch = 0
    best_val_acc = 0.0
    if resume_path:
        global_step, start_epoch, best_val_acc = load_checkpoint(
            resume_path, asr_model, tts_model, device,
            optimizer_tts=optimizer_tts, optimizer_asr=optimizer_asr,
            ema_asr=ema_asr, ema_tts=ema_tts,
        )
        for _ in range(global_step):
            sched_tts.step()
            sched_asr.step()
        print(f"Resumed: epoch={start_epoch}, step={global_step}, "
              f"val_acc={best_val_acc:.4f}")

    history: dict[str, list] = defaultdict(list)
    early_stop_counter = 0
    val_every = 5

    print(f"\nCo-training: {cfg.num_epochs} epochs, "
          f"ctc_to_tts={cfg.cotrain.ctc_to_tts}, "
          f"lambda_roundtrip={cfg.cotrain.lambda_roundtrip}")

    # ── Main loop ──
    train_start = time.time()
    for epoch in range(start_epoch + 1, cfg.num_epochs + 1):
        asr_model.train()
        tts_model.train()
        epoch_losses: dict[str, float] = defaultdict(float)
        n_batches = 0

        pbar = tqdm(train_loader, desc=f"Ep {epoch}/{cfg.num_epochs}",
                    leave=False, dynamic_ncols=True)
        for tokens, tok_lens, wav_ref, wav_lens in pbar:
            tokens = tokens.to(device, non_blocking=True)
            tok_lens = tok_lens.to(device, non_blocking=True)

            phase = schedule.get_phase(global_step)
            lam_ctc_tts = schedule.get_lambda_ctc_tts(global_step)
            lam_mel = schedule.get_lambda_mel(global_step)
            lam_rt = schedule.get_lambda_roundtrip(global_step)
            lam_ctc_ref = cfg.cotrain.lambda_ctc_ref
            mel_snr_range = schedule.get_mel_snr_range(global_step)
            wav_snr_range = schedule.get_wav_snr_range(global_step)
            lam_dur = cfg.cotrain.lambda_dur if phase < 2 else 0.0

            # Reference mels
            has_ref = wav_ref is not None and cfg.data.use_ps_ref
            if has_ref:
                wav_gpu = wav_ref.to(device, non_blocking=True)
                wl_gpu = wav_lens.to(device, non_blocking=True)
                if cfg.aug.use_waveform_aug:
                    wav_gpu, wl_gpu = wav_augmentor.augment_batch(
                        wav_gpu, wl_gpu,
                        seed=cfg.seed + global_step,
                        snr_range_override=wav_snr_range,
                    )
                with torch.amp.autocast("cuda", enabled=use_amp, dtype=amp_dtype):
                    mels_ref = _wav_to_logmel(wav_gpu)
                mel_lens = (wl_gpu // cfg.mel.hop_length + 1).clamp(
                    max=mels_ref.size(2)
                )
            else:
                mels_ref = None
                mel_lens = (tok_lens * 6).clamp(max=cfg.max_mel_len).to(device)

            # Duration targets
            use_ref_dur = has_ref and phase < 2
            if use_ref_dur:
                target_dur = compute_target_durations(
                    tok_lens, mel_lens, tokens.size(1)
                ).to(device)
                max_mel = int(mel_lens.max().item())
            else:
                target_dur = None
                max_mel = (
                    int(mel_lens.max().item()) if has_ref else cfg.max_mel_len
                )

            with torch.amp.autocast("cuda", enabled=use_amp, dtype=amp_dtype):
                mel_pred, log_dur_pred, dur_used = tts_model(
                    tokens, tok_lens,
                    target_durations=target_dur, max_mel_len=max_mel,
                )
                if not has_ref:
                    mel_lens = dur_used.sum(dim=1).clamp(
                        min=(tok_lens * 4 + 4), max=mel_pred.size(2),
                    )

                # CTC on TTS mel
                if lam_ctc_tts > 0 and not schedule.should_detach_tts(global_step):
                    mel_for_asr = mel_augmentor(
                        mel_pred, seed=cfg.seed + global_step,
                        snr_range_override=mel_snr_range,
                    )
                else:
                    mel_for_asr = mel_augmentor(
                        mel_pred.detach(), seed=cfg.seed + global_step,
                        snr_range_override=mel_snr_range,
                    )
                logits_tts, sub_lens_tts = asr_model(mel_for_asr, mel_lens)
                loss_ctc_tts = compute_ctc_loss(
                    logits_tts, sub_lens_tts, tokens, tok_lens, cfg.blank_id,
                )

                # CTC on reference
                if has_ref and lam_ctc_ref > 0:
                    ref_aug = mel_augmentor(
                        mels_ref,
                        seed=cfg.seed + global_step + 50000,
                        snr_range_override=mel_snr_range,
                    )
                    logits_ref, sub_lens_ref = asr_model(ref_aug, mel_lens)
                    loss_ctc_ref = compute_ctc_loss(
                        logits_ref, sub_lens_ref, tokens, tok_lens, cfg.blank_id,
                    )
                else:
                    loss_ctc_ref = torch.tensor(0.0, device=device)

                # Mel reconstruction
                if lam_mel > 0 and mels_ref is not None:
                    min_t = min(mel_pred.shape[2], mels_ref.shape[2])
                    loss_mel = F.l1_loss(
                        mel_pred[:, :, :min_t], mels_ref[:, :, :min_t]
                    ) + F.mse_loss(
                        mel_pred[:, :, :min_t], mels_ref[:, :, :min_t]
                    )
                else:
                    loss_mel = torch.tensor(0.0, device=device)

                # Roundtrip
                if lam_rt > 0:
                    mel_rt = _mel_roundtrip(mel_pred)
                    min_t_rt = min(mel_pred.shape[2], mel_rt.shape[2])
                    loss_roundtrip = F.l1_loss(
                        mel_pred[:, :, :min_t_rt],
                        mel_rt[:, :, :min_t_rt].to(mel_pred.dtype),
                    )
                else:
                    loss_roundtrip = torch.tensor(0.0, device=device)

                # Duration
                if target_dur is not None and lam_dur > 0:
                    log_td = torch.log1p(target_dur.float())
                    dur_mask = (
                        torch.arange(log_dur_pred.size(1), device=device).unsqueeze(0)
                        < tok_lens.unsqueeze(1)
                    ).float()
                    loss_dur = F.mse_loss(
                        log_dur_pred * dur_mask, log_td * dur_mask
                    )
                else:
                    loss_dur = torch.tensor(0.0, device=device)

                reg_losses = tts_regs(mel_pred)

                loss_total = (
                    lam_ctc_tts * loss_ctc_tts
                    + lam_ctc_ref * loss_ctc_ref
                    + lam_mel * loss_mel
                    + lam_rt * loss_roundtrip
                    + lam_dur * loss_dur
                    + reg_losses["reg_total"]
                )

            optimizer_tts.zero_grad(set_to_none=True)
            optimizer_asr.zero_grad(set_to_none=True)
            grad_scaler.scale(loss_total).backward()
            grad_scaler.unscale_(optimizer_tts)
            grad_scaler.unscale_(optimizer_asr)
            torch.nn.utils.clip_grad_norm_(tts_model.parameters(), cfg.grad_clip)
            torch.nn.utils.clip_grad_norm_(asr_model.parameters(), cfg.grad_clip)
            grad_scaler.step(optimizer_tts)
            grad_scaler.step(optimizer_asr)
            grad_scaler.update()
            sched_tts.step()
            sched_asr.step()

            # EMA
            if epoch >= cfg.ema_start_epoch:
                ema_asr.update(getattr(asr_model, "_orig_mod", asr_model))
                ema_tts.update(getattr(tts_model, "_orig_mod", tts_model))

            global_step += 1
            epoch_losses["total"] += loss_total.item()
            epoch_losses["ctc_tts"] += loss_ctc_tts.item()
            epoch_losses["ctc_ref"] += loss_ctc_ref.item()
            epoch_losses["mel"] += loss_mel.item()
            epoch_losses["roundtrip"] += loss_roundtrip.item()
            epoch_losses["dur"] += loss_dur.item()
            epoch_losses["reg"] += reg_losses["reg_total"].item()
            n_batches += 1

            if torch.isnan(loss_total):
                print(f"\nNaN at step {global_step}!")
                return history

            pbar.set_postfix({
                "P": phase,
                "loss": f"{loss_total.item():.3f}",
                "ctc": f"{loss_ctc_tts.item():.3f}",
                "rt": f"{loss_roundtrip.item():.3f}",
            })

        pbar.close()
        avg = {k: v / max(n_batches, 1) for k, v in epoch_losses.items()}
        for k, v in avg.items():
            history[f"train_{k}"].append(v)

        # ── Validation ──
        if epoch % val_every == 0 or epoch == 1:
            val_acc, val_em = validate(
                asr_model, tts_model, val_loader, device,
                use_amp=use_amp, amp_dtype=amp_dtype,
            )
            history["val_acc"].append(val_acc)
            ps_acc, _ = validate_ps(
                asr_model, val_loader, device, _wav_to_logmel,
                use_amp=use_amp, amp_dtype=amp_dtype,
            )
            history["ps_acc"].append(ps_acc)

            ema_acc = 0.0
            if epoch >= cfg.ema_start_epoch:
                ema_acc, _ = validate(
                    asr_model, tts_model, val_loader, device,
                    use_amp=use_amp, amp_dtype=amp_dtype,
                    use_ema=True, ema_asr=ema_asr, ema_tts=ema_tts,
                )
                history["ema_acc"].append(ema_acc)

            best_this = max(val_acc, ema_acc)
            is_best = best_this > best_val_acc
            elapsed = time.time() - train_start
            print(
                f"Ep {epoch:3d} | {schedule.status(global_step)} | "
                f"Loss:{avg['total']:.3f} | "
                f"TTS->ASR:{val_acc:.4f} PS:{ps_acc:.4f} "
                f"EMA:{ema_acc:.4f} | {elapsed/60:.0f}min"
                + (" ***BEST***" if is_best else "")
            )

            if is_best:
                best_val_acc = best_this
                use_ema_w = ema_acc > val_acc and epoch >= cfg.ema_start_epoch
                asr_b = getattr(asr_model, "_orig_mod", asr_model)
                tts_b = getattr(tts_model, "_orig_mod", tts_model)
                if use_ema_w:
                    ema_asr.apply_shadow(asr_b)
                    ema_tts.apply_shadow(tts_b)
                torch.save(
                    {
                        "tts_state": tts_b.state_dict(),
                        "asr_state": asr_b.state_dict(),
                        "cfg": asdict(cfg),
                        "step": global_step,
                        "epoch": epoch,
                        "val_acc": best_val_acc,
                        "used_ema": use_ema_w,
                    },
                    os.path.join(save_dir, "best_model.pt"),
                )
                if use_ema_w:
                    ema_asr.restore(asr_b)
                    ema_tts.restore(tts_b)

            # Early stopping
            if best_this >= cfg.early_stop_acc:
                early_stop_counter += 1
                if early_stop_counter >= cfg.early_stop_patience:
                    print(f"Early stop: acc >= {cfg.early_stop_acc}")
                    break
            else:
                early_stop_counter = 0

        # Periodic checkpoint
        if epoch % 25 == 0:
            asr_b = getattr(asr_model, "_orig_mod", asr_model)
            tts_b = getattr(tts_model, "_orig_mod", tts_model)
            torch.save(
                {
                    "tts_state": tts_b.state_dict(),
                    "asr_state": asr_b.state_dict(),
                    "opt_tts_state": optimizer_tts.state_dict(),
                    "opt_asr_state": optimizer_asr.state_dict(),
                    "ema_asr_state": ema_asr.state_dict(),
                    "ema_tts_state": ema_tts.state_dict(),
                    "cfg": asdict(cfg),
                    "step": global_step,
                    "epoch": epoch,
                    "val_acc": best_val_acc,
                },
                os.path.join(save_dir, "latest.pt"),
            )

    total_time = time.time() - train_start
    print(f"\nTraining complete in {total_time / 3600:.1f}h")
    print(f"Best TTS->ASR acc: {best_val_acc:.4f}")
    return history


# ---------------------------------------------------------------------------
#  CLI
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(description="TTR Co-Training")
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--save-dir", type=str, default="./checkpoints")
    parser.add_argument("--resume", type=str, default=None)
    parser.add_argument("--dataset-size", type=int, default=None)
    parser.add_argument("--lr-tts", type=float, default=None)
    parser.add_argument("--lr-asr", type=float, default=None)
    args = parser.parse_args()

    cfg = Config()
    if args.epochs is not None:
        cfg.num_epochs = args.epochs
    if args.batch_size is not None:
        cfg.batch_size = args.batch_size
    if args.dataset_size is not None:
        cfg.data.dataset_size = args.dataset_size
    if args.lr_tts is not None:
        cfg.lr_tts = args.lr_tts
    if args.lr_asr is not None:
        cfg.lr_asr = args.lr_asr

    train(cfg, save_dir=args.save_dir, resume_path=args.resume)


if __name__ == "__main__":
    main()
