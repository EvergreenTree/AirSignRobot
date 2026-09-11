"""Single-GPU ACT training, episode-held-out validation, resumable optimizer/RNG state."""
from __future__ import annotations
import argparse
from dataclasses import asdict
import hashlib
import json
import math
from pathlib import Path
import random
import time

import numpy as np
import torch
from torch.utils.data import DataLoader, Sampler, Subset

from .data import EpisodeDataset
from .download import sha256
from .model import ACT, ACTConfig, loss_fn
from .schema import PROFILES, JOINT_ACTION_INDICES, JOINT_STATE_INDICES


def augment(images):
    # Same brightness/contrast across all three views. Preserve geometry, cable colors and camera identity.
    b = images.shape[0]
    brightness = torch.empty(b, 1, 1, 1, 1, device=images.device).uniform_(0.8, 1.2)
    contrast = torch.empty_like(brightness).uniform_(0.85, 1.15)
    return ((images.float() - 127.5) * contrast + 127.5).mul(brightness).clamp(0, 255)


@torch.inference_mode()
def evaluate(model, loader, device, stats, amp):
    model.eval()
    error = first = counts = None
    mean_error = 0.0
    persistence_error = 0.0
    n = 0
    for batch in loader:
        images, state, target, mask = [batch[k].to(device, non_blocking=True) for k in ("images", "state", "action", "mask")]
        with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=amp):
            pred, _, _ = model(images, state)  # No target/posterior leakage during evaluation.
        diff = (pred.float()-target).abs()
        valid = mask[..., None].expand_as(diff)
        batch_error = (diff*valid).sum((0, 1)).double().cpu()
        batch_first = diff[:, 0].sum(0).double().cpu()
        batch_counts = valid.sum((0, 1)).double().cpu()
        error = batch_error if error is None else error + batch_error
        first = batch_first if first is None else first + batch_first
        counts = batch_counts if counts is None else counts + batch_counts
        mean_error += float((target.abs()*valid).sum())
        raw_state = state.float()*torch.as_tensor(stats['state_std'], device=device) + torch.as_tensor(stats['state_mean'], device=device)
        first_target = target[:, 0].float()*torch.as_tensor(stats['action_std'], device=device) + torch.as_tensor(stats['action_mean'], device=device)
        persistence_error += float((raw_state[:, JOINT_STATE_INDICES]-first_target[:, JOINT_ACTION_INDICES]).abs().sum())
        n += len(target)
    normalized = error / counts
    scale = torch.tensor(stats["action_std"])
    return {"normalized_mae": float(normalized.mean()), "native_mae_by_dimension": (normalized*scale).tolist(),
            "first_action_native_mae_by_dimension": (first/n*scale).tolist(),
            "joint_persistence_first_action_mae_rad": persistence_error/(n*14),
            "model_first_joint_action_mae_rad": float((first/n*scale)[JOINT_ACTION_INDICES].mean()),
            "train_mean_baseline_normalized_mae": mean_error/float(counts.sum()), "samples": n,
            "metric_scope": "offline held-out action prediction; no task-success claim"}


def atomic_checkpoint(path, record):
    temp = path.with_suffix(".partial")
    torch.save(record, temp)
    temp.replace(path)


class GlobalStepBatchSampler(Sampler):
    """Each batch depends only on seed and optimizer step, never prefetch state."""

    def __init__(self, size: int, batch_size: int, seed: int, start_step: int, total_steps: int):
        if size <= 0 or batch_size <= 0 or start_step < 0:
            raise ValueError("Invalid deterministic sampler dimensions")
        self.size, self.batch_size, self.seed = size, batch_size, seed
        self.start_step, self.total_steps = start_step, total_steps

    def __len__(self):
        return max(0, self.total_steps-self.start_step)

    def __iter__(self):
        for step in range(self.start_step, self.total_steps):
            digest = hashlib.sha256(f"airsign-batch-v1:{self.seed}:{step}".encode()).digest()
            generator = torch.Generator().manual_seed(int.from_bytes(digest[:8], "little"))
            yield torch.randint(self.size, (self.batch_size,), generator=generator).tolist()


def initialize_resume_best(output: Path, checkpoint: dict) -> tuple[float, str]:
    """Keep an actual best artifact, not a score for unavailable model weights."""
    destination = output / "best.pt"
    if destination.exists():
        existing = torch.load(destination, map_location="cpu", weights_only=False)
        for key in ("format", "task", "dataset_sha256", "model_config"):
            left, right = existing.get(key), checkpoint.get(key)
            if key == 'model_config':
                left = {**left, 'joint_residual': left.get('joint_residual', False)}
                right = {**right, 'joint_residual': right.get('joint_residual', False)}
            if left != right:
                raise ValueError("Output best checkpoint belongs to a different run configuration")
        score = float(existing["validation"]["normalized_mae"])
        if not math.isfinite(score):
            raise ValueError("Output best checkpoint has invalid validation")
        return score, "kept_existing_best_artifact"
    score = float(checkpoint["validation"]["normalized_mae"])
    if not math.isfinite(score):
        raise ValueError("Resume checkpoint needs a finite validation score")
    historical = float(checkpoint.get("best_validation", score))
    note = ("resumed_best_artifact" if math.isclose(score, historical, rel_tol=1e-12, abs_tol=1e-12)
            else "resumed_checkpoint_baseline; historical_best_weights_unavailable")
    # The resumed file may be last.pt, whose weights do not have the historical
    # best score. Use its measured score as the baseline when that best artifact
    # is absent, and retain the old number only as provenance.
    baseline = {**checkpoint, "best_validation": score, "historical_best_validation": historical,
                "resume_best_note": note}
    atomic_checkpoint(destination, baseline)
    return score, note


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--dataset", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--task", type=int, choices=[1, 2], required=True)
    p.add_argument("--steps", type=int, default=10000)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--chunk-size", type=int, default=32)
    p.add_argument("--hidden-dim", type=int, default=256)
    p.add_argument("--layers", type=int, default=3)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--vision-lr", type=float, default=1e-5)
    p.add_argument("--kl-weight", type=float, default=1.0)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--eval-every", type=int, default=500)
    p.add_argument("--eval-samples", type=int, default=1024)
    p.add_argument("--device", default="cuda")
    p.add_argument("--no-pretrained", action="store_true")
    p.add_argument("--absolute-actions", action="store_true", help="Disable the measured-joint residual anchor")
    p.add_argument("--backbone-checkpoint", type=Path, help="Local audited ResNet backbone weights")
    p.add_argument("--resume", type=Path)
    p.add_argument("--evaluate", choices=["validation", "test"])
    args = p.parse_args()
    if min(args.steps, args.batch_size, args.eval_every, args.eval_samples) < 1 or args.workers < 0:
        p.error("steps, batch size, evaluation interval/samples must be positive; workers nonnegative")
    torch.set_num_threads(4)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device(args.device)
    amp = device.type == "cuda"
    if amp and not torch.cuda.is_available():
        raise RuntimeError("CUDA required by this run; choose --device cpu explicitly for a smoke check")
    if amp:
        torch.backends.cuda.matmul.allow_tf32 = True
    if args.output.exists() and any(args.output.iterdir()) and not args.resume:
        raise FileExistsError("Refusing to overwrite a run; use a new output or --resume")
    args.output.mkdir(parents=True, exist_ok=True)
    manifest_hash = sha256(args.dataset / "manifest.json")
    checkpoint = torch.load(args.resume, map_location="cpu", weights_only=False) if args.resume else None
    if checkpoint and checkpoint.get('inference_only') and not args.evaluate:
        raise ValueError('Inference exports omit optimizer/RNG state; resume from a training checkpoint')
    if checkpoint and (checkpoint["dataset_sha256"] != manifest_hash or checkpoint["task"] != args.task):
        raise ValueError("Resume dataset/task differs from checkpoint")
    start_step = checkpoint["step"] if checkpoint else 0
    best, resume_note = float("inf"), None
    if checkpoint and not args.evaluate:
        best, resume_note = initialize_resume_best(args.output, checkpoint)
        if args.steps <= start_step:
            print(json.dumps({"complete": True, "step": start_step, "optimizer_updates": 0,
                              "best_validation": best, "resume_best_note": resume_note,
                              "checkpoint_sha256": sha256(args.output / "best.pt")}), flush=True)
            return
    train = EpisodeDataset(args.dataset, args.task, "train", args.chunk_size, checkpoint["stats"] if checkpoint else None)
    profile = PROFILES[args.task]
    config = ACTConfig(profile.state_dim, profile.action_dim, train.manifest["image_size"], args.chunk_size,
                       args.hidden_dim, encoder_layers=args.layers, decoder_layers=args.layers,
                       joint_residual=not args.absolute_actions)
    if checkpoint and 'joint_residual' not in checkpoint['model_config']:
        checkpoint['model_config']['joint_residual'] = False
    if checkpoint and asdict(config) != checkpoint["model_config"]:
        raise ValueError("Resume model config differs; pass the original model flags")
    model = ACT(config, pretrained=not args.no_pretrained and checkpoint is None,
                backbone_path=args.backbone_checkpoint if checkpoint is None else None).to(device)
    if checkpoint:
        model.load_state_dict(checkpoint["model"])
    else:
        model.configure_joint_anchor(train.stats)
    optimizer = torch.optim.AdamW([
        {"params": model.backbone.parameters(), "lr": args.vision_lr},
        {"params": [v for k,v in model.named_parameters() if not k.startswith("backbone.")], "lr": args.lr},
    ], weight_decay=1e-4)
    if checkpoint and not args.evaluate:
        optimizer.load_state_dict(checkpoint["optimizer"])
        torch.set_rng_state(checkpoint["rng_torch"])
        if amp and checkpoint.get("rng_cuda") is not None:
            torch.cuda.set_rng_state_all(checkpoint["rng_cuda"])
        random.setstate(checkpoint["rng_python"])
        np.random.set_state(checkpoint["rng_numpy"])
    validation = EpisodeDataset(args.dataset, args.task, args.evaluate or "validation", args.chunk_size, train.stats)
    ids = np.linspace(0, len(validation)-1, min(args.eval_samples, len(validation)), dtype=int)
    val_loader = DataLoader(Subset(validation, ids), batch_size=args.batch_size, num_workers=args.workers,
                            pin_memory=amp, persistent_workers=args.workers > 0,
                            multiprocessing_context="spawn" if args.workers else None,
                            generator=torch.Generator().manual_seed(args.seed ^ 0x51A7))
    if args.evaluate:
        if checkpoint is None:
            raise ValueError("--evaluate requires --resume checkpoint")
        metrics = evaluate(model, val_loader, device, train.stats, amp)
        metrics.update({"checkpoint_sha256": sha256(args.resume), "split": args.evaluate})
        (args.output / f"{args.evaluate}_metrics.json").write_text(json.dumps(metrics, indent=2) + "\n")
        print(json.dumps(metrics), flush=True)
        return
    sampling = {"format": "global_step_v1", "seed": args.seed, "batch_size": args.batch_size}
    if checkpoint and checkpoint.get("sampling") is not None and checkpoint["sampling"] != sampling:
        raise ValueError("Resume sampling configuration differs; preserve seed and batch size")
    if checkpoint and checkpoint.get("sampling") is None:
        resume_note = f"{resume_note}; migrated_legacy_sampler_no_exact_old_stream_replay"
    sampler = GlobalStepBatchSampler(len(train), args.batch_size, args.seed, start_step, args.steps)
    # Worker-seed generation must not consume the RNG used for model dropout
    # and augmentation. Dataset reads are deterministic across worker counts.
    loader = DataLoader(train, batch_sampler=sampler, num_workers=args.workers,
                        pin_memory=amp, persistent_workers=args.workers > 0,
                        multiprocessing_context="spawn" if args.workers else None,
                        generator=torch.Generator().manual_seed(args.seed ^ 0x7A11))
    hardware = {"torch": torch.__version__, "cuda": torch.version.cuda, "device": str(device)}
    if amp:
        props = torch.cuda.get_device_properties(device)
        hardware.update({"name": props.name, "compute_capability": [props.major, props.minor],
                         "memory_bytes": props.total_memory})
    source_hashes = {x.name: sha256(x) for x in Path(__file__).parent.glob("*.py")}
    run = {"team": "AirSign", "task": args.task, "model_config": asdict(config),
           "dataset_coverage": train.manifest.get('coverage', 'full_official_release'),
           "args": {k: str(v) if isinstance(v, Path) else v for k,v in vars(args).items()},
           "hardware": hardware, "dataset_sha256": manifest_hash, "source_hashes": source_hashes,
           "train_episodes": len(train.episodes), "validation_episodes": len(validation.episodes),
           "sampling": sampling, "resume_best_note": resume_note,
           "backbone_checkpoint_sha256": sha256(args.backbone_checkpoint) if args.backbone_checkpoint and not checkpoint else None}
    run_path = args.output/'run.json'
    if run_path.exists():
        run_path = args.output/f'resume-{start_step:08d}-{time.time_ns()}.json'
    run_path.write_text(json.dumps(run, indent=2) + "\n")
    print(json.dumps(run), flush=True)
    wall = time.monotonic()
    with (args.output / "metrics.jsonl").open("a") as log:
        for step, batch in enumerate(loader, start_step+1):
            model.train()
            images, state, target, mask = [batch[k].to(device, non_blocking=True) for k in ("images", "state", "action", "mask")]
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=amp):
                prediction, mu, logvar = model(augment(images), state, target, mask)
                loss, reconstruction, kl = loss_fn(prediction, target, mask, mu, logvar, args.kl_weight)
            if not torch.isfinite(loss):
                raise FloatingPointError(f"Nonfinite training loss at step {step}")
            loss.backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0, error_if_nonfinite=True)
            optimizer.step()
            if step % 25 == 0 or step == 1:
                record = {"step": step, "loss": float(loss), "l1": float(reconstruction), "kl": float(kl),
                          "grad_norm": float(grad_norm), "elapsed_s": time.monotonic()-wall}
                log.write(json.dumps(record)+"\n"); log.flush()
                print(json.dumps(record), flush=True)
            if step % args.eval_every == 0 or step == args.steps:
                metrics = evaluate(model, val_loader, device, train.stats, amp)
                improved = metrics["normalized_mae"] < best
                best = min(best, metrics["normalized_mae"])
                record = {"step": step, "validation": metrics, "best": best}
                log.write(json.dumps(record)+"\n"); log.flush()
                print(json.dumps(record), flush=True)
                saved = {"format": "airsign_act_v1", "task": args.task, "step": step,
                         "dataset_coverage": run['dataset_coverage'],
                         "model_config": asdict(config), "model": model.state_dict(), "optimizer": optimizer.state_dict(),
                         "stats": train.stats, "dataset_sha256": manifest_hash,
                         "schema": train.manifest["schemas"][str(args.task)], "cameras": train.manifest["cameras"],
                         "best_validation": best, "validation": metrics, "source_hashes": source_hashes,
                         "rng_torch": torch.get_rng_state(), "rng_cuda": torch.cuda.get_rng_state_all() if amp else None,
                         "rng_python": random.getstate(), "rng_numpy": np.random.get_state(),
                         "sampling": sampling, "resume_best_note": resume_note}
                atomic_checkpoint(args.output / "last.pt", saved)
                if improved:
                    atomic_checkpoint(args.output / "best.pt", saved)
            if step >= args.steps:
                break
    print(json.dumps({"complete": True, "step": step, "best_validation": best,
                      "checkpoint_sha256": sha256(args.output / "best.pt")}), flush=True)


if __name__ == "__main__":
    main()
