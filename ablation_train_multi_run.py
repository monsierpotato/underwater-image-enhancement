"""
ablation_train_multi_run.py
===========================
Train the 4 UNet ablation variants (3ch, 4ch_t, 4ch_b, 5ch) from scratch
N times (default 5) using different random seeds, then evaluate every
resulting checkpoint and report mean ± std across seeds.

Each seed controls:
  - Weight initialisation (fresh model built before every run)
  - Train/val random split (10 % hold-out)
  - DataLoader shuffle order

Usage
-----
    # Default: 4 variants × 5 seeds, 50 epochs, EUVP dataset, UDCP prior
    python ablation_train_multi_run.py

    # Custom:
    python ablation_train_multi_run.py \\
        --data_train_euvp ./datasets/EUVP \\
        --checkpoint_dir  ./checkpoints/ablation_multi \\
        --val_folder      ./results/ablation_multi \\
        --prior_method    udcp \\
        --nEpochs         50 \\
        --batchSize       16 \\
        --num_runs        5 \\
        --seeds 0 1 2 3 4

Output
------
* Checkpoints : <checkpoint_dir>/<variant>_seed<N>_<timestamp>/best_model.pth
* Logs        : ./logs/<variant>_seed<N>_<timestamp>.log
* JSON report : <val_folder>/ablation_train_multi_run_results.json
* Console     : ranked mean ± std summary table
"""

import argparse
import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.utils.data as data

# ---------------------------------------------------------------------------
# Ensure project root is importable when run from any cwd
# ---------------------------------------------------------------------------
_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

# Project imports
from data.data import get_euvp_training_set
from net.registry import build_model, parse_model_variant
from loss import CompositeLoss
from measure_underwater import evaluate_loader
from train import (
    _Tee,
    _resolve_physics_extractor,
    _collate_train,
    _collate_val,
    train_epoch,
    val_loss_epoch,
    build_scheduler,
    save_ckpt,
    load_ckpt,
    EarlyStopping,
)
from eval import collect_test_pairs, TestDataset, _parse_model_name


# ---------------------------------------------------------------------------
# The 4 UNet ablation variants
# ---------------------------------------------------------------------------
ABLATION_VARIANTS = ["unet_3ch", "unet_4ch_t", "unet_4ch_b", "unet_5ch"]

# Metrics to aggregate
METRIC_KEYS = ("psnr", "ssim", "ciede2000", "uciqe", "uiqm", "inference_ms_per_img")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _make_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Multi-seed ablation: train 4 UNet variants N times, report mean±std.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    # Dataset / paths
    p.add_argument("--data_train_euvp", default="./datasets/EUVP",
                   help="EUVP dataset root (training + test_samples).")
    p.add_argument("--checkpoint_dir", default="./checkpoints/ablation_multi",
                   help="Root directory where per-run checkpoint folders are saved.")
    p.add_argument("--val_folder", default="./results/ablation_multi",
                   help="Output directory for the JSON summary.")
    p.add_argument("--prior_method", default="udcp",
                   choices=["udcp", "gdcp", "gupdm"],
                   help="Physics prior (must be consistent across all variants).")
    p.add_argument("--euvp_subset", default="all",
                   help="EUVP subset(s) to use (all | underwater_imagenet | ...).")

    # Training hyper-params
    p.add_argument("--nEpochs", type=int, default=50,
                   help="Maximum training epochs per run.")
    p.add_argument("--batchSize", type=int, default=16)
    p.add_argument("--cropSize", type=int, default=256)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--weight_decay", type=float, default=1e-5)
    p.add_argument("--threads", type=int, default=4,
                   help="DataLoader worker threads.")
    p.add_argument("--in_memory", action="store_true",
                   help="Pre-load dataset into RAM.")
    p.add_argument("--snapshots", type=int, default=10,
                   help="Save a periodic checkpoint every N epochs.")
    p.add_argument("--early_stop_patience", type=int, default=20,
                   help="Early-stopping patience (epochs without PSNR improvement).")

    # Scheduler
    p.add_argument("--cos_restart", action="store_true",
                   help="Use CosineAnnealingRestart LR scheduler.")
    p.add_argument("--cos_restart_cyclic", action="store_true",
                   help="Use cyclic cosine restart LR scheduler.")
    p.add_argument("--scheduler_step", type=int, default=30)
    p.add_argument("--scheduler_gamma", type=float, default=0.5)
    p.add_argument("--warmup_epochs", type=int, default=0)
    p.add_argument("--start_warmup", action="store_true")

    # Loss weights
    p.add_argument("--L1_weight", type=float, default=1.0)
    p.add_argument("--perceptual_weight", type=float, default=1.0)
    p.add_argument("--SSIM_weight", type=float, default=0.0)

    # Multi-run
    p.add_argument("--num_runs", type=int, default=5,
                   help="Number of independent training runs (one per seed).")
    p.add_argument("--seeds", type=int, nargs="+", default=None,
                   help="Explicit seed list (length must equal --num_runs if given).")
    p.add_argument("--variants", type=str, nargs="+", default=ABLATION_VARIANTS,
                   help="Which model variants to train.")

    # Misc
    p.add_argument("--gpu_mode", action="store_true", default=True)
    p.add_argument("--no_gpu", dest="gpu_mode", action="store_false")
    p.add_argument("--pretrained_backbone", action="store_true", default=False,
                   help="Load pretrained ImageNet backbone (not applicable to plain UNet).")
    return p


# ---------------------------------------------------------------------------
# Single training run
# ---------------------------------------------------------------------------

def train_one_run(
    variant: str,
    seed: int,
    args,
    device: torch.device,
    train_ds_full,
    physics_extractor,
) -> str:
    """
    Train ``variant`` from scratch with ``seed``.

    Returns
    -------
    str : path to the saved best_model.pth
    """
    # ---- Reproducibility --------------------------------------------------
    torch.manual_seed(seed)
    np.random.seed(seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    # ---- Parse variant ----------------------------------------------------
    _, in_channels, physics_mode = parse_model_variant(variant)

    # ---- Train / val split ------------------------------------------------
    n_val   = max(1, int(len(train_ds_full) * 0.10))
    n_train = len(train_ds_full) - n_val
    train_ds, val_ds = data.random_split(
        train_ds_full,
        [n_train, n_val],
        generator=torch.Generator().manual_seed(seed),
    )

    collate_tr = lambda b: _collate_train(b, physics_mode, physics_extractor)
    collate_vl = lambda b: _collate_val(b, physics_mode, physics_extractor)

    train_loader = data.DataLoader(
        train_ds, batch_size=args.batchSize, shuffle=True,
        num_workers=args.threads, pin_memory=(device.type == "cuda"),
        drop_last=True, collate_fn=collate_tr,
    )
    val_loader = data.DataLoader(
        val_ds, batch_size=args.batchSize, shuffle=False,
        num_workers=args.threads, pin_memory=(device.type == "cuda"),
        drop_last=False, collate_fn=collate_vl,
    )

    # ---- Model ------------------------------------------------------------
    model = build_model(variant, pretrained_backbone=args.pretrained_backbone).to(device)

    # ---- Loss / optimiser / scheduler ------------------------------------
    criterion = CompositeLoss(
        lambda_l1=args.L1_weight,
        lambda_perc=args.perceptual_weight,
        lambda_ssim=args.SSIM_weight,
        device=device,
    )
    optimizer = torch.optim.Adam(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )
    scheduler = build_scheduler(optimizer, args)

    # ---- Checkpoint / log paths ------------------------------------------
    ts      = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_id  = f"{variant}_seed{seed}_{ts}"
    ckpt_dir = os.path.join(args.checkpoint_dir, run_id)
    best_path = os.path.join(ckpt_dir, "best_model.pth")
    os.makedirs(ckpt_dir, exist_ok=True)

    log_dir  = os.path.join(str(_HERE), "logs")
    os.makedirs(log_dir, exist_ok=True)
    log_path = os.path.join(log_dir, f"{run_id}.log")

    _log_file    = open(log_path, "w", encoding="utf-8")
    _orig_stdout = sys.stdout
    sys.stdout   = _Tee(_orig_stdout, _log_file)

    print(f"\n{'='*65}")
    print(f"  variant={variant}  seed={seed}")
    print(f"  train={n_train}  val={n_val}  device={device}")
    print(f"  ckpt_dir  : {ckpt_dir}")
    print(f"  log_file  : {log_path}")
    print(f"{'='*65}")
    print(f"{'Epoch':>6}  {'TrainL':>8}  {'ValL':>8}  {'PSNR':>7}  {'SSIM':>7}  {'LR':>9}  {'Time':>6}")
    print("-" * 65)

    best_psnr  = 0.0
    best_epoch = 0
    es         = EarlyStopping(patience=args.early_stop_patience, min_delta=1e-4, mode="max")
    LOG_EVERY  = max(1, args.nEpochs // 20)

    history = {"train_loss": [], "val_loss": [], "val_psnr": [], "val_ssim": [], "lr": []}
    t_train_start = time.time()

    for epoch in range(1, args.nEpochs + 1):
        t0 = time.time()

        tr_loss, _ = train_epoch(model, train_loader, optimizer, criterion, device)
        vl_loss    = val_loss_epoch(model, val_loader, criterion, device)

        if epoch == 1 or epoch % LOG_EVERY == 0:
            val_metrics, _ = evaluate_loader(model, val_loader, device, max_samples=100)
            val_psnr = val_metrics["psnr"]
            val_ssim = val_metrics["ssim"]
        else:
            val_psnr = history["val_psnr"][-1] if history["val_psnr"] else 0.0
            val_ssim = history["val_ssim"][-1] if history["val_ssim"] else 0.0

        scheduler.step()
        cur_lr  = optimizer.param_groups[0]["lr"]
        elapsed = time.time() - t0

        history["train_loss"].append(tr_loss)
        history["val_loss"].append(vl_loss)
        history["val_psnr"].append(val_psnr)
        history["val_ssim"].append(val_ssim)
        history["lr"].append(cur_lr)

        flag = ""
        if val_psnr > best_psnr + 1e-4:
            best_psnr  = val_psnr
            best_epoch = epoch
            save_ckpt(model, optimizer, epoch,
                      {"psnr": val_psnr, "ssim": val_ssim, "val_loss": vl_loss},
                      best_path)
            flag = "  <- BEST"

        if epoch % args.snapshots == 0:
            save_ckpt(model, optimizer, epoch,
                      {"psnr": val_psnr, "ssim": val_ssim},
                      os.path.join(ckpt_dir, f"epoch_{epoch:04d}.pth"))

        if epoch == 1 or epoch % LOG_EVERY == 0 or flag:
            print(f"{epoch:>6}  {tr_loss:>8.4f}  {vl_loss:>8.4f}  "
                  f"{val_psnr:>7.3f}  {val_ssim:>7.4f}  {cur_lr:>9.2e}  "
                  f"{elapsed:>5.1f}s{flag}")

        if es(val_psnr):
            print(f"\nEarly stopping at epoch {epoch} "
                  f"(no improvement for {args.early_stop_patience} evals)")
            break

    # Save last + history
    save_ckpt(model, optimizer, epoch,
              {"psnr": val_psnr, "ssim": val_ssim},
              os.path.join(ckpt_dir, "last_model.pth"))

    total_min = (time.time() - t_train_start) / 60
    print(f"\n  Done.  Best PSNR={best_psnr:.4f} dB @ epoch {best_epoch}")
    print(f"  Total training time: {total_min:.1f} min")

    hist_path = os.path.join(ckpt_dir, "training_history.json")
    with open(hist_path, "w") as f:
        json.dump(history, f, indent=2)

    sys.stdout = _orig_stdout
    _log_file.close()
    print(f"  [INFO] Log -> {log_path}")

    return best_path


# ---------------------------------------------------------------------------
# Evaluate one checkpoint on the test set
# ---------------------------------------------------------------------------

def eval_checkpoint(
    model_name: str,
    best_path: str,
    test_ds,
    physics_extractor,
    device: torch.device,
    batch_size: int = 1,
    threads: int = 0,
) -> dict:
    """Load best_model.pth and run full metric evaluation on test_ds."""
    _, in_channels, physics_mode = parse_model_variant(model_name)
    model = build_model(model_name, pretrained_backbone=False).to(device)
    ckpt_epoch, ckpt_metrics = load_ckpt(best_path, model, device=str(device))
    model.eval()

    collate_fn = lambda b: _collate_val(b, physics_mode, physics_extractor)
    loader = data.DataLoader(
        test_ds, batch_size=batch_size, shuffle=False,
        num_workers=threads, pin_memory=(device.type == "cuda"),
        drop_last=False, collate_fn=collate_fn,
    )

    metrics, n = evaluate_loader(model, loader, device)
    metrics["best_epoch"]    = ckpt_epoch
    metrics["n_images"]      = n
    metrics["val_psnr_ckpt"] = ckpt_metrics.get("psnr")
    return metrics


# ---------------------------------------------------------------------------
# Aggregation
# ---------------------------------------------------------------------------

def _aggregate(runs_list: list) -> dict:
    agg = {}
    for key in METRIC_KEYS:
        vals = [r[key] for r in runs_list if key in r]
        if vals:
            agg[f"{key}_mean"]   = float(np.mean(vals))
            agg[f"{key}_std"]    = float(np.std(vals, ddof=1) if len(vals) > 1 else 0.0)
            agg[f"{key}_values"] = vals
    return agg


# ---------------------------------------------------------------------------
# Summary table
# ---------------------------------------------------------------------------

def _print_summary(all_agg: dict):
    COLS = [
        ("psnr",                "PSNR(dB) (up)"),
        ("ssim",                "SSIM (up)"),
        ("ciede2000",           "CIEDE2000 (dn)"),
        ("uciqe",               "UCIQE (up)"),
        ("uiqm",                "UIQM (up)"),
        ("inference_ms_per_img","Inf(ms/img)"),
    ]
    W = 130
    print(f"\n{'='*W}")
    print("  ABLATION MULTI-SEED RESULTS  (mean +/- std, Bessel-corrected, test set)")
    print(f"{'='*W}")
    header = f"{'Variant':<20}"
    for _, label in COLS:
        header += f"  {label:>22}"
    print(header)
    print("-" * W)
    rows = sorted(all_agg.items(),
                  key=lambda x: x[1].get("psnr_mean", 0), reverse=True)
    for variant, agg in rows:
        row = f"{variant:<20}"
        for key, _ in COLS:
            mean = agg.get(f"{key}_mean")
            std  = agg.get(f"{key}_std")
            if mean is not None:
                row += f"  {mean:>9.4f} +/- {std:>6.4f}"
            else:
                row += f"  {'N/A':>22}"
        print(row)
    print("-" * W)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = _make_parser()
    args   = parser.parse_args()

    # Resolve seeds
    seeds = args.seeds if args.seeds is not None else list(range(args.num_runs))
    num_runs = len(seeds)

    device = torch.device("cuda" if (args.gpu_mode and torch.cuda.is_available()) else "cpu")
    physics_extractor = _resolve_physics_extractor(args.prior_method)

    print(f"Device         : {device}")
    print(f"Prior method   : {args.prior_method}")
    print(f"Variants       : {args.variants}")
    print(f"Seeds          : {seeds}  (num_runs={num_runs})")
    print(f"Epochs/run     : {args.nEpochs}")
    print(f"Total runs     : {len(args.variants) * num_runs}")
    print(f"Checkpoint dir : {args.checkpoint_dir}")

    # -----------------------------------------------------------------------
    # Load training dataset ONCE (shared across all variants & seeds)
    # -----------------------------------------------------------------------
    print(f"\nLoading EUVP dataset from '{args.data_train_euvp}' ...")
    train_ds_full = get_euvp_training_set(
        args.data_train_euvp,
        img_size=args.cropSize,
        subset=args.euvp_subset,
        in_memory=args.in_memory,
    )
    print(f"Full dataset size : {len(train_ds_full)} images")

    # Load test set ONCE for evaluation
    print(f"Loading test pairs ...")
    test_pairs = collect_test_pairs(args.data_train_euvp)
    if not test_pairs:
        print("[ERROR] No test pairs found. Check test_samples/Inp and GTr.")
        sys.exit(1)
    test_ds = TestDataset(test_pairs, img_size=args.cropSize)
    print(f"Test set size     : {len(test_ds)} images")

    os.makedirs(args.checkpoint_dir, exist_ok=True)
    os.makedirs(args.val_folder, exist_ok=True)

    # -----------------------------------------------------------------------
    # PHASE 1: Train all variants × all seeds
    # -----------------------------------------------------------------------
    # best_paths[variant][seed_idx] = path to best_model.pth
    best_paths: dict[str, list] = {v: [] for v in args.variants}

    total = len(args.variants) * num_runs
    done  = 0

    for seed_idx, seed in enumerate(seeds):
        for variant in args.variants:
            done += 1
            print(f"\n[TRAIN {done}/{total}]  variant={variant}  seed={seed}  "
                  f"(run {seed_idx+1}/{num_runs})")
            bp = train_one_run(
                variant=variant,
                seed=seed,
                args=args,
                device=device,
                train_ds_full=train_ds_full,
                physics_extractor=physics_extractor,
            )
            best_paths[variant].append(bp)
            print(f"  Saved -> {bp}")

    # -----------------------------------------------------------------------
    # PHASE 2: Evaluate all checkpoints on the test set
    # -----------------------------------------------------------------------
    print(f"\n{'='*65}")
    print("  PHASE 2: Evaluating all checkpoints on test set ...")
    print(f"{'='*65}")

    # per_variant_runs[variant] = list of metric dicts (one per seed)
    per_variant_runs: dict[str, list] = {v: [] for v in args.variants}

    eval_done  = 0
    eval_total = len(args.variants) * num_runs

    for variant in args.variants:
        for seed_idx, (seed, bp) in enumerate(zip(seeds, best_paths[variant])):
            eval_done += 1
            print(f"\n[EVAL {eval_done}/{eval_total}]  variant={variant}  seed={seed}")
            try:
                m = eval_checkpoint(
                    model_name=variant,
                    best_path=bp,
                    test_ds=test_ds,
                    physics_extractor=physics_extractor,
                    device=device,
                    batch_size=1,
                    threads=0,
                )
                per_variant_runs[variant].append({"seed": seed, "checkpoint": bp, **m})
                print(f"  PSNR={m['psnr']:.4f}  SSIM={m['ssim']:.4f}  "
                      f"CIEDE2000={m['ciede2000']:.4f}  "
                      f"UCIQE={m['uciqe']:.4f}  UIQM={m['uiqm']:.4f}")
            except Exception as exc:
                print(f"  [ERROR] {variant} seed={seed}: {exc}")
                import traceback; traceback.print_exc()

    # -----------------------------------------------------------------------
    # Aggregate
    # -----------------------------------------------------------------------
    all_agg: dict[str, dict] = {}
    for variant, runs_list in per_variant_runs.items():
        if runs_list:
            all_agg[variant] = _aggregate(runs_list)

    # -----------------------------------------------------------------------
    # Print summary
    # -----------------------------------------------------------------------
    _print_summary(all_agg)

    # -----------------------------------------------------------------------
    # Save JSON
    # -----------------------------------------------------------------------
    out_path = os.path.join(args.val_folder, "ablation_train_multi_run_results.json")
    output = {
        "_meta": {
            "timestamp":      datetime.now().isoformat(timespec="seconds"),
            "device":         str(device),
            "variants":       args.variants,
            "num_runs":       num_runs,
            "seeds":          seeds,
            "nEpochs":        args.nEpochs,
            "prior_method":   args.prior_method,
            "checkpoint_dir": args.checkpoint_dir,
            "data_root":      args.data_train_euvp,
            "crop_size":      args.cropSize,
        },
        "per_variant": {
            variant: {
                "runs":       per_variant_runs[variant],
                "aggregated": all_agg.get(variant, {}),
            }
            for variant in args.variants
        },
    }
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)

    print(f"\n{'='*65}")
    print(f"  Results saved -> {out_path}")
    print(f"{'='*65}")


if __name__ == "__main__":
    main()
