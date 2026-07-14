"""
ablation_train_multi_run_uieb.py
=================================
Train the 4 UNet ablation variants (3ch, 4ch_t, 4ch_b, 5ch) from scratch
on the UIEB dataset using both UDCP and GUPDM physics priors.

Dataset split (fixed, sorted order):
  - Train / val pool : first 800 images (UIEB convention)
  - Test             : last 90 images   (UIEB test-90 convention)

Within each run the 10 % hold-out validation set is carved out of the
800-image pool per-seed via random_split.

Total runs: 4 variants x 2 priors x N seeds
  Default:  4 x 2 x 3 = 24 runs

Usage
-----
    # Default: 4 variants x 2 priors x 3 seeds, 50 epochs
    python ablation_train_multi_run_uieb.py

    # Custom seeds / epochs:
    python ablation_train_multi_run_uieb.py \
        --data_uieb      ./datasets/UIEB \
        --checkpoint_dir ./checkpoints/ablation_uieb \
        --val_folder     ./results/ablation_uieb \
        --prior_methods  udcp gupdm \
        --nEpochs        50 \
        --batchSize      16 \
        --num_runs       3 \
        --seeds 0 1 2

Output
------
* Checkpoints : <checkpoint_dir>/<prior>/<variant>_seed<N>_<ts>/best_model.pth
* Logs        : ./logs/<prior>_<variant>_seed<N>_<ts>.log
* JSON report : <val_folder>/ablation_uieb_results.json
* Console     : ranked mean +/- std summary table (per prior + combined)
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
import torch.utils.data as data

# ---------------------------------------------------------------------------
# Ensure project root is importable when run from any cwd
# ---------------------------------------------------------------------------
_PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

# Project imports
from uwir.data.datasets import UIEBDataset
from uwir.data.factory import _train_transform
from uwir.models import build_model, parse_model_variant
from uwir.losses import CompositeLoss
from uwir.metrics import evaluate_loader
from uwir.cli.train import (
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


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

ABLATION_VARIANTS = ["unet_3ch", "unet_4ch_t", "unet_4ch_b", "unet_5ch"]
METRIC_KEYS       = ("psnr", "ssim", "ciede2000", "uciqe", "uiqm",
                     "inference_ms_per_img")

# UIEB convention: first 800 sorted images -> train/val pool
#                  last 90  sorted images  -> test set
UIEB_TRAIN_VAL_N = 800
UIEB_TEST_N      = 90   # 800 + 90 = 890


# ---------------------------------------------------------------------------
# UIEB-specific test dataset wrapper
# ---------------------------------------------------------------------------

class _UIEBTestDataset(data.Dataset):
    """
    Thin wrapper around a Subset of UIEBDataset.
    Exposes the same (img_in, img_gt, fname_in, fname_gt) interface
    as UIEBDataset -- compatible with _collate_val.
    """
    def __init__(self, subset: data.Subset):
        self._subset = subset

    def __len__(self):
        return len(self._subset)

    def __getitem__(self, idx):
        return self._subset[idx]


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _make_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description=(
            "Multi-seed UIEB ablation: train 4 UNet variants with "
            "UDCP & GUPDM priors, N seeds each. "
            "Default: 4 variants x 2 priors x 3 seeds = 24 runs."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    # Dataset / paths
    p.add_argument(
        "--data_uieb", default="./datasets/UIEB",
        help="UIEB dataset root (contains raw-890/ and reference-890/).",
    )
    p.add_argument(
        "--checkpoint_dir", default="./checkpoints/ablation_uieb",
        help="Root directory for per-run checkpoint folders.",
    )
    p.add_argument(
        "--val_folder", default="./results/ablation_uieb",
        help="Output directory for the JSON summary.",
    )
    p.add_argument(
        "--prior_methods", nargs="+",
        default=["udcp", "gupdm"],
        choices=["udcp", "gdcp", "gupdm"],
        help="Physics prior(s) to iterate over.",
    )

    # Training hyper-params
    p.add_argument("--nEpochs",              type=int,   default=50)
    p.add_argument("--batchSize",            type=int,   default=16)
    p.add_argument("--cropSize",             type=int,   default=256)
    p.add_argument("--lr",                   type=float, default=1e-4)
    p.add_argument("--weight_decay",         type=float, default=1e-5)
    p.add_argument("--threads",              type=int,   default=4)
    p.add_argument("--in_memory",            action="store_true",
                   help="Pre-load dataset into RAM.")
    p.add_argument("--snapshots",            type=int,   default=10,
                   help="Save a periodic checkpoint every N epochs.")
    p.add_argument("--early_stop_patience",  type=int,   default=20,
                   help="Early-stopping patience (epochs without PSNR improvement).")

    # Scheduler
    p.add_argument("--cos_restart",          action="store_true")
    p.add_argument("--cos_restart_cyclic",   action="store_true")
    p.add_argument("--scheduler_step",       type=int,   default=30)
    p.add_argument("--scheduler_gamma",      type=float, default=0.5)
    p.add_argument("--warmup_epochs",        type=int,   default=0)
    p.add_argument("--start_warmup",         action="store_true")

    # Loss weights
    p.add_argument("--L1_weight",            type=float, default=1.0)
    p.add_argument("--perceptual_weight",    type=float, default=1.0)
    p.add_argument("--SSIM_weight",          type=float, default=0.0)

    # Multi-run
    p.add_argument(
        "--num_runs", type=int, default=3,
        help=(
            "Number of independent seeds per (variant, prior) pair. "
            "Default 3 -> 4 variants x 2 priors x 3 seeds = 24 runs."
        ),
    )
    p.add_argument(
        "--seeds", type=int, nargs="+", default=None,
        help="Explicit seed list (overrides --num_runs if given).",
    )
    p.add_argument(
        "--variants", type=str, nargs="+", default=ABLATION_VARIANTS,
        help="Which model variants to train.",
    )

    # Misc
    p.add_argument("--gpu_mode",             action="store_true", default=True)
    p.add_argument("--no_gpu",               dest="gpu_mode", action="store_false")
    p.add_argument("--pretrained_backbone",  action="store_true", default=False)
    return p


# ---------------------------------------------------------------------------
# Single training run
# ---------------------------------------------------------------------------

def train_one_run(
    variant: str,
    seed: int,
    prior_method: str,
    args,
    device: torch.device,
    train_ds_full: data.Dataset,   # 800-image pool (augmented)
    physics_extractor,
) -> str:
    """
    Train ``variant`` from scratch with ``seed`` on the 800-image UIEB pool.

    Returns
    -------
    str : absolute path to the saved best_model.pth
    """
    # ---- Reproducibility --------------------------------------------------
    torch.manual_seed(seed)
    np.random.seed(seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark     = False

    # ---- Parse variant ----------------------------------------------------
    _, in_channels, physics_mode = parse_model_variant(variant)

    # ---- Train / val split (10 % hold-out from 800-image pool) -----------
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
    model = build_model(
        variant, pretrained_backbone=args.pretrained_backbone
    ).to(device)

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
    ts       = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_id   = f"{prior_method}_{variant}_seed{seed}_{ts}"
    ckpt_dir = os.path.join(args.checkpoint_dir, prior_method, run_id)
    best_path = os.path.join(ckpt_dir, "best_model.pth")
    os.makedirs(ckpt_dir, exist_ok=True)

    log_dir  = os.path.join(str(_PROJECT_ROOT), "logs")
    os.makedirs(log_dir, exist_ok=True)
    log_path = os.path.join(log_dir, f"{run_id}.log")

    _log_file    = open(log_path, "w", encoding="utf-8")
    _orig_stdout = sys.stdout
    sys.stdout   = _Tee(_orig_stdout, _log_file)

    print(f"\n{'='*72}")
    print(f"  prior={prior_method}  variant={variant}  seed={seed}")
    print(f"  train={n_train}  val={n_val}  device={device}")
    print(f"  ckpt_dir : {ckpt_dir}")
    print(f"  log_file : {log_path}")
    print(f"{'='*72}")
    print(
        f"{'Epoch':>6}  {'TrainL':>8}  {'ValL':>8}  "
        f"{'PSNR':>7}  {'SSIM':>7}  {'LR':>9}  {'Time':>6}"
    )
    print("-" * 72)

    best_psnr  = 0.0
    best_epoch = 0
    es         = EarlyStopping(
        patience=args.early_stop_patience, min_delta=1e-4, mode="max"
    )
    LOG_EVERY  = max(1, args.nEpochs // 20)

    history = {
        "train_loss": [], "val_loss": [],
        "val_psnr": [], "val_ssim": [], "lr": [],
    }
    t_train_start = time.time()

    for epoch in range(1, args.nEpochs + 1):
        t0 = time.time()

        tr_loss, _ = train_epoch(model, train_loader, optimizer,
                                 criterion, device)
        vl_loss    = val_loss_epoch(model, val_loader, criterion, device)

        if epoch == 1 or epoch % LOG_EVERY == 0:
            val_metrics, _ = evaluate_loader(
                model, val_loader, device, max_samples=100
            )
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
            save_ckpt(
                model, optimizer, epoch,
                {"psnr": val_psnr, "ssim": val_ssim, "val_loss": vl_loss},
                best_path,
            )
            flag = "  <- BEST"

        if epoch % args.snapshots == 0:
            save_ckpt(
                model, optimizer, epoch,
                {"psnr": val_psnr, "ssim": val_ssim},
                os.path.join(ckpt_dir, f"epoch_{epoch:04d}.pth"),
            )

        if epoch == 1 or epoch % LOG_EVERY == 0 or flag:
            print(
                f"{epoch:>6}  {tr_loss:>8.4f}  {vl_loss:>8.4f}  "
                f"{val_psnr:>7.3f}  {val_ssim:>7.4f}  {cur_lr:>9.2e}  "
                f"{elapsed:>5.1f}s{flag}"
            )

        if es(val_psnr):
            print(
                f"\nEarly stopping at epoch {epoch} "
                f"(no improvement for {args.early_stop_patience} evals)"
            )
            break

    # Save last + history
    save_ckpt(
        model, optimizer, epoch,
        {"psnr": val_psnr, "ssim": val_ssim},
        os.path.join(ckpt_dir, "last_model.pth"),
    )

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
# Evaluate one checkpoint on the UIEB test-90 set
# ---------------------------------------------------------------------------

def eval_checkpoint(
    model_name: str,
    best_path: str,
    test_ds: data.Dataset,
    physics_extractor,
    device: torch.device,
    batch_size: int = 1,
    threads: int = 0,
) -> dict:
    """Load best_model.pth and evaluate on the UIEB test-90 set."""
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
# Aggregation helpers
# ---------------------------------------------------------------------------

def _aggregate(runs_list: list) -> dict:
    agg = {}
    for key in METRIC_KEYS:
        vals = [r[key] for r in runs_list if key in r]
        if vals:
            agg[f"{key}_mean"]   = float(np.mean(vals))
            agg[f"{key}_std"]    = float(
                np.std(vals, ddof=1) if len(vals) > 1 else 0.0
            )
            agg[f"{key}_values"] = vals
    return agg


# ---------------------------------------------------------------------------
# Summary table
# ---------------------------------------------------------------------------

_SUMMARY_COLS = [
    ("psnr",                 "PSNR(dB) up"),
    ("ssim",                 "SSIM up"),
    ("ciede2000",            "CIEDE2000 dn"),
    ("uciqe",                "UCIQE up"),
    ("uiqm",                 "UIQM up"),
    ("inference_ms_per_img", "Inf(ms/img)"),
]


def _print_summary(all_agg: dict, title: str = "ABLATION RESULTS"):
    W = 144
    print(f"\n{'='*W}")
    print(f"  {title}  (mean +/- std, Bessel-corrected, UIEB test-90)")
    print(f"{'='*W}")
    header = f"{'Variant @ Prior':<32}"
    for _, label in _SUMMARY_COLS:
        header += f"  {label:>22}"
    print(header)
    print("-" * W)
    rows = sorted(
        all_agg.items(),
        key=lambda x: x[1].get("psnr_mean", 0.0),
        reverse=True,
    )
    for key, agg in rows:
        row = f"{key:<32}"
        for col_key, _ in _SUMMARY_COLS:
            mean = agg.get(f"{col_key}_mean")
            std  = agg.get(f"{col_key}_std")
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

    seeds    = args.seeds if args.seeds is not None else list(range(args.num_runs))
    num_runs = len(seeds)

    device = torch.device(
        "cuda" if (args.gpu_mode and torch.cuda.is_available()) else "cpu"
    )

    total_runs = len(args.variants) * len(args.prior_methods) * num_runs
    print(f"Device         : {device}")
    print(f"Prior methods  : {args.prior_methods}")
    print(f"Variants       : {args.variants}")
    print(f"Seeds          : {seeds}  (num_runs={num_runs})")
    print(f"Epochs/run     : {args.nEpochs}")
    print(
        f"Total runs     : {total_runs}  "
        f"({len(args.variants)} variants x "
        f"{len(args.prior_methods)} priors x "
        f"{num_runs} seeds)"
    )
    print(f"Checkpoint dir : {args.checkpoint_dir}")

    # -----------------------------------------------------------------------
    # Load UIEB dataset ONCE  (both priors share the same raw images)
    # -----------------------------------------------------------------------
    print(f"\nLoading UIEB dataset from '{args.data_uieb}' ...")

    # No-augment instance — used to build the deterministic test subset
    # (augment=False ensures consistent sorted index ordering)
    uieb_noaug = UIEBDataset(
        args.data_uieb,
        transform=_train_transform(args.cropSize),
        augment=False,
        in_memory=args.in_memory,
    )
    # Augmented instance — used as the 800-image train/val pool
    uieb_aug = UIEBDataset(
        args.data_uieb,
        transform=_train_transform(args.cropSize),
        augment=True,
        in_memory=args.in_memory,
    )

    total_n = len(uieb_noaug)
    if total_n != UIEB_TRAIN_VAL_N + UIEB_TEST_N:
        print(
            f"[WARN] Expected {UIEB_TRAIN_VAL_N + UIEB_TEST_N} UIEB pairs, "
            f"got {total_n}. Adjusting split proportionally."
        )
    train_val_n = min(UIEB_TRAIN_VAL_N, total_n - UIEB_TEST_N)
    test_n      = total_n - train_val_n

    train_val_indices = list(range(train_val_n))             # 0 .. 799
    test_indices      = list(range(train_val_n, total_n))    # 800 .. 889

    # 800-image augmented pool  (further split per-seed inside train_one_run)
    train_ds_full = data.Subset(uieb_aug, train_val_indices)
    # 90-image test set — no augmentation, no random_split
    test_ds = _UIEBTestDataset(data.Subset(uieb_noaug, test_indices))

    print(
        f"Train+val pool : {len(train_ds_full)} images "
        f"(indices 0-{train_val_n - 1})"
    )
    print(
        f"Test set       : {len(test_ds)} images "
        f"(indices {train_val_n}-{total_n - 1})"
    )

    os.makedirs(args.checkpoint_dir, exist_ok=True)
    os.makedirs(args.val_folder,     exist_ok=True)

    # -----------------------------------------------------------------------
    # PHASE 1: Train all combinations
    # best_paths[(prior, variant)] = [path_seed0, path_seed1, ...]
    # -----------------------------------------------------------------------
    best_paths: dict = {
        (prior, variant): []
        for prior   in args.prior_methods
        for variant in args.variants
    }

    done = 0
    for prior_method in args.prior_methods:
        physics_extractor = _resolve_physics_extractor(prior_method)
        for seed_idx, seed in enumerate(seeds):
            for variant in args.variants:
                done += 1
                print(
                    f"\n[TRAIN {done}/{total_runs}]  "
                    f"prior={prior_method}  variant={variant}  "
                    f"seed={seed}  (seed {seed_idx+1}/{num_runs})"
                )
                bp = train_one_run(
                    variant=variant,
                    seed=seed,
                    prior_method=prior_method,
                    args=args,
                    device=device,
                    train_ds_full=train_ds_full,
                    physics_extractor=physics_extractor,
                )
                best_paths[(prior_method, variant)].append(bp)
                print(f"  Saved -> {bp}")

    # -----------------------------------------------------------------------
    # PHASE 2: Evaluate all checkpoints on UIEB test-90
    # -----------------------------------------------------------------------
    print(f"\n{'='*72}")
    print("  PHASE 2: Evaluating all checkpoints on UIEB test-90 ...")
    print(f"{'='*72}")

    # per_runs[(prior, variant)] = list of metric dicts (one per seed)
    per_runs: dict = {k: [] for k in best_paths}

    eval_done  = 0
    eval_total = len(best_paths) * num_runs

    for prior_method in args.prior_methods:
        physics_extractor = _resolve_physics_extractor(prior_method)
        for variant in args.variants:
            key = (prior_method, variant)
            for seed_idx, (seed, bp) in enumerate(
                    zip(seeds, best_paths[key])):
                eval_done += 1
                print(
                    f"\n[EVAL {eval_done}/{eval_total}]  "
                    f"prior={prior_method}  variant={variant}  seed={seed}"
                )
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
                    per_runs[key].append({"seed": seed, "checkpoint": bp, **m})
                    print(
                        f"  PSNR={m['psnr']:.4f}  SSIM={m['ssim']:.4f}  "
                        f"CIEDE2000={m['ciede2000']:.4f}  "
                        f"UCIQE={m['uciqe']:.4f}  UIQM={m['uiqm']:.4f}"
                    )
                except Exception as exc:
                    print(
                        f"  [ERROR] {prior_method}/{variant} "
                        f"seed={seed}: {exc}"
                    )
                    import traceback; traceback.print_exc()

    # -----------------------------------------------------------------------
    # Aggregate
    # -----------------------------------------------------------------------
    # Combined lookup keyed as "variant@prior" for the final summary table
    all_agg: dict = {}
    per_prior_agg: dict = {p: {} for p in args.prior_methods}

    for (prior_method, variant), runs_list in per_runs.items():
        if not runs_list:
            continue
        agg = _aggregate(runs_list)
        agg_key = f"{variant}@{prior_method}"
        all_agg[agg_key] = agg
        per_prior_agg[prior_method][variant] = agg

    # -----------------------------------------------------------------------
    # Print summary tables
    # -----------------------------------------------------------------------
    # One table per prior
    for prior_method in args.prior_methods:
        agg_subset = {
            f"{v}@{prior_method}": per_prior_agg[prior_method][v]
            for v in args.variants
            if v in per_prior_agg[prior_method]
        }
        if agg_subset:
            _print_summary(
                agg_subset,
                title=f"UIEB ABLATION -- prior={prior_method.upper()}",
            )

    # Combined table
    if all_agg:
        _print_summary(all_agg, title="UIEB ABLATION -- ALL PRIORS COMBINED")

    # -----------------------------------------------------------------------
    # Save JSON
    # -----------------------------------------------------------------------
    out_path = os.path.join(args.val_folder, "ablation_uieb_results.json")
    output = {
        "_meta": {
            "timestamp":      datetime.now().isoformat(timespec="seconds"),
            "device":         str(device),
            "variants":       args.variants,
            "prior_methods":  args.prior_methods,
            "num_runs":       num_runs,
            "seeds":          seeds,
            "nEpochs":        args.nEpochs,
            "dataset":        "UIEB",
            "train_val_n":    train_val_n,
            "test_n":         test_n,
            "checkpoint_dir": args.checkpoint_dir,
            "data_root":      args.data_uieb,
            "crop_size":      args.cropSize,
        },
        "per_combination": {
            f"{prior}_{variant}": {
                "prior":      prior,
                "variant":    variant,
                "runs":       per_runs[(prior, variant)],
                "aggregated": all_agg.get(f"{variant}@{prior}", {}),
            }
            for prior   in args.prior_methods
            for variant in args.variants
        },
    }
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)

    print(f"\n{'='*72}")
    print(f"  Results saved -> {out_path}")
    print(f"{'='*72}")


if __name__ == "__main__":
    main()
