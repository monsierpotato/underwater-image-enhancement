# ============================================================
# eval.py  –  Test-set evaluation for checkpoint folders
# ============================================================

import json
import os

import torch
import torch.utils.data as data
import numpy as np

from data.options import option
from data.data import (
    get_euvp_training_set,
    get_uieb_training_set,
)
from net.registry import ALL_MODEL_NAMES, build_model, parse_model_variant
from pathlib import Path
from PIL import Image
import torchvision.transforms as T

IMG_EXTS = {".jpg", ".jpeg", ".png", ".JPG", ".JPEG", ".PNG"}

def collect_test_pairs(data_root):
    inp_dir = Path(data_root) / "test_samples" / "Inp"
    gt_dir  = Path(data_root) / "test_samples" / "GTr"
    pairs   = []
    if not (inp_dir.exists() and gt_dir.exists()):
        print("[WARN] test_samples not found – test set will be empty.")
        return pairs
    gt_dict = {f.stem: f for f in gt_dir.iterdir() if f.suffix.lower() in IMG_EXTS}
    for inp_file in sorted(inp_dir.iterdir()):
        if inp_file.suffix.lower() not in IMG_EXTS:
            continue
        if inp_file.stem in gt_dict:
            pairs.append((str(inp_file), str(gt_dict[inp_file.stem])))
    return pairs

class TestDataset(data.Dataset):
    def __init__(self, pairs, img_size=256):
        self.pairs = pairs
        self.resize = T.Resize((img_size, img_size), antialias=True)
        self.to_tensor = T.ToTensor()

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        inp_path, gt_path = self.pairs[idx]
        inp_pil = Image.open(inp_path).convert("RGB")
        gt_pil  = Image.open(gt_path).convert("RGB")

        inp_t = self.to_tensor(self.resize(inp_pil))
        gt_t  = self.to_tensor(self.resize(gt_pil))

        return inp_t, gt_t, "", ""
from measure_underwater import evaluate_loader
from train import load_ckpt, _collate_val, _resolve_physics_extractor


def main():
    parser = option()
    args = parser.parse_args()

    device = torch.device("cuda" if (args.gpu_mode and torch.cuda.is_available()) else "cpu")
    print(f"Evaluation Device: {device}")

    # Reproducibility for dataset split
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    physics_extractor = _resolve_physics_extractor(args.prior_method)
    print(f"Prior method: {args.prior_method}")

    # ------------------------------------------------------------------
    # Datasets
    # ------------------------------------------------------------------
    print(f"Loading test samples from '{args.data_train_euvp}' for evaluation...")
    test_pairs = collect_test_pairs(args.data_train_euvp)
    val_ds = TestDataset(test_pairs, img_size=args.cropSize)
    print(f"Validation set size: {len(val_ds)} images")

    checkpoint_dir = args.checkpoint_dir
    all_results = {}

    if not os.path.exists(checkpoint_dir):
        print(f"Checkpoint directory {checkpoint_dir} does not exist.")
        return

    for run_name in sorted(os.listdir(checkpoint_dir)):
        run_dir = os.path.join(checkpoint_dir, run_name)
        if not os.path.isdir(run_dir):
            continue
        
        best_path = os.path.join(run_dir, "best_model.pth")
        if not os.path.isfile(best_path):
            continue

        model_name = None
        # Try to parse model name from run_name
        # The run_name usually looks like model_dataset_timestamp or run_name_timestamp
        
        normalized = run_name.replace("3ch", "_3ch").replace("4ch", "_4ch").replace("5ch", "_5ch")
        normalized = normalized.replace("__", "_")
        for mn in sorted(ALL_MODEL_NAMES, key=len, reverse=True):
            if run_name.startswith(mn) or normalized.startswith(mn):
                model_name = mn
                break
        
        if model_name is None:
            print(f"Skipping {run_name}: cannot parse model variant")
            continue

        print(f"\n{'='*65}")
        print(f"  Evaluating {run_name}")
        print(f"{'='*65}")

        _, in_channels, physics_mode = parse_model_variant(model_name)
        model = build_model(model_name, pretrained_backbone=False).to(device)

        ckpt_epoch, ckpt_metrics = load_ckpt(best_path, model, device=str(device))
        print(f"Loaded best checkpoint: epoch={ckpt_epoch}  stored metrics={ckpt_metrics}")

        collate_fn_val = lambda b: _collate_val(b, physics_mode, physics_extractor)

        val_loader = data.DataLoader(
            val_ds,
            batch_size  = 1,
            shuffle     = False,
            num_workers = 0,
            pin_memory  = False,
            drop_last   = False,
            collate_fn  = collate_fn_val,
        )

        test_metrics, n_test = evaluate_loader(model, val_loader, device, desc=f"Testing {run_name}")

        if n_test > 0:
            print(f"\n  TEST RESULTS – {model_name}  (n={n_test})")
            print(f"  PSNR      : {test_metrics['psnr']:>8.4f}  dB")
            print(f"  SSIM      : {test_metrics['ssim']:>8.4f}")
            print(f"  CIEDE2000 : {test_metrics['ciede2000']:>8.4f}  (lower=better)")
            print(f"  UCIQE     : {test_metrics['uciqe']:>8.4f}  (higher=better)")
            print(f"  UIQM      : {test_metrics['uiqm']:>8.4f}  (higher=better)")
        else:
            print(f"\n  TEST RESULTS – {model_name}  (n=0) — Skipped (no data)")

        print("\n  STORED CHECKPOINT METRICS:")
        for k, v in ckpt_metrics.items():
            print(f"  {k:<9} : {v:>8.4f}")

        all_results[run_name] = {
            "model_name":   model_name,
            "in_channels":  in_channels,
            "physics_mode": physics_mode,
            "best_epoch":   ckpt_epoch,
            "ckpt_metrics": ckpt_metrics,
            "test_metrics": test_metrics if n_test > 0 else None,
        }

    os.makedirs(args.val_folder, exist_ok=True)
    out_path = os.path.join(args.val_folder, "test_results_all.json")
    with open(out_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\n{'='*65}")
    print(f"All results saved → {out_path}")
    print(f"{'='*65}")


if __name__ == "__main__":
    main()
