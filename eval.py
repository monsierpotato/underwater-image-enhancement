# ============================================================
# eval.py  –  Test-set evaluation
# ============================================================

import json

import torch

from config import cfg          # adjust import path as needed
from dataset import test_loader
from model import model
from measure import evaluate_loader
from train import load_ckpt     # shared helper


def main(device):
    BEST_PATH = f"{cfg.OUTPUT_DIR}/checkpoints/best_model.pth"

    # Load best checkpoint
    ckpt_epoch, ckpt_metrics = load_ckpt(BEST_PATH, model, device=device)
    print(f"Loaded best checkpoint: epoch={ckpt_epoch}  stored metrics={ckpt_metrics}")

    model.to(device)

    # Full test evaluation
    test_metrics, n_test = evaluate_loader(model, test_loader, device, desc="Testing")

    print(f"\n{'='*55}")
    print(f"  TEST RESULTS – {cfg.MODEL_NAME}  (n={n_test})")
    print(f"{'='*55}")
    print(f"  PSNR      : {test_metrics['psnr']:>8.4f}  dB")
    print(f"  SSIM      : {test_metrics['ssim']:>8.4f}")
    print(f"  CIEDE2000 : {test_metrics['ciede2000']:>8.4f}  (lower=better)")
    print(f"  UCIQE     : {test_metrics['uciqe']:>8.4f}  (higher=better)")
    print(f"  UIQM      : {test_metrics['uiqm']:>8.4f}  (higher=better)")
    print(f"{'='*55}")

    # Save results
    results_dict = {
        "model_name":   cfg.MODEL_NAME,
        "in_channels":  cfg.IN_CHANNELS,
        "use_physics":  cfg.USE_PHYSICS,
        "best_epoch":   ckpt_epoch,
        "test_metrics": test_metrics,
    }
    out_path = f"{cfg.OUTPUT_DIR}/test_results.json"
    with open(out_path, "w") as f:
        json.dump(results_dict, f, indent=2)
    print(f"Results saved → {out_path}")


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    main(device)