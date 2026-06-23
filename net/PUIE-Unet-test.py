"""
PUIE-Unet-test.py
=================
Inference / evaluation for a trained **PUIE-UNet** checkpoint.

* Loads ``best_model.pth`` (or any ``--resume`` path) produced by
  ``PUIE-Unet.py``.
* Runs the **prior** branch only (target unknown at test time):
    - ``--num_samples 1``  → deterministic MC mode (prior means).
    - ``--num_samples N>1`` → MP mode: draw N prior samples and average
      (PUIE-Net's probabilistic ensembling — reduces variance, often a
      small PSNR/SSIM bump).
* Saves enhanced RGB images to ``--val_folder``.
* If paired ground truth is available it also reports the full metric
  suite (PSNR / SSIM / CIEDE2000 / UCIQE / UIQM) via ``evaluate_loader``.

Examples
--------
    # Paired UIEB test set → save outputs + metrics
    python PUIE-Unet-test.py --model unet_5ch \
        --resume ./checkpoints/puie_unet_uieb_XXXX/best_model.pth \
        --data_val_uieb ./datasets/UIEB/test/input \
        --data_valgt_uieb ./datasets/UIEB/test/reference \
        --num_samples 8 --val_folder ./results/puie_unet_uieb

    # No-reference folder (U45) → save outputs only
    python PUIE-Unet-test.py --model unet_3ch \
        --resume ./checkpoints/run/best_model.pth \
        --data_val_u45 ./datasets/U45 --num_samples 8 \
        --val_folder ./results/puie_u45
"""

import importlib.util
import os

import numpy as np
import torch
import torch.nn as nn
import torch.utils.data as data
import torchvision.transforms as T
from PIL import Image

from data.options import option
from data.util import is_image_file
from net.registry import parse_model_variant
from measure_underwater import evaluate_loader
from train import load_ckpt, _add_physics_channels


# ---------------------------------------------------------------------------
# Import PUIEUNet from the (hyphenated) training file.
# A hyphen isn't import-friendly, so load it by path.
# ---------------------------------------------------------------------------

def _load_puie_unet_class():
    here = os.path.dirname(os.path.abspath(__file__))
    spec = importlib.util.spec_from_file_location(
        "puie_unet_train", os.path.join(here, "PUIE-Unet.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.PUIEUNet


# ---------------------------------------------------------------------------
# MP wrapper: makes model(inp) average N prior samples so the existing
# evaluate_loader (which calls model(inp) with no kwargs) gets MP output.
# ---------------------------------------------------------------------------

class _MPWrapper(nn.Module):
    def __init__(self, model, num_samples):
        super().__init__()
        self.model = model
        self.num_samples = num_samples

    def forward(self, x):
        return self.model(x, num_samples=self.num_samples)


# ---------------------------------------------------------------------------
# Datasets
# ---------------------------------------------------------------------------

class _PairedTestDataset(data.Dataset):
    """Paired input/GT folders, matched by filename stem."""

    def __init__(self, inp_dir, gt_dir, img_size):
        self.resize    = T.Resize((img_size, img_size), antialias=True)
        self.to_tensor = T.ToTensor()
        gt_map = {os.path.splitext(f)[0]: os.path.join(gt_dir, f)
                  for f in os.listdir(gt_dir) if is_image_file(f)}
        self.pairs, self.names = [], []
        for f in sorted(os.listdir(inp_dir)):
            if not is_image_file(f):
                continue
            stem = os.path.splitext(f)[0]
            if stem in gt_map:
                self.pairs.append((os.path.join(inp_dir, f), gt_map[stem]))
                self.names.append(f)

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        inp_p, gt_p = self.pairs[idx]
        inp = self.to_tensor(self.resize(Image.open(inp_p).convert("RGB")))
        gt  = self.to_tensor(self.resize(Image.open(gt_p ).convert("RGB")))
        return inp, gt, self.names[idx], self.names[idx]


class _UnpairedTestDataset(data.Dataset):
    """Flat folder of input images, no ground truth."""

    def __init__(self, inp_dir, img_size):
        self.resize    = T.Resize((img_size, img_size), antialias=True)
        self.to_tensor = T.ToTensor()
        self.files = [os.path.join(inp_dir, f) for f in sorted(os.listdir(inp_dir))
                      if is_image_file(f)]
        self.names = [os.path.basename(f) for f in self.files]

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        inp = self.to_tensor(self.resize(Image.open(self.files[idx]).convert("RGB")))
        # Return inp twice so a paired-style collate still works (gt unused).
        return inp, inp, self.names[idx], self.names[idx]


def main():
    parser = option()
    parser.add_argument('--num_samples', type=int, default=8,
                        help='Prior samples to average (1 = MC, >1 = MP ensemble)')
    parser.add_argument('--latent_dim', type=int, default=20,
                        help='Must match the value used during training')
    parser.add_argument('--save_images', type=lambda v: v.lower() in ('1','true','yes','y'),
                        default=True, help='Write enhanced images to --val_folder')
    args = parser.parse_args()

    device = torch.device("cuda" if (args.gpu_mode and torch.cuda.is_available()) else "cpu")
    print(f"Device: {device}")

    if not (args.resume and os.path.isfile(args.resume)):
        raise FileNotFoundError(
            "Pass a checkpoint via --resume, e.g. "
            "--resume ./checkpoints/<run>/best_model.pth")

    _, in_channels, physics_mode = parse_model_variant(args.model)

    # ------------------------------------------------------------------
    # Build model + load weights
    # ------------------------------------------------------------------
    PUIEUNet = _load_puie_unet_class()
    model = PUIEUNet(in_channels=in_channels, latent_dim=args.latent_dim).to(device)
    epoch, metrics = load_ckpt(args.resume, model, device=str(device))
    model.eval()
    print(f"Loaded    : {args.resume}  (epoch={epoch}, stored metrics={metrics})")
    print(f"Model     : PUIE-UNet  (in_channels={in_channels}, physics={physics_mode}, "
          f"num_samples={args.num_samples})")

    # ------------------------------------------------------------------
    # Pick the test source. Priority: paired UIEB (has GT) → else U45 (no GT).
    # ------------------------------------------------------------------
    paired = (os.path.isdir(args.data_val_uieb) and
              os.path.isdir(args.data_valgt_uieb))
    if paired:
        ds = _PairedTestDataset(args.data_val_uieb, args.data_valgt_uieb, args.cropSize)
        print(f"Test set  : paired  ({len(ds)} images from {args.data_val_uieb})")
    elif os.path.isdir(args.data_val_u45):
        ds = _UnpairedTestDataset(args.data_val_u45, args.cropSize)
        print(f"Test set  : unpaired ({len(ds)} images from {args.data_val_u45})")
    else:
        raise FileNotFoundError(
            "No test data found. Provide either --data_val_uieb + --data_valgt_uieb "
            "(paired) or --data_val_u45 (unpaired).")

    # Collate: append physics channels, keep names.
    def collate(batch):
        inps, gts, names = [], [], []
        for inp, gt, name, _ in batch:
            inps.append(_add_physics_channels(inp, physics_mode))
            gts.append(gt)
            names.append(name)
        return torch.stack(inps), torch.stack(gts), names

    loader = data.DataLoader(ds, batch_size=1, shuffle=False,
                             num_workers=0, collate_fn=collate)

    # ------------------------------------------------------------------
    # Inference: save enhanced images (MP mode).
    # ------------------------------------------------------------------
    if args.save_images:
        os.makedirs(args.val_folder, exist_ok=True)
        to_pil = T.ToPILImage()
        with torch.no_grad():
            for inp, _, names in loader:
                inp  = inp.to(device)
                pred = model(inp, num_samples=args.num_samples).clamp(0, 1).cpu()
                for img, name in zip(pred, names):
                    to_pil(img).save(os.path.join(args.val_folder, name))
        print(f"Saved enhanced images → {args.val_folder}")

    # ------------------------------------------------------------------
    # Metrics (only meaningful when GT is real, i.e. paired set).
    # evaluate_loader expects (inp, gt) pairs and calls model(inp); wrap so
    # model(inp) runs the MP-averaged prior path.
    # ------------------------------------------------------------------
    if paired:
        metric_loader = data.DataLoader(
            ds, batch_size=1, shuffle=False, num_workers=0,
            collate_fn=lambda b: (
                torch.stack([_add_physics_channels(i, physics_mode) for i, *_ in b]),
                torch.stack([g for _, g, *_ in b]),
            ))
        wrapped = _MPWrapper(model, args.num_samples).to(device).eval()
        res, n = evaluate_loader(wrapped, metric_loader, device)
        print(f"\n  TEST RESULTS  (n={n}, num_samples={args.num_samples})")
        print(f"  PSNR      : {res['psnr']:>8.4f} dB")
        print(f"  SSIM      : {res['ssim']:>8.4f}")
        print(f"  CIEDE2000 : {res['ciede2000']:>8.4f}  (lower=better)")
        print(f"  UCIQE     : {res['uciqe']:>8.4f}  (higher=better)")
        print(f"  UIQM      : {res['uiqm']:>8.4f}  (higher=better)")
    else:
        print("\n  No ground truth → skipped reference metrics "
              "(use eval scripts with UCIQE/UIQM for no-reference scoring).")


if __name__ == "__main__":
    main()
