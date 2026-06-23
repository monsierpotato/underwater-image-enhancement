"""
PUIE-Unet.py
============
PUIE-Net (Probabilistic Underwater Image Enhancement) **grafted onto the
deeper 4-level U-Net backbone** from this repository.

Idea (combining the two projects)
----------------------------------
* From **PUIE-Net** (../PUIE-Net-main):
    - A *conditional VAE*: a **prior** branch sees only the (physics-augmented)
      input, a **posterior** branch sees input + ground-truth.
    - Two latent distributions per branch — a *mean* latent (u) and a
      *std* latent (s) — injected back into the decoder features via FiLM
      modulation:  ``InstanceNorm(feat) * |s| + u``.
    - Trained with an **ELBO**:  reconstruction + KL(posterior ‖ prior).
    - At inference only the prior branch runs (target is unknown).

* From **underwater-image-enhancement** (this repo):
    - The deeper ``UNet5ch`` backbone (enc 64/128/256/512 + bottleneck 1024).
    - The physics front-end (UDCP t(x) + background light B) via the
      ``--model <backbone>_<variant>`` channel convention.
    - ``CompositeLoss`` (L1 + VGG perceptual + SSIM) used as the
      reconstruction term of the ELBO.
    - Dataset loaders (EUVP / UIEB / UFO-120), metric suite, scheduler,
      early-stopping and checkpoint helpers — all imported, not re-written.

Run from inside the ``underwater-image-enhancement`` directory, e.g.::

    python PUIE-Unet.py --dataset uieb  --data_train_uieb ./datasets/UIEB \
        --model unet_5ch --batchSize 8 --nEpochs 200 --run_name puie_unet_uieb

    python PUIE-Unet.py --dataset euvp  --data_train_euvp ./datasets/EUVP \
        --model unet_3ch --kl_weight 1.0 --kl_anneal_epochs 20

Note: the *backbone* part of ``--model`` is ignored here (the backbone is
always this PUIE-UNet); only the *channel variant* (3ch / 4ch_t / 4ch_b /
5ch) is used to decide the physics front-end.
"""

import json
import os
import time
from datetime import datetime

import numpy as np
import torch
import torch.nn as nn
import torch.utils.data as data
from torch.distributions import Normal, Independent, kl_divergence

# ---- Reuse this repo's building blocks & plumbing -------------------------
from net.unet import DoubleConv, Down, Up
from net.registry import parse_model_variant
from data.options import option
from data.data import (
    get_euvp_training_set,
    get_uieb_training_set,
    get_ufo120_training_set,
)
from loss import CompositeLoss
from measure_underwater import evaluate_loader
# Helpers already defined in train.py (importing train.py does NOT run main —
# it is guarded by `if __name__ == "__main__"`).
from train import (
    EarlyStopping,
    build_scheduler,
    save_ckpt,
    load_ckpt,
    _collate_train,
    _unwrap,
)


# ===========================================================================
# Backbone: U-Net that returns full-resolution 64-channel features (no head)
# ===========================================================================

class UNetFeatures(nn.Module):
    """
    4-level U-Net identical to ``net.unet.UNet5ch`` but **without** the final
    head — it returns the decoder's full-resolution feature map (``f[0]``
    channels, default 64) so PUIE can attach its latent modulation + head.

    Args:
        in_channels (int):        Number of input channels (3 / 4 / 5, or
                                  in_channels+3 for the posterior branch).
        features (tuple[int]):    Encoder widths. Default (64, 128, 256, 512).
        bilinear (bool):          Bilinear upsampling vs ConvTranspose2d.
    """

    def __init__(
        self,
        in_channels: int,
        features: tuple = (64, 128, 256, 512),
        bilinear: bool = True,
    ):
        super().__init__()
        f = features
        self.out_ch = f[0]

        # Encoder
        self.enc1 = DoubleConv(in_channels, f[0])
        self.enc2 = Down(f[0], f[1])
        self.enc3 = Down(f[1], f[2])
        self.enc4 = Down(f[2], f[3])
        self.bottleneck = Down(f[3], f[3] * 2)

        # Decoder  (from_below, skip, out)
        self.dec4 = Up(f[3] * 2, f[3], f[3], bilinear)
        self.dec3 = Up(f[3],     f[2], f[2], bilinear)
        self.dec2 = Up(f[2],     f[1], f[1], bilinear)
        self.dec1 = Up(f[1],     f[0], f[0], bilinear)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        e1 = self.enc1(x)
        e2 = self.enc2(e1)
        e3 = self.enc3(e2)
        e4 = self.enc4(e3)
        bn = self.bottleneck(e4)

        d4 = self.dec4(bn, e4)
        d3 = self.dec3(d4, e3)
        d2 = self.dec2(d3, e2)
        d1 = self.dec1(d2, e1)      # (N, f[0], H, W)
        return d1


# ===========================================================================
# Latent head: PUIE's "Compute_z" adapted to arbitrary feature width
# ===========================================================================

class LatentDist(nn.Module):
    """
    Produce two axis-aligned Gaussian latents from a feature map, following
    PUIE-Net's dual (mean / std) encoding:

      * **u** latent  — from the *spatial mean* of the features.
      * **s** latent  — from the *spatial std*  of the features.

    Args:
        feat_ch    (int): Channels of the incoming feature map.
        latent_dim (int): Dimensionality of each latent (u and s).
    """

    def __init__(self, feat_ch: int, latent_dim: int):
        super().__init__()
        self.latent_dim = latent_dim
        self.u_conv = nn.Conv2d(feat_ch, 2 * latent_dim, kernel_size=1)
        self.s_conv = nn.Conv2d(feat_ch, 2 * latent_dim, kernel_size=1)

    @staticmethod
    def _to_dist(mu_log_sigma, latent_dim):
        mu_log_sigma = mu_log_sigma.squeeze(-1).squeeze(-1)        # (N, 2z)
        mu        = mu_log_sigma[:, :latent_dim]
        log_sigma = mu_log_sigma[:, latent_dim:]
        sigma     = torch.exp(log_sigma)
        dist = Independent(Normal(loc=mu, scale=sigma), 1)
        return dist, mu, sigma

    def forward(self, x: torch.Tensor):
        # u: spatial mean   (global average pooling)
        u_enc = x.mean(dim=2, keepdim=True).mean(dim=3, keepdim=True)
        u_dist, u_mu, u_sigma = self._to_dist(self.u_conv(u_enc), self.latent_dim)

        # s: spatial std    (PUIE applies std twice, dim 2 then dim 3)
        s_enc = x.std(dim=2, keepdim=True).std(dim=3, keepdim=True)
        s_dist, s_mu, s_sigma = self._to_dist(self.s_conv(s_enc), self.latent_dim)

        return u_dist, s_dist, u_mu, u_sigma, s_mu, s_sigma


# ===========================================================================
# PUIE-UNet model
# ===========================================================================

class PUIEUNet(nn.Module):
    """
    Probabilistic U-Net for underwater image enhancement.

    Training  :  ``out, kl = model(x, y)``   (posterior latent injected)
    Inference :  ``out      = model(x)``      (prior latent — deterministic
                                               mean, i.e. PUIE-Net "MC" mode;
                                               pass ``num_samples > 1`` to draw
                                               and average prior samples, i.e.
                                               PUIE-Net "MP" mode.)

    Args:
        in_channels (int): Physics-augmented input channels (3 / 4 / 5).
        latent_dim  (int): Latent dimensionality for each of the u / s codes.
        features (tuple):  U-Net encoder widths.
    """

    def __init__(
        self,
        in_channels: int = 5,
        latent_dim: int = 20,
        features: tuple = (64, 128, 256, 512),
    ):
        super().__init__()
        self.in_channels = in_channels
        self.latent_dim  = latent_dim
        feat_ch = features[0]

        # Prior sees the input; posterior also sees the 3-ch GT.
        self.prior_net = UNetFeatures(in_channels,     features)
        self.post_net  = UNetFeatures(in_channels + 3, features)

        self.prior_dist = LatentDist(feat_ch, latent_dim)
        self.post_dist  = LatentDist(feat_ch, latent_dim)

        # Map latent codes back to feature-channel modulation parameters.
        self.conv_u  = nn.Conv2d(latent_dim, feat_ch, kernel_size=1)
        self.conv_s  = nn.Conv2d(latent_dim, feat_ch, kernel_size=1)
        self.insnorm = nn.InstanceNorm2d(feat_ch)

        # Refinement + RGB head.
        self.refine = DoubleConv(feat_ch, feat_ch)
        self.head   = nn.Sequential(
            nn.Conv2d(feat_ch, 3, kernel_size=1),
            nn.Sigmoid(),
        )

    # -- FiLM-style latent injection (PUIE): IN(feat) * |s| + u -------------
    def _inject(self, feat, latent_u, latent_s):
        u = self.conv_u(latent_u.unsqueeze(-1).unsqueeze(-1))   # (N, C, 1, 1)
        s = self.conv_s(latent_s.unsqueeze(-1).unsqueeze(-1))   # (N, C, 1, 1)
        return self.insnorm(feat) * torch.abs(s) + u

    def _decode(self, feat, latent_u, latent_s):
        feat = self._inject(feat, latent_u, latent_s)
        return self.head(self.refine(feat))

    def forward(self, x: torch.Tensor, y: torch.Tensor = None, num_samples: int = 1):
        prior_feat = self.prior_net(x)
        (pr_u_dist, pr_s_dist,
         pr_u_mu, _, pr_s_mu, _) = self.prior_dist(prior_feat)

        # -------- Training: use the posterior, return ELBO's KL term -------
        if y is not None:
            post_feat = self.post_net(torch.cat([x, y], dim=1))
            po_u_dist, po_s_dist, *_ = self.post_dist(post_feat)

            latent_u = po_u_dist.rsample()          # reparameterised sample
            latent_s = po_s_dist.rsample()
            out = self._decode(prior_feat, latent_u, latent_s)

            kl = (kl_divergence(po_u_dist, pr_u_dist).mean()
                  + kl_divergence(po_s_dist, pr_s_dist).mean())
            return out, kl

        # -------- Inference: prior only -----------------------------------
        if num_samples <= 1:
            # PUIE-Net "MC": deterministic — use the prior means.
            return self._decode(prior_feat, pr_u_mu, pr_s_mu)

        # PUIE-Net "MP": average several prior samples (uncertainty/ensemble).
        acc = 0.0
        for _ in range(num_samples):
            lu = pr_u_dist.sample()
            ls = pr_s_dist.sample()
            acc = acc + self._decode(prior_feat, lu, ls)
        return acc / num_samples


# ===========================================================================
# Train / validate epoch loops (ELBO = reconstruction + beta * KL)
# ===========================================================================

def train_epoch(model, loader, optimizer, criterion, device, kl_weight, grad_clip):
    model.train()
    tot_loss = 0.0
    tot_recon = 0.0
    tot_kl    = 0.0

    try:
        from tqdm.auto import tqdm
        pbar = tqdm(loader, desc="Training Batch", leave=False, mininterval=2.0)
    except ImportError:
        pbar = loader

    for inp, gt in pbar:
        inp, gt = inp.to(device), gt.to(device)
        optimizer.zero_grad(set_to_none=True)

        pred, kl = model(inp, gt)                    # posterior path
        recon, _ = criterion(pred, gt)               # L1 + VGG + SSIM
        loss = recon + kl_weight * kl

        loss.backward()
        if grad_clip:
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        tot_loss  += loss.item()
        tot_recon += recon.item()
        tot_kl    += kl.item()

    n = len(loader)
    return tot_loss / n, {"recon": tot_recon / n, "kl": tot_kl / n}


@torch.no_grad()
def val_loss_epoch(model, loader, criterion, device, kl_weight):
    model.eval()
    tot = 0.0
    for inp, gt in loader:
        inp, gt = inp.to(device), gt.to(device)
        pred, kl = model(inp, gt)                     # posterior available at val
        recon, _ = criterion(pred, gt)
        tot += (recon + kl_weight * kl).item()
    return tot / len(loader)


# ===========================================================================
# Main
# ===========================================================================

def main():
    # ------------------------------------------------------------------
    # Arguments: reuse the repo's option() and add PUIE-specific knobs.
    # ------------------------------------------------------------------
    parser = option()
    parser.add_argument('--latent_dim', type=int, default=20,
                        help='Dimensionality of each (u / s) latent code (PUIE default 20)')
    parser.add_argument('--kl_weight', type=float, default=1.0,
                        help='Max weight (beta) on the KL term of the ELBO')
    parser.add_argument('--kl_anneal_epochs', type=int, default=20,
                        help='Linearly ramp the KL weight from 0 to --kl_weight '
                             'over this many epochs (0 = no annealing)')
    parser.add_argument('--num_samples', type=int, default=1,
                        help='Prior samples to average at validation-metric time '
                             '(1 = deterministic MC mode; >1 = MP ensemble)')
    args = parser.parse_args()

    # ------------------------------------------------------------------
    # Device & reproducibility
    # ------------------------------------------------------------------
    device = torch.device("cuda" if (args.gpu_mode and torch.cuda.is_available()) else "cpu")
    print(f"Device: {device}")

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(args.seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    # ------------------------------------------------------------------
    # Channels / physics front-end (backbone part of --model is ignored).
    # ------------------------------------------------------------------
    _, in_channels, physics_mode = parse_model_variant(args.model)

    # ------------------------------------------------------------------
    # Model
    # ------------------------------------------------------------------
    model = PUIEUNet(in_channels=in_channels, latent_dim=args.latent_dim).to(device)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model     : PUIE-UNet  (in_channels={in_channels}, physics={physics_mode}, "
          f"latent_dim={args.latent_dim}, params={n_params/1e6:.2f}M)")

    # ------------------------------------------------------------------
    # Datasets & DataLoaders  (physics channels appended in the collate)
    # ------------------------------------------------------------------
    collate_fn = lambda b: _collate_train(b, physics_mode)

    if args.dataset == "euvp":
        train_ds = get_euvp_training_set(
            args.data_train_euvp, img_size=args.cropSize,
            subset=args.euvp_subset, in_memory=args.in_memory)
    elif args.dataset == "uieb":
        train_ds = get_uieb_training_set(
            args.data_train_uieb, img_size=args.cropSize, in_memory=args.in_memory)
    elif args.dataset == "ufo120":
        train_ds = get_ufo120_training_set(
            args.data_train_euvp, img_size=args.cropSize, in_memory=args.in_memory)
    elif args.dataset == "euvp+uieb":
        euvp_ds = get_euvp_training_set(
            args.data_train_euvp, img_size=args.cropSize,
            subset=args.euvp_subset, in_memory=args.in_memory)
        uieb_ds = get_uieb_training_set(
            args.data_train_uieb, img_size=args.cropSize, in_memory=args.in_memory)
        train_ds = data.ConcatDataset([euvp_ds, uieb_ds])
    else:
        raise ValueError(f"Unknown --dataset: {args.dataset}")

    # 10% paired hold-out for validation (gives GT for loss + metrics).
    n_val   = max(1, int(len(train_ds) * 0.10))
    n_train = len(train_ds) - n_val
    train_ds, val_ds = data.random_split(
        train_ds, [n_train, n_val],
        generator=torch.Generator().manual_seed(args.seed))
    print(f"Dataset   : {args.dataset}  (train={n_train}, val={n_val})")

    train_loader = data.DataLoader(
        train_ds, batch_size=args.batchSize, shuffle=args.shuffle,
        num_workers=args.threads, pin_memory=device.type == "cuda",
        drop_last=True, collate_fn=collate_fn)
    val_loader = data.DataLoader(
        val_ds, batch_size=args.batchSize, shuffle=False,
        num_workers=args.threads, pin_memory=device.type == "cuda",
        drop_last=False, collate_fn=collate_fn)

    print(f"Train set : {len(train_ds)} samples ({len(train_loader)} batches of {args.batchSize})")
    print(f"Val set   : {len(val_ds)} samples")

    # ------------------------------------------------------------------
    # Loss (reconstruction term), optimizer, scheduler
    # ------------------------------------------------------------------
    criterion = CompositeLoss(
        lambda_l1=args.L1_weight, lambda_perc=args.perceptual_weight,
        lambda_ssim=args.SSIM_weight, device=device)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = build_scheduler(optimizer, args)
    print(f"Optimizer : Adam (lr={args.lr})  |  KL beta={args.kl_weight} "
          f"(anneal {args.kl_anneal_epochs} epochs)")

    # ------------------------------------------------------------------
    # Checkpoint directory
    # ------------------------------------------------------------------
    RUN_TS = datetime.now().strftime('%Y%m%d_%H%M%S')
    RUN_ID = (f"{args.run_name.strip()}_{RUN_TS}" if args.run_name.strip()
              else f"puie_unet_{args.dataset}_{RUN_TS}")
    CKPT_DIR  = os.path.join(args.checkpoint_dir, RUN_ID)
    BEST_PATH = os.path.join(CKPT_DIR, "best_model.pth")
    LAST_PATH = os.path.join(CKPT_DIR, "last_model.pth")
    os.makedirs(CKPT_DIR, exist_ok=True)
    print(f"Run ID    : {RUN_ID}")
    print(f"Ckpt dir  : {CKPT_DIR}")

    start_epoch = 1
    if args.resume and os.path.isfile(args.resume):
        start_epoch, _ = load_ckpt(args.resume, model, optimizer, device=str(device))
        start_epoch += 1
        print(f"Resumed   : {args.resume}  → continuing from epoch {start_epoch}")

    # ------------------------------------------------------------------
    # Training loop
    # ------------------------------------------------------------------
    history = {"train_loss": [], "val_loss": [], "val_psnr": [], "val_ssim": [], "lr": []}
    best_psnr, best_epoch = 0.0, 0
    es = EarlyStopping(patience=args.early_stop_patience, min_delta=1e-4, mode="max")
    LOG_EVERY = max(1, args.nEpochs // 20)

    print(f"\n{'='*72}")
    print(f"  Training PUIE-UNet   epochs={args.nEpochs}   batch={args.batchSize}")
    print(f"{'='*72}")
    print(f"{'Epoch':>6}  {'TrainL':>8}  {'Recon':>8}  {'KL':>8}  {'ValL':>8}  "
          f"{'PSNR':>7}  {'SSIM':>7}  {'LR':>9}  {'Time':>6}")
    print("-" * 72)

    train_start = time.time()

    for epoch in range(start_epoch, args.nEpochs + 1):
        t0 = time.time()

        # KL annealing: ramp beta 0 -> kl_weight over kl_anneal_epochs.
        if args.kl_anneal_epochs > 0:
            kl_w = args.kl_weight * min(1.0, epoch / args.kl_anneal_epochs)
        else:
            kl_w = args.kl_weight

        tr_loss, tr_parts = train_epoch(
            model, train_loader, optimizer, criterion, device, kl_w, args.grad_clip)
        vl_loss = val_loss_epoch(model, val_loader, criterion, device, kl_w)

        # Full-reference metrics use the prior path (model(inp) → pred).
        if epoch == 1 or epoch % LOG_EVERY == 0:
            val_metrics, _ = evaluate_loader(model, val_loader, device, max_samples=100)
            val_psnr, val_ssim = val_metrics["psnr"], val_metrics["ssim"]
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

        if val_psnr > best_psnr + 1e-4:
            best_psnr, best_epoch = val_psnr, epoch
            save_ckpt(model, optimizer, epoch,
                      {"psnr": val_psnr, "ssim": val_ssim, "val_loss": vl_loss}, BEST_PATH)
            flag = "  ← BEST ✓"
        else:
            flag = ""

        if epoch % args.snapshots == 0:
            save_ckpt(model, optimizer, epoch, {"psnr": val_psnr, "ssim": val_ssim},
                      os.path.join(CKPT_DIR, f"epoch_{epoch:04d}.pth"))

        if epoch == 1 or epoch % LOG_EVERY == 0 or flag:
            print(f"{epoch:>6}  {tr_loss:>8.4f}  {tr_parts['recon']:>8.4f}  "
                  f"{tr_parts['kl']:>8.4f}  {vl_loss:>8.4f}  {val_psnr:>7.3f}  "
                  f"{val_ssim:>7.4f}  {cur_lr:>9.2e}  {elapsed:>5.1f}s{flag}")

        if es(val_psnr):
            print(f"\nEarly stopping at epoch {epoch} "
                  f"(no PSNR improvement for {args.early_stop_patience} evals)")
            break

    save_ckpt(model, optimizer, epoch, {"psnr": val_psnr, "ssim": val_ssim}, LAST_PATH)

    total_min = (time.time() - train_start) / 60
    print(f"\n{'='*72}")
    print(f"  Done.  Best PSNR: {best_psnr:.4f} dB at epoch {best_epoch}")
    print(f"  Total training time: {total_min:.1f} min")
    print(f"{'='*72}")

    with open(os.path.join(CKPT_DIR, "training_history.json"), "w") as f:
        json.dump(history, f, indent=2)
    print("History saved →", os.path.join(CKPT_DIR, "training_history.json"))


if __name__ == "__main__":
    main()
