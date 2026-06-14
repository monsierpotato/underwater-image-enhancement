# ============================================================
# test_run.py  –  Smoke-test for the full training pipeline
#
# Runs a few steps of training + validation on synthetic random
# data (no real dataset required).  Checks:
#   ✓  Model forward pass (3-ch and 5-ch)
#   ✓  CompositeLoss backward pass
#   ✓  train_epoch / val_loss_epoch helpers
#   ✓  evaluate_loader metrics
#   ✓  Checkpoint save / reload
#   ✓  Scheduler step
# ============================================================

import os
import tempfile
import torch
import torch.utils.data as data

# ── project imports ────────────────────────────────────────
from net import UNet5ch
from loss import CompositeLoss
from measure_underwater import evaluate_loader
from train import train_epoch, val_loss_epoch, save_ckpt, load_ckpt, EarlyStopping
from data.scheduler import CosineAnnealingRestartLR


# ── configuration ──────────────────────────────────────────
DEVICE     = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BATCH_SIZE = 2
IMG_SIZE   = 64          # small patches → fast test
N_BATCHES  = 4           # batches per fake epoch
N_EPOCHS   = 3


# ── helpers ────────────────────────────────────────────────

def fake_loader(in_ch: int, n_batches: int = N_BATCHES):
    """Return a DataLoader that yields (inp, gt) random tensor pairs."""
    inp = torch.rand(n_batches * BATCH_SIZE, in_ch, IMG_SIZE, IMG_SIZE)
    gt  = torch.rand(n_batches * BATCH_SIZE, 3,    IMG_SIZE, IMG_SIZE)
    ds  = data.TensorDataset(inp, gt)
    return data.DataLoader(ds, batch_size=BATCH_SIZE, shuffle=False)


def ok(msg: str):
    print(f"  ✓  {msg}")


# ── tests ──────────────────────────────────────────────────

def test_forward():
    print("\n[1] Forward pass")
    for in_ch in (3, 5):
        model = UNet5ch(in_channels=in_ch).to(DEVICE)
        x     = torch.rand(1, in_ch, IMG_SIZE, IMG_SIZE).to(DEVICE)
        with torch.no_grad():
            y = model(x)
        assert y.shape == (1, 3, IMG_SIZE, IMG_SIZE), f"Bad output shape: {y.shape}"
        ok(f"UNet5ch(in_channels={in_ch})  →  {tuple(y.shape)}")


def test_loss():
    print("\n[2] Loss backward")
    model     = UNet5ch(in_channels=5).to(DEVICE)
    criterion = CompositeLoss(device=DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

    inp  = torch.rand(BATCH_SIZE, 5, IMG_SIZE, IMG_SIZE).to(DEVICE)
    gt   = torch.rand(BATCH_SIZE, 3, IMG_SIZE, IMG_SIZE).to(DEVICE)
    pred = model(inp)
    loss, parts = criterion(pred, gt)
    loss.backward()
    optimizer.step()

    ok(f"total={loss.item():.4f}  l1={parts['l1']:.4f}  "
       f"perc={parts['perceptual']:.4f}  ssim={parts['ssim_loss']:.4f}")


def test_train_val_loop():
    print("\n[3] train_epoch / val_loss_epoch")
    model     = UNet5ch(in_channels=5).to(DEVICE)
    criterion = CompositeLoss(device=DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    scheduler = CosineAnnealingRestartLR(optimizer, periods=[N_EPOCHS], restart_weights=[1.0])
    loader    = fake_loader(in_ch=5)

    for epoch in range(1, N_EPOCHS + 1):
        tr_loss, comps = train_epoch(model, loader, optimizer, criterion, DEVICE)
        vl_loss        = val_loss_epoch(model, loader, criterion, DEVICE)
        scheduler.step()
        lr = optimizer.param_groups[0]["lr"]
        ok(f"epoch {epoch}  train={tr_loss:.4f}  val={vl_loss:.4f}  lr={lr:.2e}")


def test_metrics():
    print("\n[4] evaluate_loader metrics")
    model  = UNet5ch(in_channels=5).to(DEVICE)
    loader = fake_loader(in_ch=5, n_batches=2)
    means, count = evaluate_loader(model, loader, DEVICE, max_samples=4)
    ok(f"n={count}  PSNR={means['psnr']:.3f}  SSIM={means['ssim']:.4f}  "
       f"CIEDE2000={means['ciede2000']:.3f}")


def test_checkpoint():
    print("\n[5] Checkpoint save / reload")
    model     = UNet5ch(in_channels=5).to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    metrics   = {"psnr": 28.5, "ssim": 0.91}

    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "ckpt", "best.pth")
        save_ckpt(model, optimizer, epoch=7, metrics=metrics, path=path)
        assert os.path.isfile(path), "Checkpoint file not written"
        ok(f"saved  →  {path}")

        model2 = UNet5ch(in_channels=5).to(DEVICE)
        epoch_loaded, metrics_loaded = load_ckpt(path, model2, device=str(DEVICE))
        assert epoch_loaded == 7
        assert metrics_loaded["psnr"] == 28.5
        ok(f"loaded  epoch={epoch_loaded}  metrics={metrics_loaded}")


def test_early_stopping():
    print("\n[6] EarlyStopping")
    es = EarlyStopping(patience=3, min_delta=0.01, mode="max")
    scores = [20.0, 20.5, 20.4, 20.3, 20.2]
    stopped_at = None
    for i, s in enumerate(scores):
        if es(s):
            stopped_at = i
            break
    assert stopped_at == 4, f"Expected stop at index 4, got {stopped_at}"
    ok(f"stopped at step {stopped_at} (patience=3)")


# ── main ───────────────────────────────────────────────────

if __name__ == "__main__":
    print("=" * 55)
    print(f"  Smoke-test  –  device: {DEVICE}")
    print("=" * 55)

    test_forward()
    test_loss()
    test_train_val_loop()
    test_metrics()
    test_checkpoint()
    test_early_stopping()

    print("\n" + "=" * 55)
    print("  All tests passed ✓")
    print("=" * 55)
