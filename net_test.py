from thop import profile
import torch
import time
from net import MambaVisionUNet, MambaUNet


def test_model(model, in_channels=3, label=""):
    input = torch.rand(1, in_channels, 256, 256).to('cuda')
    model = model.to('cuda').eval()
    torch.cuda.synchronize()
    # warm-up
    with torch.no_grad():
        _ = model(input)
    torch.cuda.synchronize()
    t0 = time.time()
    with torch.no_grad():
        _ = model(input)
    torch.cuda.synchronize()
    elapsed = time.time() - t0
    n_param = sum(p.nelement() for p in model.parameters())
    macs, _ = profile(model, inputs=(input,), verbose=False)
    print(f"\n{'='*60}")
    print(f"  {label}")
    print(f"{'='*60}")
    print(f"  Params  : {n_param / 2**20:.3f} M")
    print(f"  FLOPs   : {macs / 2**30:.3f} G")
    print(f"  Time    : {elapsed:.4f} s  (single forward, no grad)")


# MambaVision-T encoder + UNet decoder  (existing baseline)
test_model(MambaVisionUNet(in_channels=5), in_channels=5,
           label="MambaVisionUNet-T  (5ch, pretrained encoder)")

# Native Mamba U-Net  (new)
test_model(MambaUNet(in_channels=3), in_channels=3,
           label="MambaUNet-T  (3ch, from scratch)")
