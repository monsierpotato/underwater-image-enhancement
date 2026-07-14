# ============================================================
# measure_underwater.py  –  Full metric suite
#
# Reference-based:
#   compute_psnr()       – Peak Signal-to-Noise Ratio
#   compute_ssim()       – Structural Similarity Index
#   compute_ciede2000()  – Perceptual colour difference (lower = better)
#
# No-reference / underwater-specific:
#   compute_uciqe()      – Underwater Colour Image Quality Evaluation
#   compute_uiqm()       – Underwater Image Quality Measure
#
# Batch runner:
#   evaluate_loader()    – iterates a DataLoader and returns mean metrics
# ============================================================

import time

import cv2
import numpy as np
import torch
from scipy.ndimage import sobel
from skimage.color import deltaE_ciede2000, rgb2lab
from skimage.metrics import peak_signal_noise_ratio as psnr_sk
from skimage.metrics import structural_similarity as ssim_sk

# ============================================================
# Reference-based metrics
# ============================================================


def compute_psnr(pred, target):
    """
    Peak Signal-to-Noise Ratio.
    pred, target : numpy (H, W, 3) float32 in [0, 1]
    Returns      : float (dB) – higher is better
    """
    return float(psnr_sk(target, pred, data_range=1.0))


def compute_ssim(pred, target):
    """
    Structural Similarity Index.
    pred, target : numpy (H, W, 3) float32 in [0, 1]
    Returns      : float in [0, 1] – higher is better
    """
    return float(ssim_sk(target, pred, data_range=1.0, channel_axis=2))


def compute_ciede2000(pred, target):
    """
    Mean CIEDE2000 perceptual colour difference.
    pred, target : numpy (H, W, 3) float32 in [0, 1]
    Returns      : float – lower is better
    """
    pred_lab = rgb2lab(pred.clip(0, 1))
    target_lab = rgb2lab(target.clip(0, 1))
    return float(np.mean(deltaE_ciede2000(target_lab, pred_lab)))


# ============================================================
# No-reference underwater metrics
# ============================================================


def compute_uciqe(image):
    """
    UCIQE – Yang & Sowmya (2015).
    image: (H,W,3) float [0,1] RGB
    Fix: chuyển OpenCV LAB uint8 → chuẩn CIELab trước khi tính.
    """
    img_u8 = (np.nan_to_num(image, nan=0.0).clip(0, 1) * 255).astype(np.uint8)

    # ── OpenCV LAB uint8 → standard CIELab ──────────────────────
    # OpenCV:  L ∈ [0,255],  A ∈ [0,255],  B ∈ [0,255]
    # Standard: L ∈ [0,100], a ∈ [-128,127], b ∈ [-128,127]
    lab = cv2.cvtColor(img_u8, cv2.COLOR_RGB2LAB).astype(np.float64)
    L_std = lab[:, :, 0] * (100.0 / 255.0)  # L: 0 → 100
    a_std = lab[:, :, 1] - 128.0  # a: -128 → 127
    b_std = lab[:, :, 2] - 128.0  # b: -128 → 127

    # σc – std của chroma
    chroma = np.sqrt(a_std**2 + b_std**2)
    sigma_c = np.std(chroma)

    # con_l – contrast của L (top 1% − bottom 1%), trong [0,100]
    sorted_L = np.sort(L_std.flatten())
    n = len(sorted_L)
    con_l = np.mean(sorted_L[int(0.99 * n) :]) - np.mean(sorted_L[: max(1, int(0.01 * n))])

    # μs – mean saturation HSV ∈ [0,1]
    hsv = cv2.cvtColor(img_u8, cv2.COLOR_RGB2HSV).astype(np.float64)
    mu_s = np.mean(hsv[:, :, 1]) / 255.0

    return float(0.4680 * sigma_c + 0.2745 * con_l + 0.2576 * mu_s)


def compute_uiqm(image):
    """
    UIQM – Underwater Image Quality Measure
    (Panetta et al., 2016).

    image  : numpy (H, W, 3) float32 in [0, 1] RGB
    Returns: float – higher is better

    Sub-components
    --------------
    UICM   : colorfulness  (RG / YB opponent channels)
    UISM   : sharpness     (Sobel edge magnitude)
    UIConM : contrast      (σ_gray / μ_gray)
    """
    img = image.clip(0, 1).astype(np.float64)
    R, G, B = img[:, :, 0], img[:, :, 1], img[:, :, 2]

    # UICM – colorfulness
    RG = R - G
    YB = 0.5 * (R + G) - B
    uicm = -0.0268 * np.sqrt(np.mean(RG) ** 2 + np.mean(YB) ** 2) + 0.1586 * np.sqrt(
        np.std(RG) ** 2 + np.std(YB) ** 2
    )

    # UISM – sharpness (Sobel on luminance)
    gray = 0.2989 * R + 0.5870 * G + 0.1140 * B
    sx = sobel(gray, axis=0)
    sy = sobel(gray, axis=1)
    uism = np.mean(np.sqrt(sx**2 + sy**2))

    # UIConM – contrast
    mu_g = np.mean(gray)
    sigma_g = np.std(gray)
    uiconm = sigma_g / (mu_g + 1e-8)

    c1, c2, c3 = 0.0282, 0.2953, 3.5753
    return float(c1 * uicm + c2 * uism + c3 * uiconm)


# ============================================================
# Batch evaluation runner
# ============================================================


@torch.no_grad()
def evaluate_loader(model, loader, device, max_samples=None, desc="Evaluating"):
    """
    Run the full metric suite over a DataLoader.

    Parameters
    ----------
    model       : nn.Module (eval mode is set internally)
    loader      : DataLoader yielding (inp, gt) pairs
    device      : torch.device
    max_samples : int | None  – stop after this many images (None = all)
    desc        : str  – label for progress messages

    Returns
    -------
    means  : dict  {metric_name: float}  – mean over all evaluated samples
    count  : int   – number of images evaluated
    """
    model.eval()
    results = {k: [] for k in ("psnr", "ssim", "ciede2000", "uciqe", "uiqm")}
    count = 0
    total_time_ms = 0.0

    for inp, gt in loader:
        inp = inp.to(device)

        # Measure pure forward pass time
        if device.type == "cuda":
            start_evt = torch.cuda.Event(enable_timing=True)
            end_evt = torch.cuda.Event(enable_timing=True)
            start_evt.record()
            pred = model(inp)
            end_evt.record()
            torch.cuda.synchronize()
            total_time_ms += start_evt.elapsed_time(end_evt)
        else:
            t0 = time.perf_counter()
            pred = model(inp)
            total_time_ms += (time.perf_counter() - t0) * 1000.0

        pred_np = pred.cpu().permute(0, 2, 3, 1).numpy().clip(0, 1)
        gt_np = gt.permute(0, 2, 3, 1).numpy().clip(0, 1)

        for i in range(pred_np.shape[0]):
            p, g = pred_np[i], gt_np[i]
            results["psnr"].append(compute_psnr(p, g))
            results["ssim"].append(compute_ssim(p, g))
            results["ciede2000"].append(compute_ciede2000(p, g))
            results["uciqe"].append(compute_uciqe(p))
            results["uiqm"].append(compute_uiqm(p))
            count += 1
            if max_samples and count >= max_samples:
                break
        if max_samples and count >= max_samples:
            break

    means = {k: float(np.mean(v)) for k, v in results.items()}
    if count > 0:
        means["inference_ms_per_img"] = total_time_ms / count
    return means, count


if __name__ == "__main__":
    # Quick smoke-test on a random image pair
    dummy_pred = np.random.rand(256, 256, 3).astype(np.float32)
    dummy_target = np.random.rand(256, 256, 3).astype(np.float32)

    print(f"PSNR      : {compute_psnr(dummy_pred, dummy_target):.4f} dB")
    print(f"SSIM      : {compute_ssim(dummy_pred, dummy_target):.4f}")
    print(f"CIEDE2000 : {compute_ciede2000(dummy_pred, dummy_target):.4f}")
    print(f"UCIQE     : {compute_uciqe(dummy_pred):.4f}")
    print(f"UIQM      : {compute_uiqm(dummy_pred):.4f}")
    print("All metrics OK.")
