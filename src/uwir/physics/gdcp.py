"""
physics_gdcp.py
---------------
Ucolor/GDCP physics-guided channel extraction.

This module ports the GDCP pipeline bundled with Ucolor:

  getGrad.m + GetDepth.m -> DepthMap
  atmLight.m             -> background / ambient light B
  calcTrans.m            -> transmission map t(x)

It preserves the same public output contract used by this project:
``compute_physics_maps(image_np) -> (t_map, b_map)``.
"""

import cv2
import numpy as np
from scipy.ndimage import maximum_filter, median_filter, minimum_filter
from skimage.morphology import reconstruction


def _as_float_rgb(image_np: np.ndarray) -> np.ndarray:
    """Validate and normalize an RGB image to float32 in [0, 1]."""
    image = np.asarray(image_np, dtype=np.float32)
    if image.ndim != 3 or image.shape[2] != 3:
        raise ValueError(f"Expected image shape (H, W, 3), got {image.shape}")
    if image.size and image.max() > 1.0:
        image = image / 255.0
    return np.clip(image, 0.0, 1.0).astype(np.float32)


def _stretch01(x: np.ndarray) -> np.ndarray:
    """MATLAB Stretch helper with a constant-image guard."""
    x = np.asarray(x, dtype=np.float32)
    xmin = float(np.min(x))
    xmax = float(np.max(x))
    denom = xmax - xmin
    if denom <= 1e-12:
        return np.zeros_like(x, dtype=np.float32)
    return ((x - xmin) / denom).astype(np.float32)


def _fill_holes_gray(image: np.ndarray) -> np.ndarray:
    """Approximate MATLAB imfill(gray, 8, 'holes') for grayscale images."""
    image = np.asarray(image, dtype=np.float32)
    seed = image.copy()
    seed[1:-1, 1:-1] = float(np.max(image))
    return reconstruction(seed, image, method="erosion").astype(np.float32)


def _rgb_to_lab_matlab(image: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Port of GDCP_Release/getRGB2Lab.m."""
    image = _as_float_rgb(image)
    r = image[:, :, 0].astype(np.float64)
    g = image[:, :, 1].astype(np.float64)
    b = image[:, :, 2].astype(np.float64)

    threshold = 0.008856
    h, w = r.shape
    rgb = np.vstack([r.reshape(1, -1), g.reshape(1, -1), b.reshape(1, -1)])
    mat = np.array(
        [
            [0.412453, 0.357580, 0.180423],
            [0.212671, 0.715160, 0.072169],
            [0.019334, 0.119193, 0.950227],
        ],
        dtype=np.float64,
    )
    xyz = mat @ rgb
    x = xyz[0, :] / 0.950456
    y = xyz[1, :]
    z = xyz[2, :] / 1.088754

    xt = x > threshold
    yt = y > threshold
    zt = z > threshold
    y3 = np.cbrt(y)

    fx = xt * np.cbrt(x) + (~xt) * (7.787 * x + 16.0 / 116.0)
    fy = yt * y3 + (~yt) * (7.787 * y + 16.0 / 116.0)
    fz = zt * np.cbrt(z) + (~zt) * (7.787 * z + 16.0 / 116.0)

    lab_l = (yt * (116.0 * y3 - 16.0) + (~yt) * (903.3 * y)).reshape(h, w)
    lab_a = (500.0 * (fx - fy)).reshape(h, w)
    lab_b = (200.0 * (fy - fz)).reshape(h, w)
    return lab_l, lab_a, lab_b


def _color_cast_scale(image: np.ndarray) -> np.ndarray:
    """Port of CC.m/getColorCast.m, kept for GDCP completeness."""
    image = _as_float_rgb(image)
    _, lab_a, lab_b = _rgb_to_lab_matlab(image)
    var_sq = float(np.sqrt(np.var(lab_a) + np.var(lab_b)))
    if var_sq <= 1e-12:
        return np.ones(3, dtype=np.float32)

    u = float(np.sqrt(np.mean(lab_a) ** 2 + np.mean(lab_b) ** 2))
    dl = (u - var_sq) / var_sq
    if dl <= 0:
        return np.ones(3, dtype=np.float32)

    avg_ic = np.mean(image.reshape(-1, 3), axis=0)
    numerator = max(float(np.max(avg_ic)), 0.1)
    denominator = np.maximum(avg_ic, 0.1)
    exponent = 1.0 / max(float(np.sqrt(dl)), 1.0)
    return np.power(numerator / denominator, exponent).astype(np.float32)


def estimate_depth_gdcp(image_np: np.ndarray, win: int = 15) -> tuple[np.ndarray, np.ndarray]:
    """Estimate the GDCP depth map from Ucolor's GetDepth.m."""
    image = _as_float_rgb(image_np)

    y_channel = cv2.cvtColor(image, cv2.COLOR_RGB2YCrCb)[:, :, 0]
    grad_x = cv2.Sobel(y_channel, cv2.CV_32F, 1, 0, ksize=3)
    grad_y = cv2.Sobel(y_channel, cv2.CV_32F, 0, 1, ksize=3)
    grad_mag = cv2.magnitude(grad_x, grad_y)

    dilated = maximum_filter(grad_mag, size=win, mode="reflect")
    grad_map = _stretch01(_fill_holes_gray(dilated))
    rough_depth = 1.0 - grad_map

    dep_vec = rough_depth.reshape(-1).astype(np.float64)
    im_vec = image.reshape(-1, 3).astype(np.float64)

    # Closed-form normal equations for X = [ones, d]  (2×2 system, O(n))
    # weights = (XᵀX)⁻¹ Xᵀ Y  — avoids SVD used by lstsq
    n = float(len(dep_vec))
    sd = float(dep_vec.sum())
    sd2 = float((dep_vec * dep_vec).sum())
    det = n * sd2 - sd * sd
    if abs(det) < 1e-12:
        # Degenerate (flat depth map): intercept = mean(Y), slope = 0
        weights = np.zeros((2, 3), dtype=np.float64)
        weights[0] = im_vec.mean(axis=0)
    else:
        s_y = im_vec.sum(axis=0)  # (3,)
        sd_y = dep_vec @ im_vec  # (3,)
        weights = np.empty((2, 3), dtype=np.float64)
        weights[0] = (sd2 * s_y - sd * sd_y) / det  # intercept
        weights[1] = (n * sd_y - sd * s_y) / det  # slope

    slopes = weights[1, :]
    ws = np.tanh(4.0 * np.abs(slopes))
    signs = (slopes <= 0).astype(np.float64)

    channel_mins = []
    for channel in range(3):
        distance = np.abs(signs[channel] - image[:, :, channel])
        local_min = minimum_filter(distance, size=win, mode="reflect")
        channel_mins.append(ws[channel] * local_min + (1.0 - ws[channel]))

    depth_map = np.min(np.stack(channel_mins, axis=2), axis=2)
    return np.clip(depth_map, 0.0, 1.0).astype(np.float32), grad_map.astype(np.float32)


def estimate_background_light_gdcp(
    image_np: np.ndarray,
    depth_map: np.ndarray | None = None,
    win: int = 15,
) -> np.ndarray:
    """Estimate background / ambient light B_rgb from the GDCP depth map."""
    image = _as_float_rgb(image_np)
    if depth_map is None:
        depth_map, _ = estimate_depth_gdcp(image, win=win)

    depth = np.asarray(depth_map, dtype=np.float32)
    if depth.shape != image.shape[:2]:
        raise ValueError(f"Depth map shape {depth.shape} does not match image {image.shape[:2]}")

    imsize = depth.size
    numpx = max(1, int(np.floor(imsize / 1000.0)))
    indices = np.argsort(depth.reshape(-1))[-numpx:]
    b_rgb = np.mean(image.reshape(-1, 3)[indices, :], axis=0)
    return np.clip(b_rgb, 0.0, 1.0).astype(np.float32)


def estimate_transmission_gdcp(
    image_np: np.ndarray,
    B: np.ndarray,
    win: int = 15,
    t0: float = 0.2,
) -> np.ndarray:
    """Estimate transmission map using Ucolor's calcTrans.m + IR_GDCP.m."""
    image = _as_float_rgb(image_np)
    b_rgb = np.asarray(B, dtype=np.float32).reshape(3)

    dl_map = np.abs(b_rgb.reshape(1, 1, 3) - image)
    bm = np.maximum(b_rgb, 1.0 - b_rgb)
    bm = np.maximum(bm, 1e-6).reshape(1, 1, 3)
    dl_map_nor = dl_map / bm

    max_dl = np.max(dl_map_nor, axis=2)
    trans = median_filter(max_dl, size=win, mode="reflect")
    trans = np.clip(trans, 0.0, 1.0).astype(np.float32)

    max_t = float(np.max(trans))
    min_t = float(np.min(trans))
    denom = max_t - min_t
    if denom <= 1e-12:
        trans_pro = np.full_like(trans, float(t0), dtype=np.float32)
    else:
        trans_pro = ((trans - min_t) / denom) * (max_t - float(t0)) + float(t0)

    return np.clip(trans_pro, 0.0, 1.0).astype(np.float32)


def estimate_background_light(image_np: np.ndarray, percentile: float = 0.1) -> np.ndarray:
    """Compatibility alias for GDCP background light."""
    del percentile
    return estimate_background_light_gdcp(image_np)


def estimate_transmission_udcp(
    image_np: np.ndarray,
    B: np.ndarray,
    omega: float = 0.95,
    patch_size: int = 15,
) -> np.ndarray:
    """Compatibility alias; returns the GDCP transmission map in this module."""
    del omega
    return estimate_transmission_gdcp(image_np, B, win=patch_size)


def compute_physics_maps(image_np: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return ``(t_map, b_map)`` using Ucolor/GDCP extraction."""
    image = _as_float_rgb(image_np)  # validate + normalise once
    depth_map, _ = estimate_depth_gdcp(image)
    b_rgb = estimate_background_light_gdcp(image, depth_map=depth_map)
    t_map = estimate_transmission_gdcp(image, b_rgb)
    b_map = np.full(t_map.shape, float(np.mean(b_rgb)), dtype=np.float32)
    return t_map.astype(np.float32), b_map
