"""
physics.py
----------
Physics-guided channel extraction for underwater image restoration.

Implements the Underwater Dark Channel Prior (UDCP) pipeline to estimate
two spatially-varying maps from a raw underwater RGB image:

  t(x)  – transmission map  ∈ [0.1, 1]   (how much light reaches sensor)
  B_map – background light  (scalar per image, broadcast to H×W)

These are concatenated to the RGB tensor to form the 5-channel input
[R, G, B, t(x), B_map] fed into UNet5ch.

References
----------
- He et al. (2011) "Single Image Haze Removal Using Dark Channel Prior"
- Drews et al. (2013) "Transmission Estimation in Underwater Single Images"
- He et al. (2013) "Guided Image Filtering"
"""

import cv2
import numpy as np
from scipy.ndimage import minimum_filter

# ---------------------------------------------------------------------------
# Background Light Estimation
# ---------------------------------------------------------------------------


def estimate_background_light(
    image_np: np.ndarray,
    percentile: float = 0.1,
) -> np.ndarray:
    """
    Estimate global background light B via the UDCP strategy.

    In underwater imagery the red channel attenuates fastest, so only the
    green and blue channels are used to locate the dark-channel candidates.

    Args:
        image_np  (ndarray): (H, W, 3) float32 RGB in [0, 1].
        percentile (float):  Top-% brightest dark-channel pixels used as
                             candidates. Default: 0.1.

    Returns:
        ndarray: shape (3,) float32 – estimated background light.
    """
    # Use only G & B since red attenuates fastest underwater
    dark_gb = np.min(image_np[:, :, 1:], axis=2)  # (H, W)

    n_pixels = dark_gb.size
    n_top = max(1, int(n_pixels * percentile / 100.0))
    flat_idx = np.argsort(dark_gb.flatten())[-n_top:]
    h_idx, w_idx = np.unravel_index(flat_idx, dark_gb.shape)

    # Among candidates, pick the brightest overall pixel
    candidate_intensity = np.mean(image_np[h_idx, w_idx, :], axis=1)
    best = np.argmax(candidate_intensity)
    return image_np[h_idx[best], w_idx[best], :].astype(np.float32)


# ---------------------------------------------------------------------------
# Guided Filter (edge-preserving smoothing)
# ---------------------------------------------------------------------------


def _guided_filter(
    guide: np.ndarray,
    src: np.ndarray,
    radius: int = 15,
    eps: float = 1e-3,
) -> np.ndarray:
    """
    Edge-preserving guided image filter (He et al., 2013).

    Args:
        guide  (ndarray): (H, W) float32 guidance image.
        src    (ndarray): (H, W) float32 input to be filtered.
        radius (int):     Filter radius. Default: 15.
        eps    (float):   Regularisation constant. Default: 1e-3.

    Returns:
        ndarray: (H, W) float32 filtered output.
    """
    guide = guide.astype(np.float64)
    src = src.astype(np.float64)
    ksize = (2 * radius + 1, 2 * radius + 1)

    def box(img: np.ndarray) -> np.ndarray:
        return cv2.boxFilter(img, -1, ksize)

    N = box(np.ones_like(guide))
    mI = box(guide) / N
    mp = box(src) / N
    mIp = box(guide * src) / N
    covIp = mIp - mI * mp

    mII = box(guide * guide) / N
    varI = mII - mI * mI

    a = covIp / (varI + eps)
    b = mp - a * mI

    ma = box(a) / N
    mb = box(b) / N
    return (ma * guide + mb).astype(np.float32)


# ---------------------------------------------------------------------------
# Transmission Map Estimation
# ---------------------------------------------------------------------------


def estimate_transmission_udcp(
    image_np: np.ndarray,
    B: np.ndarray,
    omega: float = 0.95,
    patch_size: int = 15,
) -> np.ndarray:
    """
    Estimate spatially-varying transmission map t(x) ∈ [0.1, 1].

    Args:
        image_np   (ndarray): (H, W, 3) float32 RGB in [0, 1].
        B          (ndarray): (3,) float32 background light estimate.
        omega      (float):   Controls how much haze to remove. Default: 0.95.
        patch_size (int):     Dark-channel minimum-filter patch size. Default: 15.

    Returns:
        ndarray: (H, W) float32 transmission map clipped to [0.1, 1].
    """
    B_safe = np.maximum(B, 1e-6)
    normalized = np.clip(image_np / B_safe, 0.0, 1.0)

    # UDCP: dark channel over green & blue only
    dark = np.min(normalized[:, :, 1:], axis=2)
    dark_ch = minimum_filter(dark, size=patch_size)

    t_rough = np.clip(1.0 - omega * dark_ch, 0.1, 1.0)

    # Edge-preserving refinement via guided filter
    guide = np.mean(image_np, axis=2).astype(np.float32)
    t_ref = _guided_filter(guide, t_rough, radius=15, eps=1e-3)
    return np.clip(t_ref, 0.1, 1.0)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def compute_physics_maps(
    image_np: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Compute the two physics-guided extra channels from a float32 RGB image.

    This is the single call used inside ``EUVPDataset.__getitem__`` when
    ``USE_PHYSICS=True``.  The result is converted to 1-channel tensors
    and concatenated onto the RGB tensor to form the 5-channel model input.

    Args:
        image_np (ndarray): (H, W, 3) float32 RGB image in [0, 1].

    Returns:
        t_map (ndarray): (H, W) float32 – per-pixel transmission map.
        b_map (ndarray): (H, W) float32 – spatially broadcast scalar
                         background light (mean of B across channels).

    Example::

        t_map, b_map = compute_physics_maps(img_np)
        t_t = torch.from_numpy(t_map).unsqueeze(0)  # (1, H, W)
        b_t = torch.from_numpy(b_map).unsqueeze(0)  # (1, H, W)
        inp_5ch = torch.cat([rgb_tensor, t_t, b_t], dim=0)  # (5, H, W)
    """
    B = estimate_background_light(image_np)
    t_map = estimate_transmission_udcp(image_np, B)
    b_map = np.full(t_map.shape, float(np.mean(B)), dtype=np.float32)
    return t_map, b_map
