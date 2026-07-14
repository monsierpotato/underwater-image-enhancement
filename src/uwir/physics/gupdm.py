"""
physics_gupdm.py
----------------
GUPDM-style physical prior extraction.

The GUPDM repository uses an ``input_map`` as a transmission-guided physical
prior and states that UDCP can estimate this map. Its training code also uses
the underwater formation model

    I_degraded = t * J + (1 - t) * B

to vary transmission and global atmosphere light for ADS/TDS. This module
therefore exposes a GUPDM-oriented extractor while keeping the same project
contract: ``compute_physics_maps(image_np) -> (t_map, b_map)``.
"""

import cv2
import numpy as np
from scipy.ndimage import minimum_filter


def _as_float_rgb(image_np: np.ndarray) -> np.ndarray:
    image = np.asarray(image_np, dtype=np.float32)
    if image.ndim != 3 or image.shape[2] != 3:
        raise ValueError(f"Expected image shape (H, W, 3), got {image.shape}")
    if image.size and image.max() > 1.0:
        image = image / 255.0
    return np.clip(image, 0.0, 1.0).astype(np.float32)


def _guided_filter(
    guide: np.ndarray,
    src: np.ndarray,
    radius: int = 15,
    eps: float = 1e-3,
) -> np.ndarray:
    guide = guide.astype(np.float64)
    src = src.astype(np.float64)
    ksize = (2 * radius + 1, 2 * radius + 1)

    def box(img: np.ndarray) -> np.ndarray:
        return cv2.boxFilter(img, -1, ksize)

    n = box(np.ones_like(guide))
    mean_i = box(guide) / n
    mean_p = box(src) / n
    mean_ip = box(guide * src) / n
    cov_ip = mean_ip - mean_i * mean_p
    mean_ii = box(guide * guide) / n
    var_i = mean_ii - mean_i * mean_i

    a = cov_ip / (var_i + eps)
    b = mean_p - a * mean_i
    mean_a = box(a) / n
    mean_b = box(b) / n
    return (mean_a * guide + mean_b).astype(np.float32)


def estimate_background_light_gupdm(
    image_np: np.ndarray,
    percentile: float = 0.1,
) -> np.ndarray:
    """
    Estimate global atmosphere/background light for GUPDM's formation model.

    Candidates are selected from the underwater dark channel over green/blue,
    then the brightest RGB candidate is used as B.
    """
    image = _as_float_rgb(image_np)
    dark_gb = np.min(image[:, :, 1:], axis=2)
    n_top = max(1, int(dark_gb.size * percentile / 100.0))
    flat_idx = np.argsort(dark_gb.reshape(-1))[-n_top:]
    h_idx, w_idx = np.unravel_index(flat_idx, dark_gb.shape)
    best = np.argmax(np.mean(image[h_idx, w_idx, :], axis=1))
    return image[h_idx[best], w_idx[best], :].astype(np.float32)


def estimate_transmission_gupdm(
    image_np: np.ndarray,
    B: np.ndarray | None = None,
    omega: float = 0.95,
    patch_size: int = 15,
    min_transmission: float = 0.1,
) -> np.ndarray:
    """
    Estimate GUPDM's ``input_map`` transmission prior.

    This follows the GUPDM README guidance that UDCP can be used to estimate
    the transmission map consumed by the network.
    """
    image = _as_float_rgb(image_np)
    if B is None:
        B = estimate_background_light_gupdm(image)

    b_safe = np.maximum(np.asarray(B, dtype=np.float32), 1e-6).reshape(1, 1, 3)
    normalized = np.clip(image / b_safe, 0.0, 1.0)
    dark = np.min(normalized[:, :, 1:], axis=2)
    dark_ch = minimum_filter(dark, size=patch_size)
    t_rough = np.clip(1.0 - omega * dark_ch, min_transmission, 1.0)

    guide = np.mean(image, axis=2).astype(np.float32)
    t_ref = _guided_filter(guide, t_rough, radius=15, eps=1e-3)
    return np.clip(t_ref, min_transmission, 1.0).astype(np.float32)


def estimate_veiling_light_gupdm(
    t_map: np.ndarray,
    B: np.ndarray,
) -> np.ndarray:
    """Return scalar veiling-light prior ``(1 - t) * mean(B)``."""
    t = np.asarray(t_map, dtype=np.float32)
    b_scalar = float(np.mean(np.asarray(B, dtype=np.float32)))
    return np.clip((1.0 - t) * b_scalar, 0.0, 1.0).astype(np.float32)


def simulate_degradation_gupdm(
    clean_np: np.ndarray,
    t_map: np.ndarray,
    B: np.ndarray,
) -> np.ndarray:
    """
    Apply GUPDM's physical degradation form: ``I = t * J + (1 - t) * B``.
    """
    clean = _as_float_rgb(clean_np)
    t = np.asarray(t_map, dtype=np.float32)
    if t.ndim == 2:
        t = t[:, :, None]
    b_rgb = np.asarray(B, dtype=np.float32).reshape(1, 1, 3)
    return np.clip(t * clean + (1.0 - t) * b_rgb, 0.0, 1.0).astype(np.float32)


def compute_gupdm_feature_maps(image_np: np.ndarray) -> dict[str, np.ndarray]:
    """
    Return GUPDM-style physical features for experiments.

    Keys:
      - ``t_map``: transmission prior used as GUPDM input_map
      - ``b_rgb``: global atmosphere/background light
      - ``b_map``: mean(B_rgb) broadcast to HxW
      - ``veiling_map``: scalar ``(1 - t) * mean(B_rgb)`` prior
    """
    image = _as_float_rgb(image_np)
    b_rgb = estimate_background_light_gupdm(image)
    t_map = estimate_transmission_gupdm(image, b_rgb)
    b_map = np.full(t_map.shape, float(np.mean(b_rgb)), dtype=np.float32)
    veiling_map = estimate_veiling_light_gupdm(t_map, b_rgb)
    return {
        "t_map": t_map.astype(np.float32),
        "b_rgb": b_rgb.astype(np.float32),
        "b_map": b_map,
        "veiling_map": veiling_map,
    }


def estimate_background_light(image_np: np.ndarray, percentile: float = 0.1) -> np.ndarray:
    """Compatibility alias for GUPDM background light."""
    return estimate_background_light_gupdm(image_np, percentile=percentile)


def estimate_transmission_udcp(
    image_np: np.ndarray,
    B: np.ndarray,
    omega: float = 0.95,
    patch_size: int = 15,
) -> np.ndarray:
    """Compatibility alias for GUPDM transmission prior."""
    return estimate_transmission_gupdm(image_np, B, omega=omega, patch_size=patch_size)


def compute_physics_maps(image_np: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return GUPDM-style ``(t_map, b_map)`` while preserving project API."""
    features = compute_gupdm_feature_maps(image_np)
    return features["t_map"], features["b_map"]
