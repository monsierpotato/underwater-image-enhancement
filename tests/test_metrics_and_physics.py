import numpy as np

from uwir.metrics import compute_ciede2000, compute_psnr, compute_ssim, compute_uciqe, compute_uiqm
from uwir.physics import compute_physics_maps


def test_reference_metrics_on_identical_images():
    image = np.full((32, 32, 3), 0.5, dtype=np.float32)

    assert np.isinf(compute_psnr(image, image))
    assert compute_ssim(image, image) == 1.0
    assert compute_ciede2000(image, image) == 0.0


def test_underwater_metrics_are_finite():
    gradient = np.linspace(0, 1, 32, dtype=np.float32)
    image = np.stack(np.meshgrid(gradient, gradient), axis=-1)
    image = np.concatenate([image, image[..., :1]], axis=-1)

    assert np.isfinite(compute_uciqe(image))
    assert np.isfinite(compute_uiqm(image))


def test_udcp_physics_map_contract():
    image = np.random.default_rng(42).random((32, 32, 3), dtype=np.float32)
    transmission, background = compute_physics_maps(image)

    assert transmission.shape == (32, 32)
    assert background.shape == (32, 32)
    assert np.all(np.isfinite(transmission))
    assert np.all(np.isfinite(background))
