"""Compatibility exports for the historical :mod:`net` package."""

import _bootstrap  # noqa: F401
from uwir.models import (
    ALL_MODEL_NAMES,
    DenseNetUNet,
    MambaUNet,
    MambaVisionUNet,
    MobileNetUNet,
    ResNetUNet,
    UNet5ch,
    build_model,
    parse_model_variant,
)
from uwir.physics import (
    compute_gupdm_feature_maps,
    compute_physics_maps,
    compute_physics_maps_gdcp,
    compute_physics_maps_gupdm,
    estimate_background_light,
    estimate_transmission_udcp,
)

__all__ = [
    "UNet5ch",
    "DenseNetUNet",
    "ResNetUNet",
    "MobileNetUNet",
    "MambaVisionUNet",
    "MambaUNet",
    "compute_physics_maps",
    "compute_physics_maps_gdcp",
    "compute_physics_maps_gupdm",
    "compute_gupdm_feature_maps",
    "estimate_background_light",
    "estimate_transmission_udcp",
    "build_model",
    "parse_model_variant",
    "ALL_MODEL_NAMES",
]
