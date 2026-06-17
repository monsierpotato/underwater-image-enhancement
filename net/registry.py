"""
registry.py
-----------
Central model factory.

Usage
-----
    from net.registry import build_model, parse_model_variant

    # Instantiate any supported model by name:
    model = build_model("resnet_5ch", pretrained_backbone=True)

    # Inspect what the name implies:
    backbone, in_channels, physics_mode = parse_model_variant("mobilenet_4ch_t")
    # → ("mobilenet", 4, "t")

Naming convention
-----------------
    <backbone>_<variant>

   backbone : unet | resnet | mobilenet | mambavision | mambaunet
  variant  : 3ch | 4ch_t | 4ch_b | 5ch

Physics modes
-------------
  3ch   → physics_mode = "none"  →  input: [R, G, B]
  4ch_t → physics_mode = "t"     →  input: [R, G, B, t(x)]
  4ch_b → physics_mode = "b"     →  input: [R, G, B, B_map]
  5ch   → physics_mode = "tb"    →  input: [R, G, B, t(x), B_map]
"""

import torch.nn as nn

from net.unet import UNet5ch
from net.resnet_unet import ResNetUNet
from net.mobilenet_unet import MobileNetUNet
from net.mambavision_unet import MambaVisionUNet
from net.mamba_unet import MambaUNet


# ---------------------------------------------------------------------------
# Supported names
# ---------------------------------------------------------------------------

_BACKBONES = ("unet", "resnet", "mobilenet", "mambavision", "mambaunet")

_VARIANTS = {
    "3ch":   (3, "none"),
    "4ch_t": (4, "t"),
    "4ch_b": (4, "b"),
    "5ch":   (5, "tb"),
}

# All valid model names (used to populate argparse choices)
ALL_MODEL_NAMES = [
    f"{backbone}_{variant}"
    for backbone in _BACKBONES
    for variant in _VARIANTS
]


# ---------------------------------------------------------------------------
# Parse helper
# ---------------------------------------------------------------------------

def parse_model_variant(name: str) -> tuple[str, int, str]:
    """
    Parse a model name string into its components.

    Args:
        name (str): e.g. ``"resnet_4ch_t"``

    Returns:
        backbone     (str): One of ``"unet"``, ``"resnet"``, ``"mobilenet"``.
        in_channels  (int): Number of input channels (3, 4, or 5).
        physics_mode (str): One of ``"none"``, ``"t"``, ``"b"``, ``"tb"``.

    Raises:
        ValueError: If the name cannot be parsed.
    """
    for backbone in _BACKBONES:
        prefix = backbone + "_"
        if name.startswith(prefix):
            variant = name[len(prefix):]
            if variant not in _VARIANTS:
                raise ValueError(
                    f"Unknown variant '{variant}' in model name '{name}'. "
                    f"Valid variants: {list(_VARIANTS)}"
                )
            in_channels, physics_mode = _VARIANTS[variant]
            return backbone, in_channels, physics_mode

    raise ValueError(
        f"Cannot parse model name '{name}'. "
        f"Expected format: <backbone>_<variant>  "
        f"(backbone ∈ {_BACKBONES}, variant ∈ {list(_VARIANTS)})."
    )


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

def build_model(name: str, pretrained_backbone: bool = True) -> nn.Module:
    """
    Instantiate a model by name.

    Args:
        name               (str):  Model name, e.g. ``"resnet_5ch"``.
        pretrained_backbone(bool): Whether to load pretrained ImageNet weights
                                   for ResNet / MobileNet encoders.
                                   Ignored for ``unet`` (no pretrained variant).

    Returns:
        nn.Module: Uninitialised model (not moved to a device).

    Example::

        model = build_model("mobilenet_4ch_t").to("cuda")
    """
    backbone, in_channels, _ = parse_model_variant(name)

    if backbone == "unet":
        return UNet5ch(in_channels=in_channels)

    if backbone == "resnet":
        return ResNetUNet(in_channels=in_channels, pretrained=pretrained_backbone)

    if backbone == "mobilenet":
        return MobileNetUNet(in_channels=in_channels, pretrained=pretrained_backbone)

    if backbone == "mambavision":
        return MambaVisionUNet(in_channels=in_channels, pretrained=pretrained_backbone)

    if backbone == "mambaunet":
        return MambaUNet(in_channels=in_channels)

    # Should never reach here due to parse_model_variant guard
    raise ValueError(f"Unknown backbone: {backbone}")
