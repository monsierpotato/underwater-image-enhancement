"""Model architectures and registry with lazy architecture imports."""

from .registry import ALL_MODEL_NAMES, ModelSpec, build_model, parse_model_variant

__all__ = [
    "ALL_MODEL_NAMES",
    "ASPPUNet",
    "ContextUNet",
    "DenseNetUNet",
    "MambaASPPUNet",
    "MambaBottleneckUNet",
    "MambaUNet",
    "MambaVisionUNet",
    "MobileNetUNet",
    "ModelSpec",
    "ResNetUNet",
    "UNet5ch",
    "build_model",
    "parse_model_variant",
]


_ARCHITECTURES = {
    "ASPPUNet": ("context_unet", "ASPPUNet"),
    "ContextUNet": ("context_unet", "ContextUNet"),
    "DenseNetUNet": ("densenet_unet", "DenseNetUNet"),
    "MambaASPPUNet": ("context_unet", "MambaASPPUNet"),
    "MambaBottleneckUNet": ("context_unet", "MambaBottleneckUNet"),
    "MambaUNet": ("mamba_unet", "MambaUNet"),
    "MambaVisionUNet": ("mambavision_unet", "MambaVisionUNet"),
    "MobileNetUNet": ("mobilenet_unet", "MobileNetUNet"),
    "ResNetUNet": ("resnet_unet", "ResNetUNet"),
    "UNet5ch": ("unet", "UNet5ch"),
}


def __getattr__(name):
    if name in _ARCHITECTURES:
        from importlib import import_module

        module_name, attribute = _ARCHITECTURES[name]
        return getattr(import_module(f"{__name__}.{module_name}"), attribute)
    raise AttributeError(name)
