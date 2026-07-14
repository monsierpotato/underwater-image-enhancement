import pytest
import torch

from uwir.config import TrainConfig, option
from uwir.models import ModelSpec, build_model, parse_model_variant


def test_model_spec_is_named_and_tuple_compatible():
    spec = parse_model_variant("unet_5ch")

    assert spec == ModelSpec("unet", 5, "tb")
    backbone, channels, physics_mode = spec
    assert (backbone, channels, physics_mode) == ("unet", 5, "tb")


def test_canonical_and_legacy_flags_are_equivalent():
    parser = option()
    canonical = parser.parse_args(["--batch-size", "2", "--crop-size", "64", "--epochs", "3"])
    legacy = parser.parse_args(["--batchSize", "2", "--cropSize", "64", "--nEpochs", "3"])

    assert canonical.batch_size == legacy.batch_size == 2
    assert canonical.batchSize == legacy.batchSize == 2
    assert canonical.crop_size == legacy.crop_size == 64
    assert canonical.cropSize == legacy.cropSize == 64
    assert canonical.epochs == legacy.epochs == 3
    assert canonical.nEpochs == legacy.nEpochs == 3
    assert canonical.log_dir == "./logs/"
    assert TrainConfig.from_namespace(canonical).batch_size == 2


@pytest.mark.parametrize(
    ("model_name", "backbone"),
    [
        ("asppunet_5ch", "asppunet"),
        ("mambabottleneck_5ch", "mambabottleneck"),
        ("mambaaspp_5ch", "mambaaspp"),
    ],
)
def test_context_model_variants_are_registered(model_name, backbone):
    assert parse_model_variant(model_name) == ModelSpec(backbone, 5, "tb")


@pytest.mark.parametrize("model_name", ["asppunet_5ch", "mambabottleneck_5ch", "mambaaspp_5ch"])
def test_context_models_preserve_image_shape(model_name):
    model = build_model(model_name, pretrained_backbone=False).eval()
    with torch.no_grad():
        output = model(torch.rand(1, 5, 32, 32))
    assert output.shape == (1, 3, 32, 32)
    assert torch.isfinite(output).all()
