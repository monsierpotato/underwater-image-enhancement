from uwir.config import TrainConfig, option
from uwir.models import ModelSpec, parse_model_variant


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
