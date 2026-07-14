import importlib

import pytest

pytest.importorskip("torchvision")


def test_dataset_classes_are_preserved_through_legacy_module():
    canonical = importlib.import_module("uwir.data.datasets")
    legacy = importlib.import_module("data.UWIRdataset")

    for name in ("UIEBDataset", "EUVPDataset", "UFO120Dataset", "U45Dataset"):
        assert getattr(legacy, name) is getattr(canonical, name)
