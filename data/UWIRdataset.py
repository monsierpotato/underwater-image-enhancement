"""Compatibility exports for the historical dataset module name."""

import _bootstrap  # noqa: F401
from uwir.data.datasets import EUVPDataset, U45Dataset, UFO120Dataset, UIEBDataset

__all__ = ["EUVPDataset", "U45Dataset", "UFO120Dataset", "UIEBDataset"]
