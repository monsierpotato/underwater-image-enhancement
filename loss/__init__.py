"""Compatibility exports for :mod:`uwir.losses`."""

import _bootstrap  # noqa: F401
from uwir.losses import CompositeLoss, SSIMLoss, VGGPerceptualLoss

__all__ = ["VGGPerceptualLoss", "SSIMLoss", "CompositeLoss"]
