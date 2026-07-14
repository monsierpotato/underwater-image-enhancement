"""Compatibility exports for the legacy option parser."""

import _bootstrap  # noqa: F401
from uwir.config import TrainConfig, option

__all__ = ["TrainConfig", "option"]
