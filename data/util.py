"""Compatibility exports for the legacy :mod:`data.util` module."""

import _bootstrap  # noqa: F401
from uwir.data.utils import is_image_file, load_img

__all__ = ["is_image_file", "load_img"]
