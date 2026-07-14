"""Compatibility entry point for probabilistic PUIE-UNet training."""

import _bootstrap  # noqa: F401
from uwir.cli.puie_train import PUIEUNet, main

__all__ = ["PUIEUNet", "main"]

if __name__ == "__main__":
    main()
