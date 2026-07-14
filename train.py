"""Compatibility entry point for :mod:`uwir.cli.train`."""

import _bootstrap  # noqa: F401
from uwir.cli.train import (
    EarlyStopping,
    build_scheduler,
    load_ckpt,
    main,
    save_ckpt,
    train_epoch,
    val_loss_epoch,
)

__all__ = [
    "EarlyStopping",
    "build_scheduler",
    "load_ckpt",
    "main",
    "save_ckpt",
    "train_epoch",
    "val_loss_epoch",
]

if __name__ == "__main__":
    main()
