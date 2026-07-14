"""Compatibility entry point for :mod:`uwir.cli.evaluate`."""

import _bootstrap  # noqa: F401
from uwir.cli.evaluate import TestDataset, collect_test_pairs, main

__all__ = ["TestDataset", "collect_test_pairs", "main"]

if __name__ == "__main__":
    main()
