"""Compatibility entry point for the EUVP ablation workflow."""

import _bootstrap  # noqa: F401
from scripts.experiments.ablation_euvp import main

if __name__ == "__main__":
    main()
