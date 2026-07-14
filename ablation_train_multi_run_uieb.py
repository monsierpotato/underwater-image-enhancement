"""Compatibility entry point for the UIEB ablation workflow."""

import _bootstrap  # noqa: F401
from scripts.experiments.ablation_uieb import main

if __name__ == "__main__":
    main()
