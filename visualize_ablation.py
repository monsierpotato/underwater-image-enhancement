"""Compatibility entry point and exports for ablation visualization."""

import _bootstrap  # noqa: F401
from scripts.visualization.ablation import *  # noqa: F403
from scripts.visualization.ablation import main

if __name__ == "__main__":
    main()
