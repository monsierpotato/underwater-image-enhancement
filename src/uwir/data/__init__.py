"""Datasets and data-loader factories with lazy dataset imports."""

__all__ = ["EUVPDataset", "U45Dataset", "UFO120Dataset", "UIEBDataset"]


def __getattr__(name):
    if name in __all__:
        from . import datasets

        return getattr(datasets, name)
    raise AttributeError(name)
