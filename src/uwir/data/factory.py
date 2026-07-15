"""
Underwater Image Restoration — Central data factory
Exposes one get_*_training_set / get_*_eval_set function per dataset so
that train.py only needs a single import.
"""

from torchvision.transforms import (
    Compose,
    Resize,
    ToTensor,
)

from .datasets import (
    EUVPDataset,
    UFO120Dataset,
    UIEBDataset,
)
from .eval_sets import PaddedEvalDataset, SimpleEvalDataset

# ---------------------------------------------------------------------------
# Transform helpers
# ---------------------------------------------------------------------------


def _train_transform(img_size: int = 256):
    """Resize to img_size×img_size + to-tensor for paired training.

    Augmentation (hflip / vflip / rotation) is applied *separately* by the
    dataset classes so that both images in a pair receive the same transform.
    """
    return Compose(
        [
            Resize((img_size, img_size)),
            ToTensor(),
        ]
    )


def _eval_transform():
    """To-tensor only — no spatial changes for evaluation."""
    return Compose([ToTensor()])


# ---------------------------------------------------------------------------
# Training sets
# ---------------------------------------------------------------------------


def get_euvp_training_set(
    data_dir: str, img_size: int = 256, subset: str = "all", in_memory: bool = False, physics_mode: str = "none", prior_method: str = "gupdm"
) -> EUVPDataset:
    """
    Primary training corpus.

    Args:
        data_dir  : Root of the EUVP release (contains 'Paired/').
        img_size  : Resize target (height = width). Default: 256.
        subset    : 'all' (notebook default, combines all three paired subsets)
                    or one of 'underwater_imagenet' | 'underwater_dark' |
                    'underwater_scenes', or a comma-separated string.
        in_memory : Load images into RAM during initialization.
    """
    return EUVPDataset(
        data_dir,
        subset=subset,
        transform=_train_transform(img_size),
        augment=True,
        in_memory=in_memory,
        physics_mode=physics_mode,
        prior_method=prior_method,
        img_size=img_size,
    )


def get_uieb_training_set(
    data_dir: str, img_size: int = 256, in_memory: bool = False, physics_mode: str = "none", prior_method: str = "gupdm"
) -> UIEBDataset:
    """
    UIEB training split (800 pairs by convention).

    Args:
        data_dir  : Root of the UIEB release (contains 'raw-890/' etc.).
        img_size  : Resize target. Default: 256.
        in_memory : Load images into RAM during initialization.
    """
    return UIEBDataset(
        data_dir, transform=_train_transform(img_size), augment=True, in_memory=in_memory, physics_mode=physics_mode, prior_method=prior_method, img_size=img_size
    )


def get_ufo120_training_set(
    data_dir: str, img_size: int = 256, in_memory: bool = False, physics_mode: str = "none", prior_method: str = "gupdm"
) -> UFO120Dataset:
    """
    UFO-120 training split (1 500 high-res AUV pairs).

    Args:
        data_dir  : Root of UFO-120 (contains 'train_val/' and 'test/').
        img_size  : Resize target. Default: 256.
        in_memory : Load images into RAM during initialization.
    """
    return UFO120Dataset(
        data_dir,
        split="train",
        transform=_train_transform(img_size),
        augment=True,
        in_memory=in_memory,
        physics_mode=physics_mode,
        prior_method=prior_method,
        img_size=img_size,
    )


# ---------------------------------------------------------------------------
# Evaluation sets
# ---------------------------------------------------------------------------


def get_uieb_eval_set(data_dir: str, factor: int = 8) -> PaddedEvalDataset:
    """
    UIEB test-90 evaluation set (padded for U-Net shape compatibility).

    Args:
        data_dir : Directory containing the 90 test input images.
        factor   : Padding factor — must match encoder depth (default 8 → 3 levels).
    """
    return PaddedEvalDataset(data_dir, factor=factor)


def get_ufo120_eval_set(data_dir: str, factor: int = 8) -> PaddedEvalDataset:
    """
    UFO-120 test set evaluation set (padded).

    Args:
        data_dir : Directory containing UFO-120 test inputs (e.g. 'test/lrd').
        factor   : Padding factor.
    """
    return PaddedEvalDataset(data_dir, factor=factor)


def get_euvp_eval_set(data_dir: str, factor: int = 8) -> PaddedEvalDataset:
    """
    EUVP test set evaluation set (padded).

    Args:
        data_dir : Directory of EUVP test inputs (e.g. 'Paired/<subset>/testA').
        factor   : Padding factor.
    """
    return PaddedEvalDataset(data_dir, factor=factor)


def get_u45_eval_set(data_dir: str) -> SimpleEvalDataset:
    """
    U45 no-reference evaluation set (45 challenging real-world images).
    No ground truth; use UCIQE / UIQM for scoring.

    Args:
        data_dir : Flat directory containing the 45 U45 images.
    """
    return SimpleEvalDataset(data_dir)
