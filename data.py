"""
Underwater Image Restoration — Central data factory
Exposes one get_*_training_set / get_*_eval_set function per dataset so
that train.py only needs a single import.
"""

from torchvision.transforms import (
    Compose, ToTensor,
    RandomCrop, RandomHorizontalFlip, RandomVerticalFlip,
)

from data.UWIRdataset import (
    UIEBDataset,
    EUVPDataset,
    UFO120Dataset,
    U45Dataset,
)
from data.eval_sets import PaddedEvalDataset, SimpleEvalDataset


# ---------------------------------------------------------------------------
# Transform helpers
# ---------------------------------------------------------------------------

def _train_transform(crop_size: int = 256):
    """Random crop + flips + to-tensor for paired training."""
    return Compose([
        RandomCrop((crop_size, crop_size)),
        RandomHorizontalFlip(),
        RandomVerticalFlip(),
        ToTensor(),
    ])


def _eval_transform():
    """To-tensor only — no spatial changes for evaluation."""
    return Compose([ToTensor()])


# ---------------------------------------------------------------------------
# Training sets
# ---------------------------------------------------------------------------

def get_euvp_training_set(data_dir: str, crop_size: int = 256,
                           subset: str = 'underwater_imagenet') -> EUVPDataset:
    """
    Primary training corpus (~12 k paired images).

    Args:
        data_dir  : Root of the EUVP release (contains 'Paired/').
        crop_size : Random crop size for training patches.
        subset    : 'underwater_imagenet' | 'underwater_dark' |
                    'underwater_scenes'.
    """
    return EUVPDataset(data_dir, subset=subset, split='train',
                       transform=_train_transform(crop_size))


def get_uieb_training_set(data_dir: str, crop_size: int = 256) -> UIEBDataset:
    """
    UIEB training split (800 pairs by convention).

    Args:
        data_dir  : Root of the UIEB release (contains 'raw-890/' etc.).
        crop_size : Random crop size for training patches.
    """
    return UIEBDataset(data_dir, transform=_train_transform(crop_size))


def get_ufo120_training_set(data_dir: str, crop_size: int = 256) -> UFO120Dataset:
    """
    UFO-120 training split (1 500 high-res AUV pairs).

    Args:
        data_dir  : Root of UFO-120 (contains 'train_val/' and 'test/').
        crop_size : Random crop size for training patches.
    """
    return UFO120Dataset(data_dir, split='train',
                         transform=_train_transform(crop_size))


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