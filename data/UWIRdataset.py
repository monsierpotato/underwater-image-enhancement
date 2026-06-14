"""
Underwater Image Restoration — Dataset Loaders
Covers: UIEB, EUVP, UFO-120 (paired, with ground truth)
        U45  (unpaired / no-reference, eval only)
"""

import os
import random

import numpy as np
import torch
import torch.utils.data as data
from os import listdir
from os.path import join

from data.util import is_image_file, load_img


# ---------------------------------------------------------------------------
# UIEB  (890 real-world pairs; 800 train / 90 test split by convention)
# Expected layout:
#   <data_dir>/raw-890/    ← degraded inputs
#   <data_dir>/reference-890/  ← human-rated references (ground truth)
# ---------------------------------------------------------------------------

class UIEBDataset(data.Dataset):
    """
    Paired UIEB dataset for training and full-reference evaluation.

    Args:
        data_dir (str): Root directory that contains 'raw-890/' and
                        'reference-890/' sub-folders.
        transform: Torchvision transform applied *identically* to both
                   the input and the ground-truth image.
    """

    INPUT_DIR = 'raw-890'
    GT_DIR    = 'reference-890'

    def __init__(self, data_dir, transform=None):
        super(UIEBDataset, self).__init__()
        self.input_dir = join(data_dir, self.INPUT_DIR)
        self.gt_dir    = join(data_dir, self.GT_DIR)
        self.transform = transform

        self.input_files = sorted(
            join(self.input_dir, f)
            for f in listdir(self.input_dir)
            if is_image_file(f)
        )
        self.gt_files = sorted(
            join(self.gt_dir, f)
            for f in listdir(self.gt_dir)
            if is_image_file(f)
        )

        assert len(self.input_files) == len(self.gt_files), (
            f"UIEB: mismatched file counts "
            f"({len(self.input_files)} inputs vs {len(self.gt_files)} GTs)"
        )

    def __getitem__(self, index):
        img_in = load_img(self.input_files[index])
        img_gt = load_img(self.gt_files[index])
        _, file_in = os.path.split(self.input_files[index])
        _, file_gt = os.path.split(self.gt_files[index])

        if self.transform:
            seed = np.random.randint(random.randint(1, 1_000_000))
            random.seed(seed)
            torch.manual_seed(seed)
            img_in = self.transform(img_in)
            random.seed(seed)
            torch.manual_seed(seed)
            img_gt = self.transform(img_gt)

        return img_in, img_gt, file_in, file_gt

    def __len__(self):
        return len(self.input_files)


# ---------------------------------------------------------------------------
# EUVP  (~12 k paired images across several scene sub-sets)
# Expected layout (mirrors official EUVP release):
#   <data_dir>/Paired/underwater_imagenet/trainA/  ← degraded
#   <data_dir>/Paired/underwater_imagenet/trainB/  ← clean reference
# The sub-set name (e.g. 'underwater_imagenet', 'underwater_dark',
# 'underwater_scenes') is controlled via the `subset` argument.
# ---------------------------------------------------------------------------

class EUVPDataset(data.Dataset):
    """
    Large-scale paired EUVP dataset.  Used as the primary training corpus.

    Actual folder layout (Paired branch only has trainA / trainB):
        <data_dir>/Paired/<subset>/trainA/   ← degraded inputs
        <data_dir>/Paired/<subset>/trainB/   ← clean references
        <data_dir>/Paired/<subset>/validation/  ← unpaired (no GT)

    There is no testA / testB.  For a held-out validation set with ground
    truth, use torch.utils.data.random_split on the training data.

    Args:
        data_dir  (str): Root directory of the EUVP release
                         (the folder that contains 'Paired/').
        subset    (str): One of 'underwater_imagenet' | 'underwater_dark' |
                         'underwater_scenes'.  Defaults to 'underwater_imagenet'.
        transform: Applied identically to both input and GT.
    """

    SUBSETS = ('underwater_imagenet', 'underwater_dark', 'underwater_scenes')

    def __init__(self, data_dir, subset='underwater_imagenet', transform=None):
        super(EUVPDataset, self).__init__()

        assert subset in self.SUBSETS, \
            f"subset must be one of {self.SUBSETS}, got '{subset}'"

        input_dir = join(data_dir, 'Paired', subset, 'trainA')
        gt_dir    = join(data_dir, 'Paired', subset, 'trainB')

        self.transform = transform
        self.input_files = sorted(
            join(input_dir, f) for f in listdir(input_dir) if is_image_file(f)
        )
        self.gt_files = sorted(
            join(gt_dir, f) for f in listdir(gt_dir) if is_image_file(f)
        )

        assert len(self.input_files) == len(self.gt_files), (
            f"EUVP/{subset}: mismatched file counts "
            f"({len(self.input_files)} inputs vs {len(self.gt_files)} GTs)"
        )

    def __getitem__(self, index):
        img_in = load_img(self.input_files[index])
        img_gt = load_img(self.gt_files[index])
        _, file_in = os.path.split(self.input_files[index])
        _, file_gt = os.path.split(self.gt_files[index])

        if self.transform:
            seed = np.random.randint(random.randint(1, 1_000_000))
            random.seed(seed)
            torch.manual_seed(seed)
            img_in = self.transform(img_in)
            random.seed(seed)
            torch.manual_seed(seed)
            img_gt = self.transform(img_gt)

        return img_in, img_gt, file_in, file_gt

    def __len__(self):
        return len(self.input_files)


# ---------------------------------------------------------------------------
# UFO-120  (high-resolution AUV pairs, 1500 train / 120 test)
# Expected layout:
#   <data_dir>/train_val/lrd/   ← low-resolution / degraded inputs
#   <data_dir>/train_val/hr/    ← high-quality references
#   <data_dir>/test/lrd/
#   <data_dir>/test/hr/
# ---------------------------------------------------------------------------

class UFO120Dataset(data.Dataset):
    """
    UFO-120 high-resolution paired dataset.

    Args:
        data_dir (str): Root UFO-120 directory.
        split    (str): 'train' or 'test'.
        transform: Applied identically to both input and GT.
    """

    SPLIT_MAP = {
        'train': ('train_val/lrd', 'train_val/hr'),
        'test' : ('test/lrd',      'test/hr'),
    }

    def __init__(self, data_dir, split='train', transform=None):
        super(UFO120Dataset, self).__init__()

        assert split in self.SPLIT_MAP, \
            f"split must be 'train' or 'test', got '{split}'"

        input_subdir, gt_subdir = self.SPLIT_MAP[split]
        input_dir = join(data_dir, input_subdir)
        gt_dir    = join(data_dir, gt_subdir)

        self.transform = transform
        self.input_files = sorted(
            join(input_dir, f) for f in listdir(input_dir) if is_image_file(f)
        )
        self.gt_files = sorted(
            join(gt_dir, f) for f in listdir(gt_dir) if is_image_file(f)
        )

        assert len(self.input_files) == len(self.gt_files), (
            f"UFO-120/{split}: mismatched file counts "
            f"({len(self.input_files)} inputs vs {len(self.gt_files)} GTs)"
        )

    def __getitem__(self, index):
        img_in = load_img(self.input_files[index])
        img_gt = load_img(self.gt_files[index])
        _, file_in = os.path.split(self.input_files[index])
        _, file_gt = os.path.split(self.gt_files[index])

        if self.transform:
            seed = np.random.randint(random.randint(1, 1_000_000))
            random.seed(seed)
            torch.manual_seed(seed)
            img_in = self.transform(img_in)
            random.seed(seed)
            torch.manual_seed(seed)
            img_gt = self.transform(img_gt)

        return img_in, img_gt, file_in, file_gt

    def __len__(self):
        return len(self.input_files)


# ---------------------------------------------------------------------------
# U45  (45 challenging real-world images, NO ground truth)
# Used only for no-reference metric evaluation (UCIQE, UIQM).
# Expected layout:  <data_dir>/*.jpg  (flat folder of degraded images)
# ---------------------------------------------------------------------------

class U45Dataset(data.Dataset):
    """
    Unpaired U45 dataset for no-reference evaluation only.

    Args:
        data_dir (str): Flat directory containing the 45 underwater images.
        transform: Applied to input images only.
    """

    def __init__(self, data_dir, transform=None):
        super(U45Dataset, self).__init__()
        self.transform = transform
        self.input_files = sorted(
            join(data_dir, f) for f in listdir(data_dir) if is_image_file(f)
        )

    def __getitem__(self, index):
        img_in = load_img(self.input_files[index])
        _, file_in = os.path.split(self.input_files[index])

        if self.transform:
            img_in = self.transform(img_in)

        return img_in, file_in

    def __len__(self):
        return len(self.input_files)
