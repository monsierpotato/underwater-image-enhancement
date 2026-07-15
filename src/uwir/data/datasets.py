"""
Underwater Image Restoration — Dataset Loaders
Covers: UIEB, EUVP, UFO-120 (paired, with ground truth)
        U45  (unpaired / no-reference, eval only)
"""

import os
import random
from os import listdir
from os.path import join

import torch.utils.data as data
import torchvision.transforms.functional as TF

try:
    from tqdm.auto import tqdm
except ImportError:

    def tqdm(iterable, **_kwargs):
        return iterable


from .utils import is_image_file, load_img

# ---------------------------------------------------------------------------
# Helper: Append physics channels BEFORE random spatial augmentation
# ---------------------------------------------------------------------------

def _append_physics_channels(img_in, file_in, physics_mode, prior_method, img_size):
    if physics_mode == "none":
        return img_in

    import torch
    import numpy as np
    parent_dir = os.path.dirname(os.path.dirname(file_in))
    cache_dir = os.path.join(parent_dir, f"physics_cache_{prior_method}_{img_size}")
    stem = os.path.splitext(os.path.basename(file_in))[0]
    cache_path = os.path.join(cache_dir, f"{stem}.npz")
    
    if not os.path.exists(cache_path):
        import hashlib
        path_hash = hashlib.md5(parent_dir.encode('utf-8')).hexdigest()
        local_cache_dir = os.path.join(os.getcwd(), "physics_cache", f"{prior_method}_{img_size}", path_hash)
        cache_path = os.path.join(local_cache_dir, f"{stem}.npz")
        
    if os.path.exists(cache_path):
        data = np.load(cache_path)
        t = torch.from_numpy(data['t']).unsqueeze(0)
        b = torch.from_numpy(data['b']).unsqueeze(0)
    else:
        # Fallback to compute on-the-fly if cache is missing (very slow, precompute recommended!)
        from uwir.physics import compute_physics_maps, compute_physics_maps_gdcp, compute_physics_maps_gupdm
        img_np = img_in.permute(1, 2, 0).numpy().astype("float32")
        if prior_method == "gupdm":
            t_map, b_map = compute_physics_maps_gupdm(img_np)
        elif prior_method == "udcp":
            t_map, b_map = compute_physics_maps(img_np)
        elif prior_method == "gdcp":
            t_map, b_map = compute_physics_maps_gdcp(img_np)
        else:
            raise ValueError(f"Unknown prior: {prior_method}")
        t = torch.from_numpy(t_map).unsqueeze(0)
        b = torch.from_numpy(b_map).unsqueeze(0)

    if physics_mode == "t":
        img_in = torch.cat([img_in, t], dim=0)
    elif physics_mode == "b":
        img_in = torch.cat([img_in, b], dim=0)
    elif physics_mode == "tb":
        img_in = torch.cat([img_in, t, b], dim=0)
        
    return img_in


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
        transform: Torchvision transform applied to both images (e.g. Resize + ToTensor).
        augment (bool): Apply random hflip / vflip / ±10° rotation to both images
                        simultaneously (training only). Default: False.
    """

    INPUT_DIR = "raw-890"
    GT_DIR = "reference-890"

    def __init__(self, data_dir, transform=None, augment=False, in_memory=False, physics_mode="none", prior_method="gupdm", img_size=256):
        super().__init__()
        self.input_dir = join(data_dir, self.INPUT_DIR)
        self.gt_dir = join(data_dir, self.GT_DIR)
        self.transform = transform
        self.augment = augment
        self.in_memory = in_memory
        self.physics_mode = physics_mode
        self.prior_method = prior_method
        self.img_size = img_size

        # Stem-name matching (robust against ordering differences)
        gt_dict = {
            os.path.splitext(f)[0]: join(self.gt_dir, f)
            for f in listdir(self.gt_dir)
            if is_image_file(f)
        }
        self.input_files = []
        self.gt_files = []
        for f in sorted(listdir(self.input_dir)):
            if not is_image_file(f):
                continue
            stem = os.path.splitext(f)[0]
            if stem in gt_dict:
                self.input_files.append(join(self.input_dir, f))
                self.gt_files.append(gt_dict[stem])

        assert len(self.input_files) == len(self.gt_files), (
            f"UIEB: mismatched file counts "
            f"({len(self.input_files)} inputs vs {len(self.gt_files)} GTs)"
        )

        if self.in_memory:
            print("Loading UIEB dataset into memory...")
            self.input_images = [load_img(f) for f in tqdm(self.input_files, desc="UIEB Inputs")]
            self.gt_images = [load_img(f) for f in tqdm(self.gt_files, desc="UIEB GTs")]

    def __getitem__(self, index):
        if self.in_memory:
            img_in = self.input_images[index]
            img_gt = self.gt_images[index]
        else:
            img_in = load_img(self.input_files[index])
            img_gt = load_img(self.gt_files[index])
        _, file_in = os.path.split(self.input_files[index])
        _, file_gt = os.path.split(self.gt_files[index])

        if self.transform:
            img_in = self.transform(img_in)
            img_gt = self.transform(img_gt)

        img_in = _append_physics_channels(img_in, self.input_files[index], self.physics_mode, self.prior_method, self.img_size)

        if self.augment:
            img_in, img_gt = _paired_augment(img_in, img_gt)

        return img_in, img_gt, self.input_files[index], file_gt

    def __len__(self):
        return len(self.input_files)


# ---------------------------------------------------------------------------
# Shared augmentation helper (coordinated on paired PIL images)
# ---------------------------------------------------------------------------


def _paired_augment(img_in, img_gt):
    """
    Apply the same random geometric transforms to both images.
    Matches the notebook's _augment() method:
      - 50 % random horizontal flip
      - 50 % random vertical flip
      - 50 % random rotation in [−10°, +10°]

    Args:
        img_in, img_gt: PIL Images (after Resize, before ToTensor).
    Returns:
        Augmented (img_in, img_gt) PIL Images.
    """
    if random.random() > 0.5:
        img_in = TF.hflip(img_in)
        img_gt = TF.hflip(img_gt)
    if random.random() > 0.5:
        img_in = TF.vflip(img_in)
        img_gt = TF.vflip(img_gt)
    if random.random() > 0.5:
        angle = random.uniform(-10, 10)
        img_in = TF.rotate(img_in, angle)
        img_gt = TF.rotate(img_gt, angle)
    return img_in, img_gt


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
        subset    (str | list[str]): One or more of
                         'underwater_imagenet' | 'underwater_dark' |
                         'underwater_scenes'.  Pass 'all' or a list to
                         combine multiple subsets (notebook default).
        transform: Applied to both input and GT images (e.g. Resize + ToTensor).
        augment (bool): Apply random hflip / vflip / ±10° rotation to both
                        images simultaneously (training only). Default: False.
    """

    SUBSETS = ("underwater_imagenet", "underwater_dark", "underwater_scenes")

    def __init__(self, data_dir, subset="all", transform=None, augment=False, in_memory=False, physics_mode="none", prior_method="gupdm", img_size=256):
        super().__init__()

        # Resolve subset list
        if subset == "all":
            subsets = list(self.SUBSETS)
        elif isinstance(subset, str):
            # Support comma-separated: "underwater_dark,underwater_scenes"
            subsets = [s.strip() for s in subset.split(",")]
        else:
            subsets = list(subset)  # already a list

        for s in subsets:
            assert s in self.SUBSETS, f"subset must be one of {self.SUBSETS}, got '{s}'"

        self.transform = transform
        self.augment = augment
        self.in_memory = in_memory
        self.physics_mode = physics_mode
        self.prior_method = prior_method
        self.img_size = img_size
        self.input_files = []
        self.gt_files = []

        for s in subsets:
            input_dir = join(data_dir, "Paired", s, "trainA")
            gt_dir = join(data_dir, "Paired", s, "trainB")

            if not (os.path.isdir(input_dir) and os.path.isdir(gt_dir)):
                print(f"  [WARN] Missing: {input_dir} or {gt_dir} — skipping subset '{s}'")
                continue

            # Stem-name matching (robust against filename order differences)
            gt_dict = {
                os.path.splitext(f)[0]: join(gt_dir, f) for f in listdir(gt_dir) if is_image_file(f)
            }
            for f in sorted(listdir(input_dir)):
                if not is_image_file(f):
                    continue
                stem = os.path.splitext(f)[0]
                if stem in gt_dict:
                    self.input_files.append(join(input_dir, f))
                    self.gt_files.append(gt_dict[stem])

        if self.in_memory:
            print("Loading EUVP dataset into memory...")
            self.input_images = [load_img(f) for f in tqdm(self.input_files, desc="EUVP Inputs")]
            self.gt_images = [load_img(f) for f in tqdm(self.gt_files, desc="EUVP GTs")]

    def __getitem__(self, index):
        if self.in_memory:
            img_in = self.input_images[index]
            img_gt = self.gt_images[index]
        else:
            img_in = load_img(self.input_files[index])
            img_gt = load_img(self.gt_files[index])
        _, file_in = os.path.split(self.input_files[index])
        _, file_gt = os.path.split(self.gt_files[index])

        if self.transform:
            img_in = self.transform(img_in)
            img_gt = self.transform(img_gt)

        img_in = _append_physics_channels(img_in, self.input_files[index], self.physics_mode, self.prior_method, self.img_size)

        if self.augment:
            img_in, img_gt = _paired_augment(img_in, img_gt)

        return img_in, img_gt, self.input_files[index], file_gt

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
        transform: Applied to both input and GT images.
        augment (bool): Apply random augmentation to both images. Default: False.
    """

    SPLIT_MAP = {
        "train": ("train_val/lrd", "train_val/hr"),
        "test": ("test/lrd", "test/hr"),
    }

    def __init__(self, data_dir, split="train", transform=None, augment=False, in_memory=False, physics_mode="none", prior_method="gupdm", img_size=256):
        super().__init__()

        assert split in self.SPLIT_MAP, f"split must be 'train' or 'test', got '{split}'"

        input_subdir, gt_subdir = self.SPLIT_MAP[split]
        input_dir = join(data_dir, input_subdir)
        gt_dir = join(data_dir, gt_subdir)

        self.transform = transform
        self.augment = augment
        self.in_memory = in_memory
        self.physics_mode = physics_mode
        self.prior_method = prior_method
        self.img_size = img_size

        # Stem-name matching
        gt_dict = {
            os.path.splitext(f)[0]: join(gt_dir, f) for f in listdir(gt_dir) if is_image_file(f)
        }
        self.input_files = []
        self.gt_files = []
        for f in sorted(listdir(input_dir)):
            if not is_image_file(f):
                continue
            stem = os.path.splitext(f)[0]
            if stem in gt_dict:
                self.input_files.append(join(input_dir, f))
                self.gt_files.append(gt_dict[stem])

        if self.in_memory:
            print("Loading UFO-120 dataset into memory...")
            self.input_images = [load_img(f) for f in tqdm(self.input_files, desc="UFO-120 Inputs")]
            self.gt_images = [load_img(f) for f in tqdm(self.gt_files, desc="UFO-120 GTs")]

    def __getitem__(self, index):
        if self.in_memory:
            img_in = self.input_images[index]
            img_gt = self.gt_images[index]
        else:
            img_in = load_img(self.input_files[index])
            img_gt = load_img(self.gt_files[index])
        _, file_in = os.path.split(self.input_files[index])
        _, file_gt = os.path.split(self.gt_files[index])

        if self.transform:
            img_in = self.transform(img_in)
            img_gt = self.transform(img_gt)

        img_in = _append_physics_channels(img_in, self.input_files[index], self.physics_mode, self.prior_method, self.img_size)

        if self.augment:
            img_in, img_gt = _paired_augment(img_in, img_gt)

        return img_in, img_gt, self.input_files[index], file_gt

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

    def __init__(self, data_dir, transform=None, in_memory=False):
        super().__init__()
        self.transform = transform
        self.in_memory = in_memory
        self.input_files = sorted(join(data_dir, f) for f in listdir(data_dir) if is_image_file(f))

        if self.in_memory:
            print("Loading U45 dataset into memory...")
            self.input_images = [load_img(f) for f in tqdm(self.input_files, desc="U45 Inputs")]

    def __getitem__(self, index):
        img_in = self.input_images[index] if self.in_memory else load_img(self.input_files[index])
        _, file_in = os.path.split(self.input_files[index])

        if self.transform:
            img_in = self.transform(img_in)

        return img_in, self.input_files[index]

    def __len__(self):
        return len(self.input_files)
