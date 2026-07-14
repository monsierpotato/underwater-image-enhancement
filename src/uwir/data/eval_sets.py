"""
Underwater Image Restoration — Evaluation-set wrappers
Provides padding-safe loaders used during inference / metric computation.
"""

import os
from os import listdir
from os.path import join

import torch.nn.functional as F
import torch.utils.data as data
from torchvision.transforms import ToTensor

from .utils import is_image_file, load_img

# ---------------------------------------------------------------------------
# Generic padded eval loader  (UIEB test-90, UFO-120 test)
# Pads each image so that both spatial dimensions are divisible by `factor`.
# Original h, w are returned so the padding can be stripped after inference.
# ---------------------------------------------------------------------------


class PaddedEvalDataset(data.Dataset):
    """
    Loads images from a flat directory and pads them to a multiple of
    `factor` pixels (using reflect padding) so that any U-Net encoder
    with `factor / 2` downsampling stages produces no shape mismatches.

    Returns:
        (tensor, filename, original_h, original_w)
    """

    def __init__(self, data_dir, factor: int = 8):
        super().__init__()
        self.factor = factor
        self.to_tensor = ToTensor()
        self.data_files = sorted(join(data_dir, f) for f in listdir(data_dir) if is_image_file(f))

    def __getitem__(self, index):
        img = load_img(self.data_files[index])
        _, filename = os.path.split(self.data_files[index])
        tensor = self.to_tensor(img)  # (3, H, W)

        h, w = tensor.shape[1], tensor.shape[2]
        pad_h = (self.factor - h % self.factor) % self.factor
        pad_w = (self.factor - w % self.factor) % self.factor
        tensor = F.pad(tensor.unsqueeze(0), (0, pad_w, 0, pad_h), mode="reflect").squeeze(0)

        return tensor, filename, h, w

    def __len__(self):
        return len(self.data_files)


# ---------------------------------------------------------------------------
# Simple (no-padding) eval loader  — U45 and any flat folder of images
# ---------------------------------------------------------------------------


class SimpleEvalDataset(data.Dataset):
    """
    Loads images from a flat directory without any padding.
    Suitable for U45 (no-reference) or any fixed-size evaluation set.

    Returns:
        (tensor, filename)
    """

    def __init__(self, data_dir):
        super().__init__()
        self.to_tensor = ToTensor()
        self.data_files = sorted(join(data_dir, f) for f in listdir(data_dir) if is_image_file(f))

    def __getitem__(self, index):
        img = load_img(self.data_files[index])
        _, filename = os.path.split(self.data_files[index])
        tensor = self.to_tensor(img)
        return tensor, filename

    def __len__(self):
        return len(self.data_files)
