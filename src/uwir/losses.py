"""
losses.py
---------
Loss functions for the physics-guided 5-channel U-Net
(underwater image restoration, EUVP dataset).

Classes
-------
VGGPerceptualLoss  – VGG-16 feature-space L1 at relu1_2 + relu2_2
SSIMLoss           – 1 − SSIM (via kornia)
CompositeLoss      – λ_l1·L1 + λ_perc·Perceptual + λ_ssim·SSIM
"""

import kornia
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import VGG16_Weights, vgg16

# ImageNet statistics for VGG normalisation
_VGG_MEAN = torch.tensor([0.485, 0.456, 0.406])
_VGG_STD = torch.tensor([0.229, 0.224, 0.225])


# ---------------------------------------------------------------------------
# VGG Perceptual Loss
# ---------------------------------------------------------------------------


class VGGPerceptualLoss(nn.Module):
    """
    VGG-16 feature-space loss using relu1_2 and relu2_2 activations.

    Args:
        device (str | torch.device): Target device for the frozen VGG backbone.
    """

    def __init__(self, device: str | torch.device = "cpu"):
        super().__init__()
        vgg = vgg16(weights=VGG16_Weights.IMAGENET1K_V1).features
        # relu1_2 → first 4 children; relu2_2 → children 4-9
        self.stage1 = nn.Sequential(*list(vgg.children())[:4]).to(device).eval()
        self.stage2 = nn.Sequential(*list(vgg.children())[4:9]).to(device).eval()
        for p in self.parameters():
            p.requires_grad = False

    def _normalise(self, x: torch.Tensor) -> torch.Tensor:
        """Apply ImageNet mean/std normalisation expected by VGG."""
        mean = _VGG_MEAN.to(x.device, x.dtype).view(1, 3, 1, 1)
        std = _VGG_STD.to(x.device, x.dtype).view(1, 3, 1, 1)
        return (x - mean) / std

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Args:
            pred   (Tensor): (N, 3, H, W) predicted RGB in [0, 1].
            target (Tensor): (N, 3, H, W) ground-truth RGB in [0, 1].

        Returns:
            Tensor: scalar perceptual loss.
        """
        # VGG expects ImageNet-normalised inputs; raw [0,1] causes large
        # intermediate activations that can overflow gradients to NaN.
        pred = self._normalise(pred.clamp(0, 1))
        target = self._normalise(target.clamp(0, 1))

        p1 = self.stage1(pred)
        t1 = self.stage1(target)
        p2 = self.stage2(p1)
        t2 = self.stage2(t1)
        return F.l1_loss(p1, t1) + F.l1_loss(p2, t2)


# ---------------------------------------------------------------------------
# SSIM Loss
# ---------------------------------------------------------------------------


class SSIMLoss(nn.Module):
    """
    SSIM-based loss: ``1 − mean(SSIM map)``.
    Uses ``kornia.metrics.ssim`` which returns the per-pixel SSIM map.

    Args:
        window_size (int): Gaussian window size. Default: 11.
    """

    def __init__(self, window_size: int = 11):
        super().__init__()
        self.window_size = window_size

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Args:
            pred   (Tensor): (N, C, H, W) in [0, 1].
            target (Tensor): (N, C, H, W) in [0, 1].

        Returns:
            Tensor: scalar loss ∈ [0, 2].
        """
        ssim_map = kornia.metrics.ssim(pred, target, self.window_size)
        return 1.0 - ssim_map.mean()


# ---------------------------------------------------------------------------
# Composite Loss
# ---------------------------------------------------------------------------


class CompositeLoss(nn.Module):
    """
    Weighted combination of L1, VGG perceptual, and SSIM losses:

        loss = λ_l1 · L1 + λ_perc · Perceptual + λ_ssim · SSIM

    Args:
        lambda_l1   (float): Weight for L1 loss.           Default: 1.0.
        lambda_perc (float): Weight for perceptual loss.   Default: 0.1.
        lambda_ssim (float): Weight for SSIM loss.         Default: 0.5.
        device (str | torch.device): Device for VGG backbone.
    """

    def __init__(
        self,
        lambda_l1: float = 1.0,
        lambda_perc: float = 0.1,
        lambda_ssim: float = 0.5,
        device: str | torch.device = "cpu",
    ):
        super().__init__()
        self.lambda_l1 = lambda_l1
        self.lambda_perc = lambda_perc
        self.lambda_ssim = lambda_ssim

        self.l1 = nn.L1Loss()
        self.perc = VGGPerceptualLoss(device) if lambda_perc else None
        self.ssim = SSIMLoss()

    def forward(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
    ) -> tuple[torch.Tensor, dict[str, float]]:
        """
        Args:
            pred   (Tensor): (N, 3, H, W) model output in [0, 1].
            target (Tensor): (N, 3, H, W) ground truth in [0, 1].

        Returns:
            total (Tensor): Scalar combined loss.
            parts (dict):   Per-component losses as Python floats
                            with keys ``"l1"``, ``"perceptual"``,
                            ``"ssim_loss"``, ``"total"``.
        """
        l_l1 = self.l1(pred, target)
        l_perc = self.perc(pred, target) if self.perc is not None else pred.new_zeros(())
        l_ssim = self.ssim(pred, target)

        total = self.lambda_l1 * l_l1 + self.lambda_perc * l_perc + self.lambda_ssim * l_ssim

        parts = {
            "l1": l_l1.item(),
            "perceptual": l_perc.item(),
            "ssim_loss": l_ssim.item(),
            "total": total.item(),
        }
        return total, parts
