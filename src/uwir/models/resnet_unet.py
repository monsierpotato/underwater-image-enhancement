"""
resnet_unet.py
--------------
ResNet-50 encoder + U-Net decoder for underwater image restoration.

Architecture
------------
                    Input (B, C_in, H, W)
                           │
    enc0  (64,  H/2)  ─────┐  stem: Conv7×7-BN-ReLU
    enc1  (256, H/4)  ─────┤  ResNet layer1
    enc2  (512, H/8)  ─────┤  ResNet layer2
    enc3  (1024,H/16) ─────┤  ResNet layer3
    enc4  (2048,H/32) ──── bottleneck (ResNet layer4)
    dec4  (512, H/16) ─────┘  upsample + skip(enc3)
    dec3  (256, H/8)  ─────┘  upsample + skip(enc2)
    dec2  (128, H/4)  ─────┘  upsample + skip(enc1)
    dec1  (64,  H/2)  ─────┘  upsample + skip(enc0)
    dec0  (32,  H)    ──────   upsample (no skip)
    head                       Conv1×1 → Sigmoid → (B, 3, H, W)

Extra channels for physics ablation
------------------------------------
  in_channels = 3  → RGB only
  in_channels = 4  → RGB + transmission map  OR  RGB + background light
  in_channels = 5  → RGB + t(x) + B  (full physics-guided)

The first Conv7×7 is re-initialised for in_channels ≠ 3 (pretrained weights
for the first 3 channels are preserved when pretrained=True).
"""

import torch
import torch.nn as nn
from torchvision.models import ResNet50_Weights, resnet50

from .blocks import DecoderBlock


class ResNetUNet(nn.Module):
    """
    ResNet-50 encoder + lightweight U-Net decoder.

    Args:
        in_channels  (int):  Input channels – 3, 4, or 5.
        out_channels (int):  Output channels. Default: 3.
        pretrained   (bool): Load ImageNet-pretrained ResNet-50 weights.
                             Default: True.

    Example::

        model = ResNetUNet(in_channels=5)
        x = torch.randn(2, 5, 256, 256)
        y = model(x)   # (2, 3, 256, 256)
    """

    def __init__(
        self,
        in_channels: int = 5,
        out_channels: int = 3,
        pretrained: bool = True,
    ):
        super().__init__()

        weights = ResNet50_Weights.IMAGENET1K_V1 if pretrained else None
        backbone = resnet50(weights=weights)

        # ------------------------------------------------------------------
        # Patch first conv when in_channels ≠ 3
        # ------------------------------------------------------------------
        if in_channels != 3:
            orig_conv = backbone.conv1
            new_conv = nn.Conv2d(
                in_channels,
                64,
                kernel_size=7,
                stride=2,
                padding=3,
                bias=False,
            )
            if pretrained:
                with torch.no_grad():
                    # Copy weights for the first min(in_channels, 3) channels.
                    n = min(in_channels, 3)
                    new_conv.weight[:, :n] = orig_conv.weight[:, :n]
            backbone.conv1 = new_conv

        # ------------------------------------------------------------------
        # Encoder stages
        # ------------------------------------------------------------------
        self.enc0 = nn.Sequential(backbone.conv1, backbone.bn1, backbone.relu)
        #             ↑ 64 ch, H/2
        self.pool = backbone.maxpool  # 64 ch, H/4
        self.enc1 = backbone.layer1  # 256 ch, H/4
        self.enc2 = backbone.layer2  # 512 ch, H/8
        self.enc3 = backbone.layer3  # 1024 ch, H/16
        self.enc4 = backbone.layer4  # 2048 ch, H/32  ← bottleneck

        # ------------------------------------------------------------------
        # Decoder  (in_ch, skip_ch, out_ch)
        # ------------------------------------------------------------------
        self.dec4 = DecoderBlock(2048, 1024, 512)
        self.dec3 = DecoderBlock(512, 512, 256)
        self.dec2 = DecoderBlock(256, 256, 128)
        self.dec1 = DecoderBlock(128, 64, 64)
        self.dec0 = DecoderBlock(64, 0, 32)  # no skip at full resolution

        # Output head
        self.head = nn.Sequential(
            nn.Conv2d(32, out_channels, kernel_size=1),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x (Tensor): (N, C_in, H, W)

        Returns:
            Tensor: (N, 3, H, W) restored RGB in [0, 1].
        """
        e0 = self.enc0(x)  # (N,   64, H/2,  W/2 )
        p = self.pool(e0)  # (N,   64, H/4,  W/4 )
        e1 = self.enc1(p)  # (N,  256, H/4,  W/4 )
        e2 = self.enc2(e1)  # (N,  512, H/8,  W/8 )
        e3 = self.enc3(e2)  # (N, 1024, H/16, W/16)
        e4 = self.enc4(e3)  # (N, 2048, H/32, W/32)

        d4 = self.dec4(e4, e3)  # (N,  512, H/16, W/16)
        d3 = self.dec3(d4, e2)  # (N,  256, H/8,  W/8 )
        d2 = self.dec2(d3, e1)  # (N,  128, H/4,  W/4 )
        d1 = self.dec1(d2, e0)  # (N,   64, H/2,  W/2 )
        d0 = self.dec0(d1)  # (N,   32, H,    W   )

        return self.head(d0)  # (N,    3, H,    W   )
