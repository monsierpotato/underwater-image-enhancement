"""
mobilenet_unet.py
-----------------
MobileNetV3-Large encoder + U-Net decoder for underwater image restoration.

Architecture
------------
MobileNetV3-Large feature stages (stride-based skip points):

  stage0   features[0:1]    16 ch,  H/2
  stage1   features[1:4]    24 ch,  H/4
  stage2   features[4:7]    40 ch,  H/8
  stage3   features[7:13]  112 ch,  H/16
  stage4   features[13:17] 960 ch,  H/32  ← bottleneck

Decoder:

  dec4  (960, skip=112) → 256,  H/16
  dec3  (256,  skip=40) → 128,  H/8
  dec2  (128,  skip=24) →  64,  H/4
  dec1  ( 64,  skip=16) →  32,  H/2
  dec0  ( 32,  skip= 0) →  32,  H
  head  Conv1×1 → Sigmoid → (B, 3, H, W)

Extra channels for physics ablation
------------------------------------
  in_channels = 3  → RGB only
  in_channels = 4  → RGB + t(x)  OR  RGB + B
  in_channels = 5  → RGB + t(x) + B

The first Conv2d inside features[0] is re-initialised for in_channels ≠ 3.
"""

import torch
import torch.nn as nn
from torchvision.models import MobileNet_V3_Large_Weights, mobilenet_v3_large

from .blocks import DecoderBlock


class MobileNetUNet(nn.Module):
    """
    MobileNetV3-Large encoder + lightweight U-Net decoder.

    Args:
        in_channels  (int):  Input channels – 3, 4, or 5.
        out_channels (int):  Output channels. Default: 3.
        pretrained   (bool): Load ImageNet-pretrained MobileNetV3 weights.
                             Default: True.

    Example::

        model = MobileNetUNet(in_channels=5)
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

        weights = MobileNet_V3_Large_Weights.IMAGENET1K_V1 if pretrained else None
        backbone = mobilenet_v3_large(weights=weights)
        features = list(backbone.features)

        # ------------------------------------------------------------------
        # Patch first conv when in_channels ≠ 3
        # features[0] is Conv2dNormActivation (a Sequential);
        # features[0][0] is the raw Conv2d.
        # ------------------------------------------------------------------
        if in_channels != 3:
            orig_conv = features[0][0]
            new_conv = nn.Conv2d(
                in_channels,
                orig_conv.out_channels,
                kernel_size=orig_conv.kernel_size,
                stride=orig_conv.stride,
                padding=orig_conv.padding,
                bias=False,
            )
            if pretrained:
                with torch.no_grad():
                    n = min(in_channels, 3)
                    new_conv.weight[:, :n] = orig_conv.weight[:, :n]
            features[0][0] = new_conv

        # ------------------------------------------------------------------
        # Encoder stages
        # ------------------------------------------------------------------
        self.stage0 = nn.Sequential(*features[0:1])  # 16 ch,  H/2
        self.stage1 = nn.Sequential(*features[1:4])  # 24 ch,  H/4
        self.stage2 = nn.Sequential(*features[4:7])  # 40 ch,  H/8
        self.stage3 = nn.Sequential(*features[7:13])  # 112 ch, H/16
        self.stage4 = nn.Sequential(*features[13:17])  # 960 ch, H/32

        # ------------------------------------------------------------------
        # Decoder  (in_ch, skip_ch, out_ch)
        # ------------------------------------------------------------------
        self.dec4 = DecoderBlock(960, 112, 256)
        self.dec3 = DecoderBlock(256, 40, 128)
        self.dec2 = DecoderBlock(128, 24, 64)
        self.dec1 = DecoderBlock(64, 16, 32)
        self.dec0 = DecoderBlock(32, 0, 32)  # no skip at full resolution

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
        s0 = self.stage0(x)  # (N,  16, H/2,  W/2 )
        s1 = self.stage1(s0)  # (N,  24, H/4,  W/4 )
        s2 = self.stage2(s1)  # (N,  40, H/8,  W/8 )
        s3 = self.stage3(s2)  # (N, 112, H/16, W/16)
        s4 = self.stage4(s3)  # (N, 960, H/32, W/32)

        d4 = self.dec4(s4, s3)  # (N, 256, H/16, W/16)
        d3 = self.dec3(d4, s2)  # (N, 128, H/8,  W/8 )
        d2 = self.dec2(d3, s1)  # (N,  64, H/4,  W/4 )
        d1 = self.dec1(d2, s0)  # (N,  32, H/2,  W/2 )
        d0 = self.dec0(d1)  # (N,  32, H,    W   )

        return self.head(d0)  # (N,   3, H,    W   )
