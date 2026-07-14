"""
unet.py
-------
4-level U-Net for underwater image restoration.

The same class handles both the RGB baseline and the physics-guided
5-channel variant; the only difference is ``in_channels``:

    in_channels = 3  →  RGB baseline  (unet_rgb notebook)
    in_channels = 5  →  RGB + t(x) + B  (this notebook, unet_5ch)

Architecture
------------
                 Input (B, C_in, H, W)
                        │
            enc1  64  ──┐ DoubleConv
            enc2  128 ──┤ MaxPool → DoubleConv
            enc3  256 ──┤ MaxPool → DoubleConv
            enc4  512 ──┤ MaxPool → DoubleConv
         bottleneck 1024  MaxPool → DoubleConv
            dec4  512 ──┘ Upsample + skip
            dec3  256 ──┘ Upsample + skip
            dec2  128 ──┘ Upsample + skip
            dec1   64 ──┘ Upsample + skip
               head   Conv1×1 → Sigmoid
                        │
                Output (B, 3, H, W) in [0, 1]
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

# ---------------------------------------------------------------------------
# Building blocks
# ---------------------------------------------------------------------------


class DoubleConv(nn.Module):
    """
    Two successive (Conv3×3 → BN → ReLU) blocks.

    Args:
        in_ch   (int):   Input channels.
        out_ch  (int):   Output channels.
        dropout (float): Dropout2d probability after second ReLU. Default: 0.
    """

    def __init__(self, in_ch: int, out_ch: int, dropout: float = 0.0):
        super().__init__()
        layers = [
            nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        ]
        if dropout > 0.0:
            layers.append(nn.Dropout2d(dropout))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class Down(nn.Module):
    """MaxPool2d(2) followed by DoubleConv."""

    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.net = nn.Sequential(nn.MaxPool2d(2), DoubleConv(in_ch, out_ch))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class Up(nn.Module):
    """
    Decoder step: upsample ``x1``, pad to match skip ``x2``, concatenate, convolve.

    Args:
        prev_ch  (int):  Channels coming from the deeper layer.
        skip_ch  (int):  Channels from the matching encoder skip connection.
        out_ch   (int):  Output channels after DoubleConv.
        bilinear (bool): Use bilinear upsampling; False uses ConvTranspose2d.
    """

    def __init__(self, prev_ch: int, skip_ch: int, out_ch: int, bilinear: bool = True):
        super().__init__()
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
            self.conv = DoubleConv(prev_ch + skip_ch, out_ch)
        else:
            self.up = nn.ConvTranspose2d(prev_ch, prev_ch // 2, kernel_size=2, stride=2)
            self.conv = DoubleConv((prev_ch // 2) + skip_ch, out_ch)

    def forward(self, x1: torch.Tensor, x2: torch.Tensor) -> torch.Tensor:
        x1 = self.up(x1)
        # Pad x1 to match x2 when spatial dims are odd
        dY = x2.size(2) - x1.size(2)
        dX = x2.size(3) - x1.size(3)
        x1 = F.pad(x1, [dX // 2, dX - dX // 2, dY // 2, dY - dY // 2])
        return self.conv(torch.cat([x2, x1], dim=1))


# ---------------------------------------------------------------------------
# U-Net
# ---------------------------------------------------------------------------


class UNet5ch(nn.Module):
    """
    Standard 4-level U-Net for underwater image restoration.

    Accepts either 3-channel (RGB) or 5-channel (RGB + physics) input
    via the ``in_channels`` argument.  The output is always a 3-channel
    RGB image in [0, 1] (Sigmoid head).

    Args:
        in_channels  (int):          Input channels – 3 for RGB, 5 for physics-guided.
        out_channels (int):          Output channels. Default: 3.
        features     (tuple[int]):   Feature map sizes at each encoder level.
                                     Default: (64, 128, 256, 512).
        bilinear     (bool):         Bilinear upsampling vs ConvTranspose2d.
                                     Default: True.

    Example::

        model = UNet5ch(in_channels=5)
        x = torch.randn(2, 5, 256, 256)
        y = model(x)          # (2, 3, 256, 256)
    """

    def __init__(
        self,
        in_channels: int = 5,
        out_channels: int = 3,
        features: tuple[int, ...] = (64, 128, 256, 512),
        bilinear: bool = True,
    ):
        super().__init__()
        f = features

        # Encoder
        self.enc1 = DoubleConv(in_channels, f[0])
        self.enc2 = Down(f[0], f[1])
        self.enc3 = Down(f[1], f[2])
        self.enc4 = Down(f[2], f[3])

        # Bottleneck  (f[3] × 2 = 1024 for default features)
        self.bottleneck = Down(f[3], f[3] * 2)

        # Decoder – channels: (from_below, skip, out)
        self.dec4 = Up(f[3] * 2, f[3], f[3], bilinear)
        self.dec3 = Up(f[3], f[2], f[2], bilinear)
        self.dec2 = Up(f[2], f[1], f[1], bilinear)
        self.dec1 = Up(f[1], f[0], f[0], bilinear)

        # Output head: 1×1 conv + Sigmoid → [0, 1]
        self.head = nn.Sequential(
            nn.Conv2d(f[0], out_channels, kernel_size=1),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x (Tensor): (N, C_in, H, W)  C_in = 3 or 5.

        Returns:
            Tensor: (N, 3, H, W) restored RGB image in [0, 1].
        """
        e1 = self.enc1(x)  # (N,  64, H,   W  )
        e2 = self.enc2(e1)  # (N, 128, H/2, W/2)
        e3 = self.enc3(e2)  # (N, 256, H/4, W/4)
        e4 = self.enc4(e3)  # (N, 512, H/8, W/8)
        bn = self.bottleneck(e4)  # (N,1024, H/16,W/16)

        d4 = self.dec4(bn, e4)  # (N, 512, H/8, W/8)
        d3 = self.dec3(d4, e3)  # (N, 256, H/4, W/4)
        d2 = self.dec2(d3, e2)  # (N, 128, H/2, W/2)
        d1 = self.dec1(d2, e1)  # (N,  64, H,   W  )

        return self.head(d1)  # (N,   3, H,   W  )
