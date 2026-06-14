"""
blocks.py
---------
Shared building blocks reused across all encoder-decoder architectures.

Classes
-------
DoubleConv   – two successive Conv3×3 → BN → ReLU blocks
DecoderBlock – bilinear upsample → optional skip-concat → DoubleConv
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


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
            nn.Conv2d(in_ch,  out_ch, kernel_size=3, padding=1, bias=False),
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


class DecoderBlock(nn.Module):
    """
    One decoder step:

        x  → bilinear upsample (×2)
           → pad to match skip
           → cat([skip, x])       (only when skip is not None)
           → DoubleConv

    Args:
        in_ch   (int): Channels from the deeper feature map.
        skip_ch (int): Channels from the encoder skip connection.
                       Pass 0 for the final decoder block (no skip).
        out_ch  (int): Output channels after DoubleConv.
    """

    def __init__(self, in_ch: int, skip_ch: int, out_ch: int):
        super().__init__()
        self.up   = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
        self.conv = DoubleConv(in_ch + skip_ch, out_ch)

    def forward(
        self,
        x:    torch.Tensor,
        skip: torch.Tensor = None,
    ) -> torch.Tensor:
        x = self.up(x)
        if skip is not None:
            dY = skip.size(2) - x.size(2)
            dX = skip.size(3) - x.size(3)
            x  = F.pad(x, [dX // 2, dX - dX // 2, dY // 2, dY - dY // 2])
            x  = torch.cat([skip, x], dim=1)
        return self.conv(x)
