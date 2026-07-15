"""
densenet_unet.py
----------------
DenseNet U-Net for underwater image restoration.

Replaces the ``DoubleConv`` blocks in the standard U-Net with **Dense Blocks**
that use dense connectivity — each layer receives the concatenated feature
maps of *all* preceding layers in the block.  The overall 4-level
encoder-decoder topology, skip connections, and output head match
:class:`UNet5ch` so that training/evaluation scripts work unchanged.

Architecture
------------
                 Input (B, C_in, H, W)
                        │
      DenseBlock  64  ──┐  enc1
      DenseDown  128  ──┤  enc2  (MaxPool → DenseBlock)
      DenseDown  256  ──┤  enc3
      DenseDown  512  ──┤  enc4
   DenseDown     1024    bottleneck
      DenseUp   512  ──┘  dec4  (Upsample + skip)
      DenseUp   256  ──┘  dec3
      DenseUp   128  ──┘  dec2
      DenseUp    64  ──┘  dec1
               head   Conv1×1 → Sigmoid
                        │
                Output (B, 3, H, W) in [0, 1]

Dense Block internals
---------------------
Each DenseBlock contains ``num_layers`` DenseLayers.  A DenseLayer uses a
bottleneck design (BN-ReLU-Conv1×1-BN-ReLU-Conv3×3) and produces
``growth_rate`` new feature channels.  After all layers, a 1×1 projection
convolution maps the accumulated channels to the desired output width.

Hyperparameters
---------------
  growth_rate : int  —  new channels per dense layer (default 32)
  num_layers  : int  —  dense layers per block (default 4)
  bn_size     : int  —  bottleneck expansion in 1×1 conv (default 4)

Example::

    model = DenseNetUNet(in_channels=5)
    x = torch.randn(2, 5, 256, 256)
    y = model(x)          # (2, 3, 256, 256)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as cp


# ---------------------------------------------------------------------------
# Building blocks
# ---------------------------------------------------------------------------


class DenseLayer(nn.Module):
    """
    Single dense layer with bottleneck design.

    Produces ``growth_rate`` new feature channels via:
        BN → ReLU → Conv1×1 (expand to ``bn_size * growth_rate``)
        → BN → ReLU → Conv3×3 (produce ``growth_rate`` channels)

    Args:
        in_ch       (int): Total input channels (accumulated from previous layers).
        growth_rate (int): Number of new channels this layer produces.
        bn_size     (int): Bottleneck expansion factor. Default: 4.
        dropout     (float): Dropout probability after Conv3×3. Default: 0.
    """

    def __init__(
        self,
        in_ch: int,
        growth_rate: int,
        bn_size: int = 4,
        dropout: float = 0.0,
    ):
        super().__init__()
        inter_ch = bn_size * growth_rate
        self.net = nn.Sequential(
            nn.BatchNorm2d(in_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_ch, inter_ch, kernel_size=1, bias=False),
            nn.BatchNorm2d(inter_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(inter_ch, growth_rate, kernel_size=3, padding=1, bias=False),
        )
        self.dropout = nn.Dropout2d(dropout) if dropout > 0.0 else None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Return only the *new* features (``growth_rate`` channels)."""
        out = self.net(x)
        if self.dropout is not None:
            out = self.dropout(out)
        return out


class DenseBlock(nn.Module):
    """
    Dense connectivity block — replacement for ``DoubleConv``.

    Contains ``num_layers`` DenseLayers.  Each layer receives the concatenation
    of ALL preceding feature maps (input + outputs of layers 0..i−1).  After
    the last layer a 1×1 projection convolution maps the accumulated channels
    to ``out_ch``.

    Args:
        in_ch       (int): Input channels.
        out_ch      (int): Output channels (after projection).
        num_layers  (int): Number of dense layers. Default: 4.
        growth_rate (int): Channels produced per layer. Default: 32.
        bn_size     (int): Bottleneck expansion factor. Default: 4.
        dropout     (float): Dropout probability inside each layer. Default: 0.
    """

    def __init__(
        self,
        in_ch: int,
        out_ch: int,
        num_layers: int = 4,
        growth_rate: int = 32,
        bn_size: int = 4,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.layers = nn.ModuleList()
        ch = in_ch
        for _ in range(num_layers):
            self.layers.append(DenseLayer(ch, growth_rate, bn_size, dropout))
            ch += growth_rate

        # Project accumulated channels → out_ch
        self.project = nn.Sequential(
            nn.BatchNorm2d(ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(ch, out_ch, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        def _forward(x_in):
            features = [x_in]
            for layer in self.layers:
                cat = torch.cat(features, dim=1)
                new_feat = layer(cat)
                features.append(new_feat)
            return self.project(torch.cat(features, dim=1))

        if self.training and x.requires_grad:
            return cp.checkpoint(_forward, x, use_reentrant=False)
        return _forward(x)



class DenseDown(nn.Module):
    """MaxPool2d(2) followed by a DenseBlock — replaces ``Down``."""

    def __init__(
        self,
        in_ch: int,
        out_ch: int,
        num_layers: int = 4,
        growth_rate: int = 32,
        bn_size: int = 4,
    ):
        super().__init__()
        self.pool = nn.MaxPool2d(2)
        self.dense = DenseBlock(in_ch, out_ch, num_layers, growth_rate, bn_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.dense(self.pool(x))


class DenseUp(nn.Module):
    """
    Decoder step: upsample ``x1``, pad to match skip ``x2``, concatenate,
    and apply a DenseBlock — replaces ``Up``.

    Args:
        prev_ch     (int): Channels from the deeper level.
        skip_ch     (int): Channels from the matching encoder skip connection.
        out_ch      (int): Output channels after DenseBlock.
        num_layers  (int): Dense layers in the block. Default: 4.
        growth_rate (int): Channels per dense layer. Default: 32.
        bn_size     (int): Bottleneck expansion. Default: 4.
        bilinear    (bool): Bilinear upsampling vs ConvTranspose2d. Default: True.
    """

    def __init__(
        self,
        prev_ch: int,
        skip_ch: int,
        out_ch: int,
        num_layers: int = 4,
        growth_rate: int = 32,
        bn_size: int = 4,
        bilinear: bool = True,
    ):
        super().__init__()
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
            dense_in_ch = prev_ch + skip_ch
        else:
            self.up = nn.ConvTranspose2d(prev_ch, prev_ch // 2, kernel_size=2, stride=2)
            dense_in_ch = (prev_ch // 2) + skip_ch
        self.dense = DenseBlock(dense_in_ch, out_ch, num_layers, growth_rate, bn_size)

    def forward(self, x1: torch.Tensor, x2: torch.Tensor) -> torch.Tensor:
        x1 = self.up(x1)
        # Pad x1 to match x2 when spatial dims are odd
        dY = x2.size(2) - x1.size(2)
        dX = x2.size(3) - x1.size(3)
        x1 = F.pad(x1, [dX // 2, dX - dX // 2, dY // 2, dY - dY // 2])
        return self.dense(torch.cat([x2, x1], dim=1))


# ---------------------------------------------------------------------------
# DenseNet U-Net
# ---------------------------------------------------------------------------


class DenseNetUNet(nn.Module):
    """
    DenseNet U-Net for underwater image restoration.

    Accepts either 3-channel (RGB) or 5-channel (RGB + physics) input via
    ``in_channels``.  Output is always 3-channel RGB in [0, 1] (Sigmoid head).

    This model replaces every ``DoubleConv`` block in :class:`UNet5ch` with a
    :class:`DenseBlock`, keeping the 4-level encoder-decoder topology, skip
    connections, and channel widths identical so that the rest of the pipeline
    (physics front-end, loss, metrics, checkpointing) works unchanged.

    Args:
        in_channels  (int):         Input channels — 3 for RGB, 5 for physics-guided.
        out_channels (int):         Output channels. Default: 3.
        features     (tuple[int]):  Encoder feature widths per level.
                                    Default: (64, 128, 256, 512).
        num_layers   (int):         Dense layers per DenseBlock. Default: 4.
        growth_rate  (int):         New channels per dense layer. Default: 32.
        bn_size      (int):         Bottleneck expansion factor. Default: 4.
        bilinear     (bool):        Bilinear upsampling vs ConvTranspose2d.
                                    Default: True.

    Example::

        model = DenseNetUNet(in_channels=5)
        x = torch.randn(2, 5, 256, 256)
        y = model(x)          # (2, 3, 256, 256)
    """

    def __init__(
        self,
        in_channels: int = 5,
        out_channels: int = 3,
        features: tuple[int, ...] = (64, 128, 256, 512),
        num_layers: int = 4,
        growth_rate: int = 16,
        bn_size: int = 4,
        bilinear: bool = True,
    ):
        super().__init__()
        f = features
        nl = num_layers
        gr = growth_rate
        bs = bn_size

        # Encoder
        self.enc1 = DenseBlock(in_channels, f[0], nl, gr, bs)
        self.enc2 = DenseDown(f[0], f[1], nl, gr, bs)
        self.enc3 = DenseDown(f[1], f[2], nl, gr, bs)
        self.enc4 = DenseDown(f[2], f[3], nl, gr, bs)

        # Bottleneck (f[3] × 2 = 1024 for default features)
        self.bottleneck = DenseDown(f[3], f[3] * 2, nl, gr, bs)

        # Decoder — channels: (from_below, skip, out)
        self.dec4 = DenseUp(f[3] * 2, f[3], f[3], nl, gr, bs, bilinear)
        self.dec3 = DenseUp(f[3], f[2], f[2], nl, gr, bs, bilinear)
        self.dec2 = DenseUp(f[2], f[1], f[1], nl, gr, bs, bilinear)
        self.dec1 = DenseUp(f[1], f[0], f[0], nl, gr, bs, bilinear)

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
        e1 = self.enc1(x)              # (N,  64, H,   W  )
        e2 = self.enc2(e1)             # (N, 128, H/2, W/2)
        e3 = self.enc3(e2)             # (N, 256, H/4, W/4)
        e4 = self.enc4(e3)             # (N, 512, H/8, W/8)
        bn = self.bottleneck(e4)       # (N,1024, H/16,W/16)

        d4 = self.dec4(bn, e4)         # (N, 512, H/8, W/8)
        d3 = self.dec3(d4, e3)         # (N, 256, H/4, W/4)
        d2 = self.dec2(d3, e2)         # (N, 128, H/2, W/2)
        d1 = self.dec1(d2, e1)         # (N,  64, H,   W  )

        return self.head(d1)           # (N,   3, H,   W  )
