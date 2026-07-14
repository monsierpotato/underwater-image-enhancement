"""U-Net variants with lightweight ASPP and Mamba bottleneck context.

The encoder, decoder, and 1024-channel bottleneck match :class:`UNet5ch`.
Context modules are residual branches, making the ablation answer a narrow
question: does multi-scale or selective-scan context improve restoration?
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from .mamba_unet import VSSStage
from .unet import DoubleConv, Down, Up


def _group_count(channels: int) -> int:
    """Choose a GroupNorm group count that divides ``channels``."""
    for groups in (32, 16, 8, 4, 2):
        if channels % groups == 0:
            return groups
    return 1


class _ConvNormAct(nn.Sequential):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        *,
        kernel_size: int,
        padding: int = 0,
        dilation: int = 1,
    ):
        super().__init__(
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=kernel_size,
                padding=padding,
                dilation=dilation,
                bias=False,
            ),
            nn.GroupNorm(_group_count(out_channels), out_channels),
            nn.GELU(),
        )


class ResidualASPP(nn.Module):
    """DeepLab-style multi-scale context with a shape-preserving residual."""

    def __init__(
        self,
        channels: int,
        branch_channels: int = 256,
        dilations: tuple[int, ...] = (1, 3, 6),
        layer_scale: float = 1e-3,
    ):
        super().__init__()
        self.local = _ConvNormAct(channels, branch_channels, kernel_size=1)
        self.dilated = nn.ModuleList(
            [
                _ConvNormAct(
                    channels,
                    branch_channels,
                    kernel_size=3,
                    padding=dilation,
                    dilation=dilation,
                )
                for dilation in dilations
            ]
        )
        self.image_pool = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            _ConvNormAct(channels, branch_channels, kernel_size=1),
        )
        n_branches = 2 + len(dilations)
        self.project = nn.Sequential(
            nn.Conv2d(branch_channels * n_branches, channels, kernel_size=1, bias=False),
            nn.GroupNorm(_group_count(channels), channels),
        )
        self.scale = nn.Parameter(torch.full((channels,), layer_scale))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        size = x.shape[-2:]
        pooled = F.interpolate(
            self.image_pool(x), size=size, mode="bilinear", align_corners=False
        )
        features = [self.local(x), *(branch(x) for branch in self.dilated), pooled]
        context = self.project(torch.cat(features, dim=1))
        return x + context * self.scale.view(1, -1, 1, 1)


class ResidualMambaContext(nn.Module):
    """Run VSS blocks at H/16 after projecting the wide U-Net bottleneck."""

    def __init__(
        self,
        channels: int,
        hidden_channels: int = 256,
        depth: int = 2,
        d_state: int = 16,
        layer_scale: float = 1e-3,
    ):
        super().__init__()
        self.in_proj = nn.Sequential(
            nn.Conv2d(channels, hidden_channels, kernel_size=1, bias=False),
            nn.GroupNorm(_group_count(hidden_channels), hidden_channels),
            nn.GELU(),
        )
        self.vss = VSSStage(hidden_channels, depth, d_state=d_state, use_checkpoint=True)
        self.out_proj = nn.Sequential(
            nn.Conv2d(hidden_channels, channels, kernel_size=1, bias=False),
            nn.GroupNorm(_group_count(channels), channels),
        )
        self.scale = nn.Parameter(torch.full((channels,), layer_scale))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        context = self.in_proj(x).permute(0, 2, 3, 1).contiguous()
        context = self.vss(context)
        context = self.out_proj(context.permute(0, 3, 1, 2).contiguous())
        return x + context * self.scale.view(1, -1, 1, 1)


class ContextUNet(nn.Module):
    """Standard restoration U-Net with configurable bottleneck context."""

    def __init__(
        self,
        in_channels: int = 5,
        out_channels: int = 3,
        features: tuple[int, ...] = (64, 128, 256, 512),
        bilinear: bool = True,
        context: str = "aspp",
    ):
        super().__init__()
        if context not in {"aspp", "mamba", "mamba_aspp"}:
            raise ValueError(f"Unknown context mode: {context}")

        f = features
        bottleneck_channels = f[3] * 2
        self.enc1 = DoubleConv(in_channels, f[0])
        self.enc2 = Down(f[0], f[1])
        self.enc3 = Down(f[1], f[2])
        self.enc4 = Down(f[2], f[3])
        self.bottleneck = Down(f[3], bottleneck_channels)

        context_layers: list[nn.Module] = []
        if context in {"aspp", "mamba_aspp"}:
            context_layers.append(ResidualASPP(bottleneck_channels))
        if context in {"mamba", "mamba_aspp"}:
            context_layers.append(ResidualMambaContext(bottleneck_channels))
        self.context = nn.Sequential(*context_layers)

        self.dec4 = Up(bottleneck_channels, f[3], f[3], bilinear)
        self.dec3 = Up(f[3], f[2], f[2], bilinear)
        self.dec2 = Up(f[2], f[1], f[1], bilinear)
        self.dec1 = Up(f[1], f[0], f[0], bilinear)
        self.head = nn.Sequential(nn.Conv2d(f[0], out_channels, kernel_size=1), nn.Sigmoid())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        e1 = self.enc1(x)
        e2 = self.enc2(e1)
        e3 = self.enc3(e2)
        e4 = self.enc4(e3)
        bn = self.context(self.bottleneck(e4))
        d4 = self.dec4(bn, e4)
        d3 = self.dec3(d4, e3)
        d2 = self.dec2(d3, e2)
        d1 = self.dec1(d2, e1)
        return self.head(d1)


class ASPPUNet(ContextUNet):
    def __init__(self, **kwargs):
        super().__init__(context="aspp", **kwargs)


class MambaBottleneckUNet(ContextUNet):
    def __init__(self, **kwargs):
        super().__init__(context="mamba", **kwargs)


class MambaASPPUNet(ContextUNet):
    def __init__(self, **kwargs):
        super().__init__(context="mamba_aspp", **kwargs)
