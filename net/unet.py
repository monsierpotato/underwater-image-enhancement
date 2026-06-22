"""
unet.py
-------
Dense U-Net architecture for underwater image restoration.
Customized with DenseNet blocks (DenseLayer, DenseConvBlock).

The same class handles both the RGB baseline and the physics-guided
variant; the only difference is ``in_channels``:
    in_channels = 3  →  RGB baseline
    in_channels = 5  →  RGB + t(x) + B
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

# ---------------------------------------------------------------------------
# Building blocks (DenseNet components)
# ---------------------------------------------------------------------------

class DenseLayer(nn.Module):
    """
    Một layer cơ bản của DenseNet gồm Bottleneck (1x1 Conv) để giảm params
    và Composite layer (3x3 Conv) để trích xuất đặc trưng.
    """
    def __init__(self, in_channels: int, growth_rate: int, bn_size: int = 4):
        super().__init__()
        # Bottleneck layer (1x1 conv)
        self.bottleneck = nn.Sequential(
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels, bn_size * growth_rate, kernel_size=1, bias=False)
        )
        # Composite layer (3x3 conv)
        self.conv = nn.Sequential(
            nn.BatchNorm2d(bn_size * growth_rate),
            nn.ReLU(inplace=True),
            nn.Conv2d(bn_size * growth_rate, growth_rate, kernel_size=3, padding=1, bias=False)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        new_features = self.conv(self.bottleneck(x))
        # Nối feature map mới với input ban đầu theo chiều channels
        return torch.cat([x, new_features], dim=1)


class DenseConvBlock(nn.Module):
    """
    Module thay thế cho DoubleConv truyền thống. 
    Chứa nhiều DenseLayer, cuối cùng đi qua một Transition Layer 
    để ép số lượng channels về đúng out_channels yêu cầu của U-Net.
    """
    def __init__(self, in_channels: int, out_channels: int, num_layers: int = 4, growth_rate: int = 32, bn_size: int = 4):
        super().__init__()
        layers = []
        current_channels = in_channels
        
        # Xây dựng các Dense Layer
        for i in range(num_layers):
            layers.append(DenseLayer(current_channels, growth_rate, bn_size))
            current_channels += growth_rate
        
        self.dense_block = nn.Sequential(*layers)
        
        # Transition layer để chuyển đổi chiều sâu feature map về out_channels
        self.transition = nn.Sequential(
            nn.BatchNorm2d(current_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(current_channels, out_channels, kernel_size=1, bias=False)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.dense_block(x)
        return self.transition(x)


class Down(nn.Module):
    """MaxPool2d(2) followed by DenseConvBlock."""
    def __init__(self, in_ch: int, out_ch: int, num_layers: int = 4, growth_rate: int = 32):
        super().__init__()
        self.net = nn.Sequential(
            nn.MaxPool2d(2),
            DenseConvBlock(in_ch, out_ch, num_layers, growth_rate)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class Up(nn.Module):
    """
    Decoder step: upsample ``x1``, pad to match skip ``x2``, concatenate, followed by DenseConvBlock.
    """
    def __init__(self, prev_ch: int, skip_ch: int, out_ch: int, bilinear: bool = True, num_layers: int = 4, growth_rate: int = 32):
        super().__init__()
        if bilinear:
            self.up   = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
            self.conv = DenseConvBlock(prev_ch + skip_ch, out_ch, num_layers, growth_rate)
        else:
            self.up   = nn.ConvTranspose2d(prev_ch, prev_ch // 2, kernel_size=2, stride=2)
            self.conv = DenseConvBlock((prev_ch // 2) + skip_ch, out_ch, num_layers, growth_rate)

    def forward(self, x1: torch.Tensor, x2: torch.Tensor) -> torch.Tensor:
        x1 = self.up(x1)
        # Pad x1 to match x2 when spatial dims are odd
        dY = x2.size(2) - x1.size(2)
        dX = x2.size(3) - x1.size(3)
        x1 = F.pad(x1, [dX // 2, dX - dX // 2, dY // 2, dY - dY // 2])
        return self.conv(torch.cat([x2, x1], dim=1))


# ---------------------------------------------------------------------------
# Dense U-Net Main Architecture
# ---------------------------------------------------------------------------

class UNet5ch(nn.Module):
    """
    Dense U-Net for underwater image restoration.
    
    Lưu ý: Tên class vẫn được giữ là `UNet5ch` để tương thích với `registry.py`,
    nhưng kiến trúc bên trong là DenseUNet.

    Args:
        in_channels  (int):          Input channels (3 or 5).
        out_channels (int):          Output channels. Default: 3.
        features     (tuple[int]):   Feature map sizes at each encoder level.
        bilinear     (bool):         Bilinear upsampling vs ConvTranspose2d.
    """
    def __init__(
        self,
        in_channels:  int          = 5,
        out_channels: int          = 3,
        features:     tuple[int, ...] = (64, 128, 256, 512),
        bilinear:     bool         = True,
    ):
        super().__init__()
        f = features
        
        # HYPERPARAMETERS CHO DENSENET:
        n_l = 2   # Số lượng DenseLayer trong mỗi Block
        gr  = 8   # Growth rate (tốc độ phình kênh ra sau mỗi layer)

        # Encoder
        self.enc1 = DenseConvBlock(in_channels, f[0], n_l, gr)
        self.enc2 = Down(f[0], f[1], n_l, gr)
        self.enc3 = Down(f[1], f[2], n_l, gr)
        self.enc4 = Down(f[2], f[3], n_l, gr)
        
        # Bottleneck (f[3] * 2 = 1024)
        self.bottleneck = Down(f[3], f[3] * 2, n_l, gr)
        
        # Decoder
        self.dec4 = Up(f[3] * 2, f[3], f[3], bilinear, n_l, gr)
        self.dec3 = Up(f[3],     f[2], f[2], bilinear, n_l, gr)
        self.dec2 = Up(f[2],     f[1], f[1], bilinear, n_l, gr)
        self.dec1 = Up(f[1],     f[0], f[0], bilinear, n_l, gr)
        
        # Output head: 1x1 conv + Sigmoid -> [0, 1]
        self.head = nn.Sequential(
            nn.Conv2d(f[0], out_channels, kernel_size=1),
            nn.Sigmoid()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x (Tensor): (N, in_channels, H, W)
        Returns:
            Tensor: (N, 3, H, W) restored RGB image in [0, 1].
        """
        e1 = self.enc1(x)           # (N,  64, H,   W  )
        e2 = self.enc2(e1)          # (N, 128, H/2, W/2)
        e3 = self.enc3(e2)          # (N, 256, H/4, W/4)
        e4 = self.enc4(e3)          # (N, 512, H/8, W/8)
        bn = self.bottleneck(e4)    # (N,1024, H/16,W/16)

        d4 = self.dec4(bn, e4)      # (N, 512, H/8, W/8)
        d3 = self.dec3(d4, e3)      # (N, 256, H/4, W/4)
        d2 = self.dec2(d3, e2)      # (N, 128, H/2, W/2)
        d1 = self.dec1(d2, e1)      # (N,  64, H,   W  )

        return self.head(d1)        # (N,   3, H,   W  )
