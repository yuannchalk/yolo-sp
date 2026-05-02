import torch
import torch.nn as nn
import torch.nn.functional as F


class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels, eps=0.001)  # 匹配SP的eps
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels, eps=0.001)

        self.shortcut = nn.Identity() if in_channels == out_channels else \
            nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1, bias=False),
                nn.BatchNorm2d(out_channels, eps=0.001)
            )

    def forward(self, x):
        residual = self.shortcut(x)
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        return F.relu(out + residual)


class ChannelAttention(nn.Module):
    def __init__(self, channels, reduction=16):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        reduced_ch = max(channels // reduction, 8)
        self.mlp = nn.Sequential(
            nn.Conv2d(channels, reduced_ch, 1, bias=False),
            nn.ReLU(),
            nn.Conv2d(reduced_ch, channels, 1, bias=False)
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.mlp(self.avg_pool(x))
        max_out = self.mlp(self.max_pool(x))
        return x * self.sigmoid(avg_out + max_out)


class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super().__init__()
        self.conv = nn.Conv2d(2, 1, kernel_size, padding=kernel_size // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        out = torch.cat([avg_out, max_out], dim=1)
        return x * self.sigmoid(self.conv(out))


class StrongAdapter(nn.Module):
    def __init__(self, in_channels=256, out_channels=128):
        super().__init__()
        # 1. 降维卷积 (256 -> 128)
        self.channel_reduction = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels, eps=0.001),
            nn.ReLU(inplace=True)
        )

        # 2. 特征增强
        self.channel_att = ChannelAttention(out_channels)
        self.spatial_att = SpatialAttention()

        # 3. 深度变换
        self.res_blocks = nn.Sequential(
            ResidualBlock(out_channels, out_channels),
            ResidualBlock(out_channels, out_channels)
        )

    def forward(self, x):
        # x: [B, 256, H, W]
        x = self.channel_reduction(x)  # [B, 128, H, W]
        x = self.channel_att(x)
        x = self.spatial_att(x)
        x = self.res_blocks(x)
        return x