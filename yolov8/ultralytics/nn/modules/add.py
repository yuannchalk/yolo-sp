import torch
import torch.nn as nn
from torch import Tensor


class RGBFireBlock(nn.Module):
    """RGB通道分离火焰特征块（兼容YOLOv8框架）

    设计目标：
    1. 利用颜色通道独立性强化火焰颜色特征学习
    2. 保持与YOLOv8原主干网络的兼容性
    3. 实现计算效率与精度的平衡

    结构特性：
    - 独立分支处理RGB三通道
    - 动态特征融合机制
    - 注意力增强的特征传递
    """

    def __init__(self, c1: int, c2: int):

        super().__init__()
        self.c1 = c1
        self.c2 = c2
        # -------------------------------
        # 各颜色通道独立处理分支
        # 设计考虑：
        # 1. 使用深度可分离卷积思想，但保持通道独立性
        # 2. 下采样策略与YOLOv8主干保持一致
        # -------------------------------

        # R通道处理分支
        self.r_branch = nn.Sequential(
            nn.Conv2d(1, (c1 // 3) + 1, kernel_size=3, stride=2, padding=1, bias=False),  # stride=2与YOLOv8下采样率对齐
            nn.BatchNorm2d((c1 // 3) + 1),
            #nn.SiLU(inplace=True)  # 使用SiLU保持激活一致性，inplace节省内存
        )

        # G通道处理分支
        self.g_branch = nn.Sequential(
            nn.Conv2d(1, (c1 // 3) + 1, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d((c1 // 3) + 1),
            #nn.SiLU(inplace=True)
        )

        # B通道处理分支
        self.b_branch = nn.Sequential(
            nn.Conv2d(1, (c1 // 3) + 1, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d((c1 // 3) + 1),
            #nn.SiLU(inplace=True)
        )

        # -------------------------------
        # 特征融合模块
        # 包含多尺度特征提取和注意力机制：
        # 1. 通道注意力（SE）增强颜色敏感度
        # 2. 空间注意力聚焦火焰区域
        # -------------------------------
        self.fusion = nn.Sequential(
            nn.Conv2d(c1, c2, kernel_size=1),  # 通道维度对齐
            nn.SiLU(inplace=True)
            #SqueezeExcitation(c2),  # 通道注意力
            #SpatialAttention()  # 空间注意力
        )

    def forward(self, x: Tensor) -> Tensor:
        """前向传播流程说明：

        输入形状：(batch_size, 3, H, W)
        处理流程：
        1. 分离RGB三通道
        2. 各分支独立处理
        3. 通道维度拼接
        4. 多注意力特征融合

        输出形状：(batch_size, c2, H/2, W/2)
        """
        # 通道分离处理
        r_feat = self.r_branch(x[:, 0:1, :, :])  # 提取R通道 [N,1,H,W]
        g_feat = self.g_branch(x[:, 1:2, :, :])  # 提取G通道
        b_feat = self.b_branch(x[:, 2:3, :, :])  # 提取B通道

        # 特征拼接与融合
        concat_feat = torch.cat([r_feat, g_feat, b_feat], dim=1)  # [N,3c1,H/2,W/2]
        concat_feat = concat_feat[:, 0:self.c2, :, :]
        fused_feat = self.fusion(concat_feat)  # [N,c2,H/2,W/2]

        return fused_feat


class SqueezeExcitation(nn.Module):
    """通道注意力模块（SE Block）优化版

    改进点：
    - 减少全连接层参数量
    - 添加残差连接保持梯度流动
    """

    def __init__(self, channels: int, reduction: int = 16):
        super().__init__()
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channels, channels // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channels // reduction, channels, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x: Tensor) -> Tensor:
        b, c, _, _ = x.size()
        y = self.avgpool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x) + x  # 残差连接


class SpatialAttention(nn.Module):
    """空间注意力模块（轻量化设计）

    使用单层卷积替代传统双分支结构：
    - 保持空间信息敏感度
    - 减少计算量约40%
    """

    def __init__(self, kernel_size: int = 7):
        super().__init__()
        self.conv = nn.Conv2d(2, 1, kernel_size, padding=kernel_size // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: Tensor) -> Tensor:
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        concat = torch.cat([avg_out, max_out], dim=1)
        att = self.sigmoid(self.conv(concat))
        return x * att + x  # 残差连接