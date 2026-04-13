
"""
src/models/cnn_lg_model.py
~~~~~~~~~~~~~~~~~~~~~~~~~~
这里放的是模型结构定义。

注意：
CNN-LG 并不是“一个端到端一起反向传播”的单体模型。
更准确地说，它分两步：
1. 先用 CNN 学到一个好的特征提取器
2. 再把 CNN 提取到的特征交给 LightGBM 做最终分类

所以这里会定义两个 PyTorch 模块：
- CNNFeatureExtractor：只负责提特征
- CNNPretrainClassifier：训练 CNN 时临时加上的分类头
"""

from __future__ import annotations

import torch
import torch.nn as nn


class CNNFeatureExtractor(nn.Module):
    """
    卷积特征提取器。

    结构尽量贴近论文：
        输入 [B, 1, 147, 7]
        -> Conv(3x3, 16)
        -> ReLU
        -> MaxPool(3x3, stride=3)
        -> Conv(3x3, 16)
        -> ReLU
        -> MaxPool(2x2, stride=2)
        -> Flatten
        -> FC(64)

    这里输出的 64 维向量，就是后面 LightGBM 要吃的特征。
    """

    def __init__(
        self,
        input_shape,
        in_channels: int = 1,
        conv_channels: int = 16,
        kernel_size: int = 3,
        pool1_kernel_size: int = 3,
        pool1_stride: int = 3,
        pool2_kernel_size: int = 2,
        pool2_stride: int = 2,
        fc_dim: int = 64,
        dropout: float = 0.2,
    ):
        super().__init__()

        # 第 1 个卷积块
        self.conv1 = nn.Conv2d(
            in_channels=in_channels,
            out_channels=conv_channels,
            kernel_size=kernel_size,
            stride=1,
            padding=1,  # 保持空间尺寸尽量不变
        )
        self.relu1 = nn.ReLU(inplace=True)
        self.pool1 = nn.MaxPool2d(kernel_size=pool1_kernel_size, stride=pool1_stride)

        # 第 2 个卷积块
        self.conv2 = nn.Conv2d(
            in_channels=conv_channels,
            out_channels=conv_channels,
            kernel_size=kernel_size,
            stride=1,
            padding=1,
        )
        self.relu2 = nn.ReLU(inplace=True)
        self.pool2 = nn.MaxPool2d(kernel_size=pool2_kernel_size, stride=pool2_stride)

        self.dropout = nn.Dropout(dropout)

        # 这里动态计算卷积输出展平后的维度
        # 这样你以后即使改输入尺寸，也不用手算 Linear 的输入维度
        with torch.no_grad():
            dummy = torch.zeros(1, *input_shape)
            feature_map = self._forward_conv(dummy)
            flatten_dim = feature_map.view(1, -1).shape[1]

        self.flatten_dim = flatten_dim
        self.fc = nn.Linear(flatten_dim, fc_dim)

    def _forward_conv(self, x: torch.Tensor) -> torch.Tensor:
        """
        只跑卷积和池化，不做最后的 FC。
        这个函数是内部辅助函数，主要用于：
        1. 动态计算 flatten 维度
        2. 保持 forward 更清晰
        """
        x = self.conv1(x)
        x = self.relu1(x)
        x = self.pool1(x)

        x = self.conv2(x)
        x = self.relu2(x)
        x = self.pool2(x)
        return x

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        输出 64 维特征。
        """
        x = self._forward_conv(x)
        x = x.view(x.size(0), -1)
        x = self.dropout(x)
        x = self.fc(x)
        return x


class CNNPretrainClassifier(nn.Module):
    """
    CNN 预训练分类器。

    为什么需要这个类？
    因为 LightGBM 不能直接参与 PyTorch 的梯度反向传播，
    所以训练 CNN 特征提取器时，我们先临时接一个线性分类头，
    用监督学习把 CNN 训练好。

    训练完后：
    - 保留 feature_extractor
    - 丢掉 classifier_head
    - 用 feature_extractor 提 64 维特征给 LightGBM
    """

    def __init__(self, feature_extractor: CNNFeatureExtractor, num_classes: int = 2):
        super().__init__()
        self.feature_extractor = feature_extractor
        self.classifier_head = nn.Linear(feature_extractor.fc.out_features, num_classes)

    def forward(self, x: torch.Tensor):
        features = self.feature_extractor(x)
        logits = self.classifier_head(features)
        return logits, features
