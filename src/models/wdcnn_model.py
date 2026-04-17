from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class PaperG1G2Transform(nn.Module):
    """
    论文 g1 / g2 趋势增强模块（固定卷积核，不参与训练）。

    设计意图：
    - 在可学习卷积前先做一次“方向趋势增强”，让后续卷积更容易捕捉时序变化。
    - g1 关注行方向变化，g2 关注列方向变化。
    """

    def __init__(self) -> None:
        super().__init__()

        # 行方向趋势核：g1(i,j) = 2*x(i,j) - x(i-1,j) - x(i+1,j)
        row_kernel = torch.tensor(
            [[0.0, -1.0, 0.0], [0.0, 2.0, 0.0], [0.0, -1.0, 0.0]],
            dtype=torch.float32,
        ).view(1, 1, 3, 3)

        # 列方向趋势核：g2(i,j) = 2*x(i,j) - x(i,j-1) - x(i,j+1)
        col_kernel = torch.tensor(
            [[0.0, 0.0, 0.0], [-1.0, 2.0, -1.0], [0.0, 0.0, 0.0]],
            dtype=torch.float32,
        ).view(1, 1, 3, 3)

        # register_buffer: 会跟随模型迁移到 GPU/CPU，但不会被优化器更新。
        self.register_buffer("row_kernel", row_kernel, persistent=False)
        self.register_buffer("col_kernel", col_kernel, persistent=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # 固定卷积求方向趋势，再与原输入相加作为增强结果。
        g1 = F.conv2d(x, self.row_kernel, stride=1, padding=1)
        g2 = F.conv2d(x, self.col_kernel, stride=1, padding=1)
        return x + g1 + g2


class WideBranch(nn.Module):
    """
    Wide 分支：面向 1D 日序列输入，提取全局模式特征。
    """

    def __init__(self, input_dim: int, alpha: int, dropout: float) -> None:
        super().__init__()
        self.fc = nn.Linear(input_dim, alpha)
        self.act = nn.ReLU(inplace=True)
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # [B, D] -> [B, alpha]
        return self.drop(self.act(self.fc(x)))


class DeepBranch(nn.Module):
    """
    Deep 分支：面向 2D 周矩阵输入，提取局部时序与周期结构特征。
    """

    def __init__(
        self,
        input_shape: tuple[int, int, int],
        gamma: int,
        r_layers: int,
        beta: int,
        dropout: float,
    ) -> None:
        super().__init__()
        if r_layers < 1:
            raise ValueError("r_layers must be >= 1")

        # 固定趋势增强层（论文风格）
        self.g1g2 = PaperG1G2Transform()

        # 堆叠 R 层卷积，每层输出通道为 gamma。
        layers: list[nn.Module] = []
        in_ch = input_shape[0]
        for _ in range(r_layers):
            layers.extend(
                [
                    nn.Conv2d(in_ch, gamma, kernel_size=3, stride=1, padding=1),
                    nn.ReLU(inplace=True),
                ]
            )
            in_ch = gamma
        self.conv_stack = nn.Sequential(*layers)

        # 池化降采样，减少空间尺寸与参数量。
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        # 用 dummy 前向自动推断 flatten 后维度，避免手动算尺寸出错。
        with torch.no_grad():
            dummy = torch.zeros(1, *input_shape)
            feat = self.pool(self.conv_stack(self.g1g2(dummy)))
            flatten_dim = int(feat.reshape(1, -1).shape[1])

        # Deep 分支输出投影到 beta 维。
        self.fc = nn.Linear(flatten_dim, beta)
        self.act = nn.ReLU(inplace=True)
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # [B, 1, week_count, 7] -> [B, beta]
        x = self.g1g2(x)
        x = self.conv_stack(x)
        x = self.pool(x)
        x = x.reshape(x.shape[0], -1)
        x = self.fc(x)
        x = self.act(x)
        x = self.drop(x)
        return x


class WideDeepCNN(nn.Module):
    """
    论文对齐的 Wide + Deep CNN 端到端二分类模型。

    输入：
    - x_1d: [B, day_count]，给 Wide 分支
    - x_2d: [B, 1, week_count, 7]，给 Deep 分支

    输出：
    - logits: [B]（未过 sigmoid）
    - feature_dict: 中间特征，便于调试与可视化
    """

    def __init__(
        self,
        wide_input_dim: int,
        deep_input_shape: tuple[int, int, int],
        alpha: int,
        beta: int,
        gamma: int,
        r_layers: int,
        dropout: float,
    ) -> None:
        super().__init__()
        self.wide = WideBranch(wide_input_dim, alpha, dropout)
        self.deep = DeepBranch(deep_input_shape, gamma, r_layers, beta, dropout)
        # 融合头：拼接 wide/deep 特征后映射到单一 logit。
        self.head = nn.Linear(alpha + beta, 1)

    def forward(self, x_1d: torch.Tensor, x_2d: torch.Tensor) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        wide_feat = self.wide(x_1d)
        deep_feat = self.deep(x_2d)
        joint_feat = torch.cat([wide_feat, deep_feat], dim=1)
        logits = self.head(joint_feat).squeeze(1)
        return logits, {"wide": wide_feat, "deep": deep_feat, "joint": joint_feat}
