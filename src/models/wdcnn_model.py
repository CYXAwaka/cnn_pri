from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class PaperG1G2Transform(nn.Module):
    """Approximate paper g1/g2 trend kernels before learnable conv layers."""

    def __init__(self) -> None:
        super().__init__()

        # Row trend: 2*x(i,j) - x(i-1,j) - x(i+1,j)
        row_kernel = torch.tensor(
            [[0.0, -1.0, 0.0], [0.0, 2.0, 0.0], [0.0, -1.0, 0.0]],
            dtype=torch.float32,
        ).view(1, 1, 3, 3)

        # Column trend: 2*x(i,j) - x(i,j-1) - x(i,j+1)
        col_kernel = torch.tensor(
            [[0.0, 0.0, 0.0], [-1.0, 2.0, -1.0], [0.0, 0.0, 0.0]],
            dtype=torch.float32,
        ).view(1, 1, 3, 3)

        self.register_buffer("row_kernel", row_kernel, persistent=False)
        self.register_buffer("col_kernel", col_kernel, persistent=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        g1 = F.conv2d(x, self.row_kernel, stride=1, padding=1)
        g2 = F.conv2d(x, self.col_kernel, stride=1, padding=1)
        return x + g1 + g2


class WideBranch(nn.Module):
    def __init__(self, input_dim: int, alpha: int, dropout: float) -> None:
        super().__init__()
        self.fc = nn.Linear(input_dim, alpha)
        self.act = nn.ReLU(inplace=True)
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.drop(self.act(self.fc(x)))


class DeepBranch(nn.Module):
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

        self.g1g2 = PaperG1G2Transform()

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
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        with torch.no_grad():
            dummy = torch.zeros(1, *input_shape)
            feat = self.pool(self.conv_stack(self.g1g2(dummy)))
            flatten_dim = int(feat.reshape(1, -1).shape[1])

        self.fc = nn.Linear(flatten_dim, beta)
        self.act = nn.ReLU(inplace=True)
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.g1g2(x)
        x = self.conv_stack(x)
        x = self.pool(x)
        x = x.reshape(x.shape[0], -1)
        x = self.fc(x)
        x = self.act(x)
        x = self.drop(x)
        return x


class WideDeepCNN(nn.Module):
    """Paper-aligned end-to-end Wide + Deep CNN binary classifier."""

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
        self.head = nn.Linear(alpha + beta, 1)

    def forward(self, x_1d: torch.Tensor, x_2d: torch.Tensor) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        wide_feat = self.wide(x_1d)
        deep_feat = self.deep(x_2d)
        joint_feat = torch.cat([wide_feat, deep_feat], dim=1)
        logits = self.head(joint_feat).squeeze(1)
        return logits, {"wide": wide_feat, "deep": deep_feat, "joint": joint_feat}
