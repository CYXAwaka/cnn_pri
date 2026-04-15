from __future__ import annotations

from dataclasses import dataclass, field
from itertools import product
from pathlib import Path
from typing import Any

import torch


@dataclass
class Config:
    """Global experiment config for paper-aligned WDCNN."""

    # Paths
    data_path: str = "data/raw/data.csv"
    checkpoint_dir: str = "checkpoints"
    result_dir: str = "results"

    # Dataset columns
    id_col: str = "CONS_NO"
    label_col: str = "FLAG"

    # Preprocess (paper-aligned)
    days_per_week: int = 7
    fill_missing_calendar_days: bool = True
    use_outlier_clip: bool = True
    outlier_sigma_k: float = 2.0
    normalize_eps: float = 1e-8
    week_pad_value: float = 0.0

    # Split and search protocol
    train_ratios: list[float] = field(default_factory=lambda: [0.5, 0.6, 0.7, 0.8])
    val_ratio_in_train: float = 0.1
    seed_list: list[int] = field(default_factory=lambda: [42, 52, 62])

    # WDCNN default params from paper's main setting
    alpha: int = 90  # wide branch FC width
    beta: int = 60   # deep branch FC width
    gamma: int = 15  # deep branch conv channels
    r_layers: int = 5
    dropout: float = 0.2

    # Coarse search space
    alpha_grid: list[int] = field(default_factory=lambda: [50, 60, 90])
    beta_grid: list[int] = field(default_factory=lambda: [60, 90, 120])
    gamma_grid: list[int] = field(default_factory=lambda: [10, 15, 20])
    r_grid: list[int] = field(default_factory=lambda: [3, 4, 5])
    refine_top_k: int = 3

    # Optimization
    batch_size: int = 128
    coarse_max_epochs: int = 20
    fine_max_epochs: int = 60
    early_stop_patience: int = 12
    lr: float = 1e-3
    weight_decay: float = 1e-4
    lr_scheduler_factor: float = 0.5
    lr_scheduler_patience: int = 4
    min_lr: float = 1e-6

    # Selection and threshold strategy
    selection_metric: str = "auc"
    threshold_metric: str = "f1"

    # Runtime and artifact settings
    num_workers: int = 0
    artifact_versioning: str = "timestamp+version"
    device: str = "cuda" if torch.cuda.is_available() else "cpu"

    def make_dirs(self) -> None:
        Path(self.checkpoint_dir).mkdir(parents=True, exist_ok=True)
        Path(self.result_dir).mkdir(parents=True, exist_ok=True)

    def coarse_param_grid(self) -> list[dict[str, Any]]:
        """Cartesian product over alpha/beta/gamma/R for stage-1 search."""
        grid: list[dict[str, Any]] = []
        for alpha, beta, gamma, r_layers in product(
            self.alpha_grid,
            self.beta_grid,
            self.gamma_grid,
            self.r_grid,
        ):
            grid.append(
                {
                    "alpha": alpha,
                    "beta": beta,
                    "gamma": gamma,
                    "r_layers": r_layers,
                    "dropout": self.dropout,
                }
            )
        return grid


config = Config()
