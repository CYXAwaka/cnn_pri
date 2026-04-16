from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import torch


@dataclass
class Config:
    """Single-run reproduction config for WDCNN."""

    # Paths
    data_path: str = "data/raw/data.csv"
    checkpoint_dir: str = "checkpoints/repro_stageA001"
    result_dir: str = "results/repro_stageA001"

    # Dataset columns
    id_col: str = "CONS_NO"
    label_col: str = "FLAG"

    # Preprocessing
    days_per_week: int = 7
    fill_missing_calendar_days: bool = True
    use_outlier_clip: bool = True
    outlier_sigma_k: float = 2.0
    normalize_eps: float = 1e-8
    week_pad_value: float = 0.0

    # Split and seed
    train_ratio: float = 0.8
    val_ratio_in_train: float = 0.1
    seed: int = 42

    # Model hyperparameters
    alpha: int = 90
    beta: int = 120
    gamma: int = 20
    r_layers: int = 3
    dropout: float = 0.2

    # Training hyperparameters
    batch_size: int = 128
    lr: float = 4e-4
    weight_decay: float = 1e-4
    train_epochs: int = 30
    scheduler_total_epochs: int = 30
    warmup_epochs: int = 1
    use_cosine_schedule: bool = True
    early_stop_patience: int = 0  # <=0 means disabled
    lr_scheduler_factor: float = 0.5
    lr_scheduler_patience: int = 4
    min_lr: float = 1e-6
    grad_clip_norm: float = 1.0
    pos_weight_scale: float = 1.0
    precision_floor: float = 0.30
    threshold_metric: str = "f1"

    # Reference target (for reproduction gap report)
    target_epoch: int = 7
    target_val_loss: float = 0.9941
    target_val_auc: float = 0.8317
    target_val_recall: float = 0.6436

    # Runtime
    num_workers: int = 0
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    chinese_log: bool = False

    def make_dirs(self) -> None:
        Path(self.checkpoint_dir).mkdir(parents=True, exist_ok=True)
        Path(self.result_dir).mkdir(parents=True, exist_ok=True)


config = Config()
