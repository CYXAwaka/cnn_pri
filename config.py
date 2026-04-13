
"""
config.py
~~~~~~~~~
这个文件专门负责“集中管理超参数”。

你后续只需要改这里，就能影响整个项目：
- 数据路径
- 模型结构参数
- 训练参数
- LightGBM 参数
- 输出目录

这样做的好处是：
1. main.py 会非常干净；
2. 后续你想做 LSTM / CNN / CNN-LG 对比实验时，不会到处改数字；
3. 写论文时，也方便你统一整理“实验参数设置”。
"""

from dataclasses import dataclass, field
from pathlib import Path
import torch


@dataclass
class Config:
    # =========================
    # 1. 路径相关
    # =========================
    # 这里默认使用你项目中的原始数据路径。
    # 如果你之后把数据换成新的 csv，只改这里即可。
    data_path: str = "data/raw/data.csv"

    # 输出目录：模型、图像、日志等都会保存到这里
    checkpoint_dir: str = "checkpoints"
    result_dir: str = "results"

    # =========================
    # 2. 数据字段相关
    # =========================
    # SGCC 常见字段
    id_col: str = "CONS_NO"
    label_col: str = "FLAG"

    # =========================
    # 3. 数据预处理相关
    # =========================
    # 允许样本最大缺失率；超过这个比例就删除该样本
    missing_threshold: float = 0.30

    # 是否执行 3σ 离群值修复
    use_outlier_repair: bool = True

    # 按论文思路，将日数据整理为“周矩阵”
    # SGCC 原始公开数据通常是 1035 天，论文中使用 147×7 的输入
    days_per_week: int = 7
    target_weeks: int = 147

    # 数据集划分比例
    train_ratio: float = 0.40
    val_ratio: float = 0.10
    test_ratio: float = 0.50

    random_state: int = 42

    # 是否在训练集上做随机过采样
    # 论文里提到为了平衡样本，采用随机过采样
    use_random_oversample: bool = True

    # =========================
    # 4. CNN 特征提取器参数
    # =========================
    # 卷积结构尽量贴近论文图 4 / 表 1：
    # Conv(3x3) -> Pool(3x3) -> Conv(3x3) -> Pool(2x2) -> FC(64)
    in_channels: int = 1
    conv_channels: int = 16
    conv_kernel_size: int = 3

    # 注意：为了复现论文中的尺寸变化：
    # 147×7 -> 池化后约 49×2 -> 再池化后约 24×1
    pool1_kernel_size: int = 3
    pool1_stride: int = 3
    pool2_kernel_size: int = 2
    pool2_stride: int = 2

    fc_dim: int = 64
    num_classes: int = 2
    dropout: float = 0.20

    # =========================
    # 5. CNN 预训练参数
    # =========================
    batch_size: int = 32
    cnn_epochs: int = 30
    cnn_lr: float = 1e-3
    weight_decay: float = 1e-4
    early_stop_patience: int = 8

    # =========================
    # 6. LightGBM 参数
    # =========================
    # 尽量贴近论文表 1：
    # num_leaves = 25, max_depth = 5, learning_rate = 0.1
    lgbm_params: dict = field(default_factory=lambda: {
        "objective": "binary",
        "n_estimators": 200,
        "learning_rate": 0.1,
        "num_leaves": 25,
        "max_depth": 5,
        "min_child_samples": 20,
        "subsample": 0.9,
        "colsample_bytree": 0.9,
        "reg_alpha": 0.0,
        "reg_lambda": 0.0,
        "random_state": 42,
        "class_weight": "balanced",
        "verbosity": -1,
        "n_jobs": 1,
    })

    # =========================
    # 7. 运行设备
    # =========================
    device: str = "cuda" if torch.cuda.is_available() else "cpu"

    def make_dirs(self):
        """创建输出目录。"""
        Path(self.checkpoint_dir).mkdir(parents=True, exist_ok=True)
        Path(self.result_dir).mkdir(parents=True, exist_ok=True)


config = Config()
