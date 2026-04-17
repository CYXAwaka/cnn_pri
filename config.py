from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import torch


@dataclass
class Config:
    """
    项目全局配置（单配置复现实验版）。

    说明：
    - 当前代码默认走“单次训练 + 固定参数”路径，便于稳定复现实验。
    - 绝大多数训练行为（数据划分、模型结构、优化参数、日志语言）都由此文件统一控制。
    """

    # =========================
    # 一、路径配置
    # =========================
    # 原始数据文件路径（支持 csv / xlsx / xls）
    data_path: str = "data/raw/data.csv"
    # 权重保存目录
    checkpoint_dir: str = "checkpoints/repro_stageA001"
    # 结果产物目录（图、json、txt 等）
    result_dir: str = "results/repro_stageA001"

    # =========================
    # 二、数据列配置
    # =========================
    # 用户唯一标识列名
    id_col: str = "CONS_NO"
    # 标签列名（0=正常，1=窃电）
    label_col: str = "FLAG"

    # =========================
    # 三、预处理配置（论文对齐）
    # =========================
    # 一周天数，默认 7 天
    days_per_week: int = 7
    # 是否按完整日历补齐日期列（缺失日期整列补出）
    fill_missing_calendar_days: bool = True
    # 是否启用异常值上截断（均值 + k*标准差）
    use_outlier_clip: bool = True
    # 异常值截断系数 k
    outlier_sigma_k: float = 2.0
    # Min-Max 分母最小值，防止除 0
    normalize_eps: float = 1e-8
    # 补齐到整周时的填充值
    week_pad_value: float = 0.0

    # =========================
    # 四、数据划分与随机性
    # =========================
    # 训练集比例（其余作为测试集）
    train_ratio: float = 0.8
    # 从训练集中再切分验证集比例
    val_ratio_in_train: float = 0.1
    # 全局随机种子
    seed: int = 42

    # =========================
    # 五、模型超参数（WDCNN）
    # =========================
    # Wide 分支全连接维度
    alpha: int = 90
    # Deep 分支全连接维度
    beta: int = 120
    # Deep 分支卷积通道数
    gamma: int = 20
    # Deep 分支卷积层数
    r_layers: int = 3
    # Dropout 比例
    dropout: float = 0.2

    # =========================
    # 六、训练超参数
    # =========================
    batch_size: int = 128
    # AdamW 学习率
    lr: float = 4e-4
    # AdamW 权重衰减
    weight_decay: float = 1e-4
    # 实际训练轮次
    train_epochs: int = 30
    # 学习率调度总轮次（可与 train_epochs 相同或分离）
    scheduler_total_epochs: int = 30
    # warmup 轮次
    warmup_epochs: int = 1
    # 是否使用余弦学习率调度
    use_cosine_schedule: bool = True
    # 早停耐心值，<=0 表示关闭早停
    early_stop_patience: int = 0
    # Plateau 调度参数（仅在 use_cosine_schedule=False 时生效）
    lr_scheduler_factor: float = 0.5
    lr_scheduler_patience: int = 4
    # 最小学习率下限
    min_lr: float = 1e-6
    # 梯度裁剪阈值（L2 范数）
    grad_clip_norm: float = 1.0
    # 正样本权重缩放系数（实际 pos_weight = neg/pos * scale）
    pos_weight_scale: float = 1.0
    # 阈值搜索时的 precision 下限约束
    precision_floor: float = 0.30
    # 无 precision_floor 可用阈值时的回退指标
    threshold_metric: str = "f1"

    # =========================
    # 七、目标日志参考（用于复现差异报告）
    # =========================
    target_epoch: int = 7
    target_val_loss: float = 0.9941
    target_val_auc: float = 0.8317
    target_val_recall: float = 0.6436

    # =========================
    # 八、运行时
    # =========================
    num_workers: int = 0
    # 自动选择设备
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    # 是否使用中文训练日志（False=英文）
    chinese_log: bool = False

    def make_dirs(self) -> None:
        """确保输出目录存在。"""
        Path(self.checkpoint_dir).mkdir(parents=True, exist_ok=True)
        Path(self.result_dir).mkdir(parents=True, exist_ok=True)


config = Config()
