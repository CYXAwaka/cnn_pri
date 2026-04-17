from __future__ import annotations

import random
from pathlib import Path
from typing import Any

import numpy as np
import torch

from config import config
from data_process import preprocess_for_wdcnn, split_dataset_for_ratio
from engine import (
    create_run_id,
    evaluate_wdcnn_model,
    plot_roc_pr_curves,
    plot_topn_precision_curve,
    plot_training_history,
    save_json,
    save_metrics_text,
    train_wdcnn_model,
)
from src.models.wdcnn_model import WideDeepCNN


def set_global_seed(seed: int) -> None:
    """
    固定所有主要随机源，提升实验可复现性。
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def build_model(split) -> WideDeepCNN:
    """
    按配置参数构建 WDCNN 模型实例。
    """
    return WideDeepCNN(
        wide_input_dim=split.wide_input_dim,
        deep_input_shape=split.deep_input_shape,
        alpha=config.alpha,
        beta=config.beta,
        gamma=config.gamma,
        r_layers=config.r_layers,
        dropout=config.dropout,
    )


def build_repro_gap(history: dict[str, list[float]]) -> dict[str, Any]:
    """
    生成“目标日志 vs 实际训练”差异报告（用于复现核对）。
    """
    if not history.get("val_loss"):
        return {}

    idx = min(config.target_epoch, len(history["val_loss"])) - 1
    val_loss = float(history["val_loss"][idx])
    val_auc = float(history["val_auc"][idx])
    val_recall = float(history["val_recall"][idx])

    return {
        "target": {
            "epoch": config.target_epoch,
            "val_loss": config.target_val_loss,
            "val_auc": config.target_val_auc,
            "val_recall": config.target_val_recall,
        },
        "actual": {
            "epoch": idx + 1,
            "val_loss": val_loss,
            "val_auc": val_auc,
            "val_recall": val_recall,
        },
        "delta_actual_minus_target": {
            "val_loss": val_loss - config.target_val_loss,
            "val_auc": val_auc - config.target_val_auc,
            "val_recall": val_recall - config.target_val_recall,
        },
    }


def main() -> None:
    """
    单配置实验入口：
    1) 初始化目录与随机种子
    2) 预处理 + 划分数据
    3) 训练并保存最佳权重
    4) 在测试集评估并输出图表/报告
    """
    config.make_dirs()
    set_global_seed(config.seed)

    checkpoint_dir = Path(config.checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    run_id, run_dir = create_run_id(config.result_dir)

    print("=" * 88)
    print("WDCNN single-run reproduction start")
    print(f"run_id: {run_id}")
    print(f"run_dir: {run_dir}")
    print(f"device: {config.device}")
    print(
        "fixed params: "
        f"ratio={config.train_ratio:.2f}, seed={config.seed}, "
        f"a{config.alpha}_b{config.beta}_g{config.gamma}_r{config.r_layers}, "
        f"lr={config.lr:.4g}, warmup={config.warmup_epochs}, "
        f"train_epochs={config.train_epochs}, scheduler_total_epochs={config.scheduler_total_epochs}, "
        f"pos_weight_scale={config.pos_weight_scale:.2f}, precision_floor={config.precision_floor:.2f}, "
        f"early_stop=off"
    )
    print("=" * 88)

    # 预处理：得到 1D + 2D 双输入。
    dataset = preprocess_for_wdcnn(
        file_path=config.data_path,
        id_col=config.id_col,
        label_col=config.label_col,
        days_per_week=config.days_per_week,
        fill_missing_calendar_days=config.fill_missing_calendar_days,
        use_outlier_clip=config.use_outlier_clip,
        outlier_sigma_k=config.outlier_sigma_k,
        normalize_eps=config.normalize_eps,
        week_pad_value=config.week_pad_value,
    )

    # 分层划分：train / val / test。
    split = split_dataset_for_ratio(
        dataset=dataset,
        train_ratio=config.train_ratio,
        val_ratio_in_train=config.val_ratio_in_train,
        batch_size=config.batch_size,
        random_state=config.seed,
        num_workers=config.num_workers,
    )

    model = build_model(split).to(config.device)
    ckpt_path = checkpoint_dir / f"wdcnn_{run_id}_seed{config.seed}.pth"

    # 训练主循环（自动保存最佳 checkpoint）。
    train_out = train_wdcnn_model(
        model=model,
        train_loader=split.train_loader,
        val_loader=split.val_loader,
        device=config.device,
        lr=config.lr,
        weight_decay=config.weight_decay,
        max_epochs=config.train_epochs,
        scheduler_total_epochs=config.scheduler_total_epochs,
        early_stop_patience=config.early_stop_patience,
        scheduler_factor=config.lr_scheduler_factor,
        scheduler_patience=config.lr_scheduler_patience,
        min_lr=config.min_lr,
        checkpoint_path=str(ckpt_path),
        threshold_metric=config.threshold_metric,
        fixed_val_threshold=None,
        use_cosine_schedule=config.use_cosine_schedule,
        warmup_epochs=config.warmup_epochs,
        grad_clip_norm=config.grad_clip_norm,
        pos_weight_scale=config.pos_weight_scale,
        precision_floor=config.precision_floor,
        chinese_log=config.chinese_log,
    )

    # 注意：测试集使用“验证阶段选出的最佳阈值”进行统计。
    test_metrics, y_test, p_test = evaluate_wdcnn_model(
        model=train_out.model,
        dataloader=split.test_loader,
        device=config.device,
        threshold=train_out.best_threshold,
    )

    # 统一产物命名。
    history_path = run_dir / f"history_{run_id}.png"
    roc_path = run_dir / f"roc_{run_id}.png"
    pr_path = run_dir / f"pr_{run_id}.png"
    topn_path = run_dir / f"topn_{run_id}.png"
    metrics_json_path = run_dir / f"metrics_{run_id}.json"
    metrics_txt_path = run_dir / f"metrics_{run_id}.txt"
    summary_json_path = run_dir / f"summary_{run_id}.json"
    summary_txt_path = run_dir / f"summary_{run_id}.txt"

    # 绘制结果图。
    plot_training_history(train_out.history, str(history_path))
    plot_roc_pr_curves(y_test, p_test, str(roc_path), str(pr_path))
    plot_topn_precision_curve(y_test, p_test, str(topn_path), max_n=200)

    repro_gap = build_repro_gap(train_out.history)

    # 详细指标报告（机器可读）
    metrics_payload = {
        "run_id": run_id,
        "stage": "single_repro",
        "config": {
            "train_ratio": config.train_ratio,
            "seed": config.seed,
            "alpha": config.alpha,
            "beta": config.beta,
            "gamma": config.gamma,
            "r_layers": config.r_layers,
            "dropout": config.dropout,
            "lr": config.lr,
            "warmup_epochs": config.warmup_epochs,
            "train_epochs": config.train_epochs,
            "scheduler_total_epochs": config.scheduler_total_epochs,
            "early_stop_patience": config.early_stop_patience,
            "pos_weight_scale": config.pos_weight_scale,
            "precision_floor": config.precision_floor,
        },
        "best_epoch": train_out.best_epoch,
        "best_threshold": train_out.best_threshold,
        "best_val_metrics": train_out.best_val_metrics,
        "test_metrics": test_metrics,
        "history_tail": {
            "val_loss_last": train_out.history["val_loss"][-1] if train_out.history.get("val_loss") else None,
            "val_auc_last": train_out.history["val_auc"][-1] if train_out.history.get("val_auc") else None,
            "val_recall_last": train_out.history["val_recall"][-1] if train_out.history.get("val_recall") else None,
        },
        "repro_gap_vs_target_log": repro_gap,
        "artifacts": {
            "checkpoint": str(ckpt_path),
            "history": str(history_path),
            "roc": str(roc_path),
            "pr": str(pr_path),
            "topn": str(topn_path),
        },
    }
    save_json(metrics_payload, str(metrics_json_path))
    save_metrics_text(metrics_payload, str(metrics_txt_path))

    # 摘要报告（人工快速查看）
    summary_payload = {
        "run_id": run_id,
        "message": "single-run training finished",
        "metrics_json": str(metrics_json_path),
        "metrics_text": str(metrics_txt_path),
        "repro_gap_vs_target_log": repro_gap,
    }
    save_json(summary_payload, str(summary_json_path))
    save_metrics_text(summary_payload, str(summary_txt_path))

    print("=" * 88)
    print("Training finished")
    print(f"best_epoch: {train_out.best_epoch}")
    print(f"best_threshold: {train_out.best_threshold:.2f}")
    print(f"best_val_auc: {train_out.best_val_metrics.get('auc', float('nan')):.4f}")
    print(f"best_val_recall: {train_out.best_val_metrics.get('recall', float('nan')):.4f}")
    print(f"summary: {summary_json_path}")
    print("=" * 88)


if __name__ == "__main__":
    main()
