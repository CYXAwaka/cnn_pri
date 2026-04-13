
"""
engine.py
~~~~~~~~~
这里放训练、验证、特征提取、LightGBM 训练、画图等逻辑。

这样 main.py 只负责“组织流程”，不会塞满细节。
"""

from __future__ import annotations

import copy
from pathlib import Path
from typing import Dict, List, Tuple

import joblib
import lightgbm as lgb
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


def compute_binary_metrics(y_true: np.ndarray, y_pred: np.ndarray, y_prob: np.ndarray) -> Dict[str, float]:
    """
    统一计算二分类指标。

    为什么要单独写这个函数？
    因为后面：
    - CNN 验证阶段要用
    - LightGBM 测试阶段也要用
    重复写会很乱。

    这里特别强调：
    不再使用 weighted F1 作为主要指标，
    因为样本严重不平衡时，weighted 指标很容易“看起来很好”，
    但其实模型可能只是在预测多数类。
    """
    metrics = {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
    }

    # 有些极端情况下，如果 y_true 只有一个类别，AUC 无法计算
    try:
        metrics["roc_auc"] = roc_auc_score(y_true, y_prob)
    except Exception:
        metrics["roc_auc"] = float("nan")

    try:
        metrics["pr_auc"] = average_precision_score(y_true, y_prob)
    except Exception:
        metrics["pr_auc"] = float("nan")

    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    metrics["tn"] = tn
    metrics["fp"] = fp
    metrics["fn"] = fn
    metrics["tp"] = tp
    return metrics


def train_one_epoch(model, dataloader, criterion, optimizer, device):
    """
    训练一个 epoch。
    """
    model.train()

    total_loss = 0.0
    all_preds = []
    all_probs = []
    all_labels = []

    for inputs, labels in dataloader:
        inputs = inputs.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()

        logits, _ = model(inputs)
        loss = criterion(logits, labels)

        loss.backward()
        optimizer.step()

        total_loss += loss.item()

        probs = torch.softmax(logits, dim=1)[:, 1]
        preds = torch.argmax(logits, dim=1)

        all_preds.extend(preds.detach().cpu().numpy())
        all_probs.extend(probs.detach().cpu().numpy())
        all_labels.extend(labels.detach().cpu().numpy())

    mean_loss = total_loss / max(len(dataloader), 1)
    metrics = compute_binary_metrics(
        y_true=np.asarray(all_labels),
        y_pred=np.asarray(all_preds),
        y_prob=np.asarray(all_probs),
    )
    return mean_loss, metrics


@torch.no_grad()
def validate_one_epoch(model, dataloader, criterion, device):
    """
    验证一个 epoch。
    """
    model.eval()

    total_loss = 0.0
    all_preds = []
    all_probs = []
    all_labels = []

    for inputs, labels in dataloader:
        inputs = inputs.to(device)
        labels = labels.to(device)

        logits, _ = model(inputs)
        loss = criterion(logits, labels)
        total_loss += loss.item()

        probs = torch.softmax(logits, dim=1)[:, 1]
        preds = torch.argmax(logits, dim=1)

        all_preds.extend(preds.detach().cpu().numpy())
        all_probs.extend(probs.detach().cpu().numpy())
        all_labels.extend(labels.detach().cpu().numpy())

    mean_loss = total_loss / max(len(dataloader), 1)
    metrics = compute_binary_metrics(
        y_true=np.asarray(all_labels),
        y_pred=np.asarray(all_preds),
        y_prob=np.asarray(all_probs),
    )
    return mean_loss, metrics


def train_cnn_feature_extractor(
    model,
    train_loader,
    val_loader,
    device,
    lr: float,
    weight_decay: float,
    epochs: int,
    patience: int,
    checkpoint_path: str,
):
    """
    训练 CNN 特征提取器（通过临时分类头来监督训练）。

    这里做了几件实用的事：
    1. 根据训练集类别不平衡自动计算 class weight
    2. 监控验证集 F1，而不是只看 Accuracy
    3. 加入 early stopping，防止过拟合
    4. 保存最佳权重

    为什么监控 F1？
    因为你的任务是窃电检测，少数类很重要，
    只看 Accuracy 很可能被“类别不平衡”骗了。
    """
    # 根据训练集统计类别权重
    train_labels = []
    for _, labels in train_loader:
        train_labels.extend(labels.numpy().tolist())
    train_labels = np.asarray(train_labels)

    class_counts = np.bincount(train_labels, minlength=2)
    class_weights = class_counts.sum() / (2.0 * np.maximum(class_counts, 1))
    class_weights = torch.tensor(class_weights, dtype=torch.float32, device=device)

    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    history = {
        "train_loss": [],
        "val_loss": [],
        "train_f1": [],
        "val_f1": [],
        "train_recall": [],
        "val_recall": [],
        "train_precision": [],
        "val_precision": [],
    }

    best_score = -1.0
    best_state = None
    early_stop_counter = 0

    for epoch in range(1, epochs + 1):
        train_loss, train_metrics = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_metrics = validate_one_epoch(model, val_loader, criterion, device)

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["train_f1"].append(train_metrics["f1"])
        history["val_f1"].append(val_metrics["f1"])
        history["train_recall"].append(train_metrics["recall"])
        history["val_recall"].append(val_metrics["recall"])
        history["train_precision"].append(train_metrics["precision"])
        history["val_precision"].append(val_metrics["precision"])

        print(
            f"[CNN 预训练] Epoch {epoch:02d} | "
            f"Train Loss={train_loss:.4f}, Train F1={train_metrics['f1']:.4f}, "
            f"Val Loss={val_loss:.4f}, Val F1={val_metrics['f1']:.4f}, "
            f"Val Recall={val_metrics['recall']:.4f}, Val AUC={val_metrics['roc_auc']:.4f}"
        )

        # 用验证集 F1 作为主指标
        current_score = val_metrics["f1"]

        if current_score > best_score:
            best_score = current_score
            best_state = copy.deepcopy(model.state_dict())
            torch.save(best_state, checkpoint_path)
            early_stop_counter = 0
        else:
            early_stop_counter += 1

        if early_stop_counter >= patience:
            print(f"早停触发：连续 {patience} 个 epoch 验证集 F1 未提升。")
            break

    if best_state is not None:
        model.load_state_dict(best_state)

    return model, history


@torch.no_grad()
def extract_features(model, dataloader, device):
    """
    用训练好的 CNN 提取 64 维特征。

    注意：
    这里提取的是 feature_extractor 的输出，不是最终分类 logits。
    """
    model.eval()

    all_features = []
    all_labels = []

    for inputs, labels in dataloader:
        inputs = inputs.to(device)
        _, features = model(inputs)

        all_features.append(features.detach().cpu().numpy())
        all_labels.append(labels.numpy())

    X_feat = np.concatenate(all_features, axis=0)
    y = np.concatenate(all_labels, axis=0)
    return X_feat, y


def train_lightgbm_classifier(
    X_train_feat: np.ndarray,
    y_train: np.ndarray,
    params: dict,
):
    """
    训练 LightGBM 分类器。

    这里使用 sklearn 风格接口，原因是：
    - 上手简单
    - 和你现有代码风格更一致
    - 容易保存 / 加载
    """
    clf = lgb.LGBMClassifier(**params)
    clf.fit(X_train_feat, y_train)
    return clf


def evaluate_lightgbm(
    clf,
    X_feat: np.ndarray,
    y_true: np.ndarray,
) -> Dict[str, float]:
    """
    评估 LightGBM。
    """
    y_prob = clf.predict_proba(X_feat)[:, 1]
    y_pred = (y_prob >= 0.5).astype(int)
    metrics = compute_binary_metrics(y_true=y_true, y_pred=y_pred, y_prob=y_prob)
    return metrics


def save_lightgbm_model(clf, path: str):
    """
    保存 LightGBM 模型。
    """
    joblib.dump(clf, path)
    print(f"LightGBM 模型已保存到: {path}")


def plot_training_history(history: dict, save_path: str):
    """
    画出 CNN 预训练阶段的曲线。

    为什么只画 CNN 曲线？
    因为 LightGBM 不是按 epoch 训练的，
    所以最直观的训练过程曲线主要来自 CNN 特征提取器阶段。
    """
    epochs = np.arange(1, len(history["train_loss"]) + 1)

    plt.figure(figsize=(12, 8))

    plt.subplot(2, 2, 1)
    plt.plot(epochs, history["train_loss"], label="train_loss")
    plt.plot(epochs, history["val_loss"], label="val_loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("CNN Pretrain Loss")
    plt.legend()

    plt.subplot(2, 2, 2)
    plt.plot(epochs, history["train_f1"], label="train_f1")
    plt.plot(epochs, history["val_f1"], label="val_f1")
    plt.xlabel("Epoch")
    plt.ylabel("F1")
    plt.title("CNN Pretrain F1")
    plt.legend()

    plt.subplot(2, 2, 3)
    plt.plot(epochs, history["train_recall"], label="train_recall")
    plt.plot(epochs, history["val_recall"], label="val_recall")
    plt.xlabel("Epoch")
    plt.ylabel("Recall")
    plt.title("CNN Pretrain Recall")
    plt.legend()

    plt.subplot(2, 2, 4)
    plt.plot(epochs, history["train_precision"], label="train_precision")
    plt.plot(epochs, history["val_precision"], label="val_precision")
    plt.xlabel("Epoch")
    plt.ylabel("Precision")
    plt.title("CNN Pretrain Precision")
    plt.legend()

    plt.tight_layout()
    plt.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"训练曲线已保存到: {save_path}")


def save_metrics_text(metrics_dict: Dict[str, Dict[str, float]], save_path: str):
    """
    将验证集/测试集指标保存成文本文件，方便你后续写论文时直接引用。
    """
    lines = []
    for split_name, metrics in metrics_dict.items():
        lines.append(f"{split_name}".upper())
        for key, value in metrics.items():
            lines.append(f"{key}: {value}")
        lines.append("")

    Path(save_path).write_text("\n".join(lines), encoding="utf-8")
    print(f"指标结果已保存到: {save_path}")
