from __future__ import annotations

import copy
import csv
import json
import math
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import (
    auc,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)


@dataclass
class TrainingOutput:
    model: nn.Module
    history: dict[str, list[float]]
    best_val_metrics: dict[str, float]
    best_threshold: float
    best_epoch: int
    checkpoint_path: str


def create_run_id(result_dir: str) -> tuple[str, Path]:
    """Create unique run id: YYYYMMDD_HHMMSS_vNNN and run directory."""
    base_ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    result_path = Path(result_dir)
    result_path.mkdir(parents=True, exist_ok=True)

    version = 1
    while True:
        run_id = f"{base_ts}_v{version:03d}"
        run_dir = result_path / f"run_{run_id}"
        if not run_dir.exists():
            run_dir.mkdir(parents=True, exist_ok=False)
            return run_id, run_dir
        version += 1


def selection_tuple(metrics: dict[str, float]) -> tuple[float, float, float, float, float]:
    """Primary then tie-break: AUC -> Recall -> MAP@100 -> MAP@200 -> Precision."""

    def _safe(v: float) -> float:
        if v is None or (isinstance(v, float) and np.isnan(v)):
            return float("-inf")
        return float(v)

    return (
        _safe(metrics.get("auc", float("nan"))),
        _safe(metrics.get("recall", float("nan"))),
        _safe(metrics.get("map100", float("nan"))),
        _safe(metrics.get("map200", float("nan"))),
        _safe(metrics.get("precision", float("nan"))),
    )


def map_at_n(y_true: np.ndarray, y_prob: np.ndarray, n: int) -> float:
    """Paper MAP@N definition using P@k over positive positions inside top-N."""
    if len(y_true) == 0:
        return 0.0
    n = min(int(n), len(y_true))
    if n <= 0:
        return 0.0

    order = np.argsort(-y_prob)
    top_labels = y_true[order][:n]

    positives = np.where(top_labels == 1)[0]
    if positives.size == 0:
        return 0.0

    cum_pos = np.cumsum(top_labels == 1)
    precisions = [cum_pos[k] / float(k + 1) for k in positives]
    return float(np.mean(precisions))


def precision_at_n_curve(y_true: np.ndarray, y_prob: np.ndarray, max_n: int = 200) -> np.ndarray:
    if len(y_true) == 0:
        return np.zeros(max_n, dtype=np.float32)

    order = np.argsort(-y_prob)
    sorted_labels = (y_true[order] == 1).astype(np.int32)
    k = min(max_n, len(sorted_labels))
    cum_pos = np.cumsum(sorted_labels[:k])
    curve = cum_pos / (np.arange(k) + 1)

    if k < max_n:
        pad = np.full(max_n - k, curve[-1] if k > 0 else 0.0, dtype=np.float32)
        curve = np.concatenate([curve.astype(np.float32), pad], axis=0)
    return curve.astype(np.float32)


def compute_binary_metrics(y_true: np.ndarray, y_prob: np.ndarray, threshold: float) -> dict[str, float]:
    y_true = y_true.astype(int)
    y_pred = (y_prob >= threshold).astype(int)

    metrics: dict[str, float] = {
        "threshold": float(threshold),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "map100": map_at_n(y_true, y_prob, 100),
        "map200": map_at_n(y_true, y_prob, 200),
    }

    try:
        metrics["auc"] = float(roc_auc_score(y_true, y_prob))
    except Exception:
        metrics["auc"] = float("nan")

    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    metrics["tn"] = float(tn)
    metrics["fp"] = float(fp)
    metrics["fn"] = float(fn)
    metrics["tp"] = float(tp)
    return metrics


def select_best_threshold(y_true: np.ndarray, y_prob: np.ndarray, metric: str = "f1") -> float:
    candidates = np.linspace(0.05, 0.95, 91)
    best_thr = 0.5
    best_score = -1.0

    for thr in candidates:
        y_pred = (y_prob >= thr).astype(int)
        if metric == "recall":
            score = recall_score(y_true, y_pred, zero_division=0)
        elif metric == "precision":
            score = precision_score(y_true, y_pred, zero_division=0)
        else:
            score = f1_score(y_true, y_pred, zero_division=0)

        if score > best_score:
            best_score = float(score)
            best_thr = float(thr)

    return best_thr


def select_threshold_with_precision_floor(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    precision_floor: float,
    fallback_metric: str = "f1",
) -> float:
    """Pick threshold by max Recall under Precision floor; fallback to metric-optimal threshold."""
    candidates = np.linspace(0.05, 0.95, 91)
    eligible: list[tuple[float, float, float, float]] = []  # recall, f1, precision, threshold

    for thr in candidates:
        y_pred = (y_prob >= thr).astype(int)
        prec = float(precision_score(y_true, y_pred, zero_division=0))
        rec = float(recall_score(y_true, y_pred, zero_division=0))
        f1 = float(f1_score(y_true, y_pred, zero_division=0))
        if prec >= float(precision_floor):
            eligible.append((rec, f1, prec, float(thr)))

    if eligible:
        eligible.sort(key=lambda x: (x[0], x[1], x[2]), reverse=True)
        return float(eligible[0][3])

    return select_best_threshold(y_true, y_prob, metric=fallback_metric)


def _run_epoch(
    model: nn.Module,
    dataloader,
    criterion,
    device: str,
    optimizer=None,
    grad_clip_norm: float | None = None,
) -> tuple[float, np.ndarray, np.ndarray]:
    is_train = optimizer is not None
    if is_train:
        model.train()
    else:
        model.eval()

    total_loss = 0.0
    all_probs: list[np.ndarray] = []
    all_labels: list[np.ndarray] = []

    context = torch.enable_grad() if is_train else torch.no_grad()
    with context:
        for x1, x2, y in dataloader:
            x1 = x1.to(device)
            x2 = x2.to(device)
            y = y.to(device)

            logits, _ = model(x1, x2)
            loss = criterion(logits, y)

            if is_train:
                optimizer.zero_grad()
                loss.backward()
                if grad_clip_norm is not None and grad_clip_norm > 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=float(grad_clip_norm))
                optimizer.step()

            total_loss += float(loss.item())
            probs = torch.sigmoid(logits)
            all_probs.append(probs.detach().cpu().numpy())
            all_labels.append(y.detach().cpu().numpy())

    mean_loss = total_loss / max(len(dataloader), 1)
    y_prob = np.concatenate(all_probs, axis=0).astype(np.float32)
    y_true = np.concatenate(all_labels, axis=0).astype(np.int64)
    return mean_loss, y_true, y_prob


def train_wdcnn_model(
    model: nn.Module,
    train_loader,
    val_loader,
    device: str,
    lr: float,
    weight_decay: float,
    max_epochs: int,
    early_stop_patience: int,
    scheduler_factor: float,
    scheduler_patience: int,
    min_lr: float,
    checkpoint_path: str,
    threshold_metric: str = "f1",
    fixed_val_threshold: float | None = None,
    use_cosine_schedule: bool = False,
    warmup_epochs: int = 0,
    grad_clip_norm: float | None = None,
    pos_weight_scale: float = 1.0,
    precision_floor: float | None = None,
) -> TrainingOutput:
    train_labels = []
    for _, _, y in train_loader:
        train_labels.extend(y.numpy().tolist())
    train_labels = np.asarray(train_labels, dtype=np.int64)

    pos = max(int(np.sum(train_labels == 1)), 1)
    neg = max(int(np.sum(train_labels == 0)), 1)
    pos_weight = torch.tensor([float(neg / pos) * float(pos_weight_scale)], dtype=torch.float32, device=device)

    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler_mode = "plateau"
    if use_cosine_schedule:
        total_epochs = max(int(max_epochs), 1)
        warm = max(int(warmup_epochs), 0)
        base_lr = max(float(lr), 1e-12)
        min_factor = float(np.clip(min_lr / base_lr, 0.0, 1.0))

        def _lr_lambda(epoch_zero_idx: int) -> float:
            ep = epoch_zero_idx + 1
            if warm > 0 and ep <= warm:
                return 0.25 + 0.75 * (ep / float(warm))

            progress = (ep - warm) / float(max(total_epochs - warm, 1))
            progress = float(np.clip(progress, 0.0, 1.0))
            cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
            return min_factor + (1.0 - min_factor) * cosine

        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=_lr_lambda)
        scheduler_mode = "cosine"
    else:
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="min",
            factor=scheduler_factor,
            patience=scheduler_patience,
            min_lr=min_lr,
        )

    history: dict[str, list[float]] = {
        "train_loss": [],
        "val_loss": [],
        "train_auc": [],
        "val_auc": [],
        "train_map100": [],
        "val_map100": [],
        "train_map200": [],
        "val_map200": [],
        "train_precision": [],
        "val_precision": [],
        "train_recall": [],
        "val_recall": [],
        "train_f1": [],
        "val_f1": [],
        "val_threshold": [],
        "lr": [],
    }

    best_epoch = 0
    best_state = None
    best_val_metrics: dict[str, float] = {}
    best_threshold = 0.5
    best_score = selection_tuple({})
    stale_epochs = 0

    for epoch in range(1, max_epochs + 1):
        train_loss, y_train, p_train = _run_epoch(
            model,
            train_loader,
            criterion,
            device,
            optimizer=optimizer,
            grad_clip_norm=grad_clip_norm,
        )
        val_loss, y_val, p_val = _run_epoch(model, val_loader, criterion, device, optimizer=None)

        if fixed_val_threshold is not None:
            val_threshold = float(fixed_val_threshold)
        elif precision_floor is not None:
            val_threshold = select_threshold_with_precision_floor(
                y_true=y_val,
                y_prob=p_val,
                precision_floor=float(precision_floor),
                fallback_metric=threshold_metric,
            )
        else:
            val_threshold = select_best_threshold(y_val, p_val, metric=threshold_metric)
        train_metrics = compute_binary_metrics(y_train, p_train, threshold=0.5)
        val_metrics = compute_binary_metrics(y_val, p_val, threshold=val_threshold)

        if scheduler_mode == "cosine":
            scheduler.step()
        else:
            scheduler.step(val_loss)

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["train_auc"].append(train_metrics["auc"])
        history["val_auc"].append(val_metrics["auc"])
        history["train_map100"].append(train_metrics["map100"])
        history["val_map100"].append(val_metrics["map100"])
        history["train_map200"].append(train_metrics["map200"])
        history["val_map200"].append(val_metrics["map200"])
        history["train_precision"].append(train_metrics["precision"])
        history["val_precision"].append(val_metrics["precision"])
        history["train_recall"].append(train_metrics["recall"])
        history["val_recall"].append(val_metrics["recall"])
        history["train_f1"].append(train_metrics["f1"])
        history["val_f1"].append(val_metrics["f1"])
        history["val_threshold"].append(val_threshold)
        history["lr"].append(float(optimizer.param_groups[0]["lr"]))

        print(
            f"[WDCNN] Epoch {epoch:03d} | "
            f"train_loss={train_loss:.4f} val_loss={val_loss:.4f} | "
            f"val_auc={val_metrics['auc']:.4f} val_map100={val_metrics['map100']:.4f} "
            f"val_map200={val_metrics['map200']:.4f} val_recall={val_metrics['recall']:.4f} "
            f"val_precision={val_metrics['precision']:.4f} thr={val_threshold:.2f} "
            f"lr={optimizer.param_groups[0]['lr']:.6f}"
        )

        cur_score = selection_tuple(val_metrics)
        if cur_score > best_score:
            best_score = cur_score
            best_state = copy.deepcopy(model.state_dict())
            best_epoch = epoch
            best_val_metrics = val_metrics
            best_threshold = val_threshold
            torch.save(best_state, checkpoint_path)
            stale_epochs = 0
        else:
            stale_epochs += 1

        if stale_epochs >= early_stop_patience:
            print(f"Early stopping at epoch={epoch}, patience={early_stop_patience}")
            break

    if best_state is not None:
        model.load_state_dict(best_state)

    return TrainingOutput(
        model=model,
        history=history,
        best_val_metrics=best_val_metrics,
        best_threshold=best_threshold,
        best_epoch=best_epoch,
        checkpoint_path=checkpoint_path,
    )


@torch.no_grad()
def evaluate_wdcnn_model(model: nn.Module, dataloader, device: str, threshold: float) -> tuple[dict[str, float], np.ndarray, np.ndarray]:
    criterion = nn.BCEWithLogitsLoss()
    loss, y_true, y_prob = _run_epoch(model, dataloader, criterion, device, optimizer=None)
    metrics = compute_binary_metrics(y_true, y_prob, threshold=threshold)
    metrics["loss"] = float(loss)
    return metrics, y_true, y_prob


def plot_training_history(history: dict[str, list[float]], save_path: str) -> None:
    epochs = np.arange(1, len(history["train_loss"]) + 1)
    if len(epochs) == 0:
        return

    plt.figure(figsize=(14, 10))

    plt.subplot(2, 3, 1)
    plt.plot(epochs, history["train_loss"], label="train")
    plt.plot(epochs, history["val_loss"], label="val")
    plt.title("Loss")
    plt.xlabel("Epoch")
    plt.legend()

    plt.subplot(2, 3, 2)
    plt.plot(epochs, history["train_auc"], label="train")
    plt.plot(epochs, history["val_auc"], label="val")
    plt.title("AUC")
    plt.xlabel("Epoch")
    plt.legend()

    plt.subplot(2, 3, 3)
    plt.plot(epochs, history["train_map100"], label="train")
    plt.plot(epochs, history["val_map100"], label="val")
    plt.title("MAP@100")
    plt.xlabel("Epoch")
    plt.legend()

    plt.subplot(2, 3, 4)
    plt.plot(epochs, history["train_map200"], label="train")
    plt.plot(epochs, history["val_map200"], label="val")
    plt.title("MAP@200")
    plt.xlabel("Epoch")
    plt.legend()

    plt.subplot(2, 3, 5)
    plt.plot(epochs, history["train_recall"], label="train_recall")
    plt.plot(epochs, history["val_recall"], label="val_recall")
    plt.plot(epochs, history["train_precision"], label="train_precision")
    plt.plot(epochs, history["val_precision"], label="val_precision")
    plt.title("Recall & Precision")
    plt.xlabel("Epoch")
    plt.legend(fontsize=8)

    plt.subplot(2, 3, 6)
    plt.plot(epochs, history["val_f1"], label="val_f1")
    plt.plot(epochs, history["val_threshold"], label="val_threshold")
    plt.title("Val F1 & Threshold")
    plt.xlabel("Epoch")
    plt.legend()

    plt.tight_layout()
    plt.savefig(save_path, dpi=220, bbox_inches="tight")
    plt.close()


def plot_roc_pr_curves(y_true: np.ndarray, y_prob: np.ndarray, roc_path: str, pr_path: str) -> None:
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    roc_auc = auc(fpr, tpr)

    precision, recall, _ = precision_recall_curve(y_true, y_prob)
    pr_auc = auc(recall, precision)

    plt.figure(figsize=(6, 5))
    plt.plot(fpr, tpr, label=f"AUC={roc_auc:.4f}")
    plt.plot([0, 1], [0, 1], linestyle="--", color="gray")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve")
    plt.legend()
    plt.tight_layout()
    plt.savefig(roc_path, dpi=220, bbox_inches="tight")
    plt.close()

    plt.figure(figsize=(6, 5))
    plt.plot(recall, precision, label=f"AUC={pr_auc:.4f}")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("PR Curve")
    plt.legend()
    plt.tight_layout()
    plt.savefig(pr_path, dpi=220, bbox_inches="tight")
    plt.close()


def plot_topn_precision_curve(y_true: np.ndarray, y_prob: np.ndarray, save_path: str, max_n: int = 200) -> None:
    curve = precision_at_n_curve(y_true, y_prob, max_n=max_n)
    x = np.arange(1, max_n + 1)
    plt.figure(figsize=(7, 5))
    plt.plot(x, curve)
    plt.axvline(100, linestyle="--", color="gray", linewidth=1)
    plt.axvline(200, linestyle="--", color="gray", linewidth=1)
    plt.title("Top-N Precision Curve")
    plt.xlabel("N")
    plt.ylabel("Precision@N")
    plt.tight_layout()
    plt.savefig(save_path, dpi=220, bbox_inches="tight")
    plt.close()


def save_json(obj: Any, save_path: str) -> None:
    Path(save_path).write_text(json.dumps(obj, indent=2, ensure_ascii=False), encoding="utf-8")


def save_metrics_text(metrics: dict[str, Any], save_path: str) -> None:
    lines: list[str] = []
    for key, value in metrics.items():
        if isinstance(value, dict):
            lines.append(str(key).upper())
            for k, v in value.items():
                lines.append(f"{k}: {v}")
            lines.append("")
        else:
            lines.append(f"{key}: {value}")
    Path(save_path).write_text("\n".join(lines), encoding="utf-8")


def save_records_csv(records: list[dict[str, Any]], save_path: str) -> None:
    if not records:
        return

    keys = sorted({k for row in records for k in row.keys()})
    with Path(save_path).open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        for row in records:
            writer.writerow(row)


def summarize_metrics(records: list[dict[str, Any]], metric_keys: list[str]) -> dict[str, dict[str, float]]:
    if not records:
        return {}
    summary: dict[str, dict[str, float]] = {}
    for key in metric_keys:
        values = np.asarray([float(r[key]) for r in records if key in r and not np.isnan(float(r[key]))], dtype=np.float64)
        if values.size == 0:
            summary[key] = {"mean": float("nan"), "std": float("nan")}
        else:
            summary[key] = {"mean": float(np.mean(values)), "std": float(np.std(values, ddof=0))}
    return summary
