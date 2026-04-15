from __future__ import annotations

import re
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
    save_records_csv,
    selection_tuple,
    summarize_metrics,
    train_wdcnn_model,
)
from src.models.wdcnn_model import WideDeepCNN

# Keep CPU runtime stable on some Windows environments.
torch.set_num_threads(1)
try:
    torch.set_num_interop_threads(1)
except Exception:
    pass


def ratio_tag(ratio: float) -> str:
    return f"r{int(round(float(ratio) * 100)):02d}"


def param_tag(params: dict[str, int | float]) -> str:
    return f"a{int(params['alpha'])}_b{int(params['beta'])}_g{int(params['gamma'])}_r{int(params['r_layers'])}"


def parse_param_tag(tag: str) -> dict[str, int]:
    m = re.match(r"^a(?P<a>\d+)_b(?P<b>\d+)_g(?P<g>\d+)_r(?P<r>\d+)$", tag)
    if not m:
        raise ValueError(f"Invalid param tag: {tag}")
    return {
        "alpha": int(m.group("a")),
        "beta": int(m.group("b")),
        "gamma": int(m.group("g")),
        "r_layers": int(m.group("r")),
    }


def build_model(split, params: dict[str, int | float]) -> WideDeepCNN:
    return WideDeepCNN(
        wide_input_dim=split.wide_input_dim,
        deep_input_shape=split.deep_input_shape,
        alpha=int(params["alpha"]),
        beta=int(params["beta"]),
        gamma=int(params["gamma"]),
        r_layers=int(params["r_layers"]),
        dropout=float(params["dropout"]),
    )


def load_global_best(target_run_id: str) -> dict[str, Any]:
    run_dir = Path(config.result_dir) / f"run_{target_run_id}"
    global_best_path = run_dir / f"global_best_{target_run_id}.json"
    if not global_best_path.exists():
        raise FileNotFoundError(f"Missing file: {global_best_path}")
    return save_load_json(global_best_path)


def save_load_json(path: Path) -> dict[str, Any]:
    import json

    return json.loads(path.read_text(encoding="utf-8"))


def mean_or_nan(values: list[float]) -> float:
    if not values:
        return float("nan")
    return float(np.mean(np.asarray(values, dtype=np.float64)))


def pick_best_seed(rows: list[dict[str, Any]]) -> dict[str, Any] | None:
    if not rows:
        return None
    return sorted(
        rows,
        key=lambda r: selection_tuple(
            {
                "auc": float(r.get("val_auc", np.nan)),
                "recall": float(r.get("val_recall", np.nan)),
                "map100": float(r.get("val_map100", np.nan)),
                "map200": float(r.get("val_map200", np.nan)),
                "precision": float(r.get("val_precision", np.nan)),
            }
        ),
        reverse=True,
    )[0]


def main() -> None:
    config.make_dirs()

    target_run_id = str(config.smooth_target_run_id or config.resume_run_id).strip()
    payload = load_global_best(target_run_id)

    best_ratio = float(payload["global_best_ratio"])
    best_p_tag = str(payload["global_best_param_tag"])
    parsed = parse_param_tag(best_p_tag)
    params = {
        "alpha": int(parsed["alpha"]),
        "beta": int(parsed["beta"]),
        "gamma": int(parsed["gamma"]),
        "r_layers": int(parsed["r_layers"]),
        "dropout": float(config.dropout),
    }

    seed_rows = payload.get("global_best_seed_rows", [])
    avg_epoch = mean_or_nan([float(x.get("best_epoch", np.nan)) for x in seed_rows if float(x.get("best_epoch", -1)) > 0])
    avg_threshold = mean_or_nan([float(x.get("best_threshold", np.nan)) for x in seed_rows if float(x.get("best_threshold", np.nan)) == float(x.get("best_threshold", np.nan))])

    smooth_run_id, smooth_run_dir = create_run_id(config.result_dir)
    smooth_run_dir = smooth_run_dir
    checkpoint_dir = Path(config.checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    smooth_lr = float(config.smooth_lr)
    smooth_epochs = int(max(config.smooth_max_epochs, int(round(avg_epoch * 3)) if avg_epoch == avg_epoch else config.smooth_max_epochs))
    smooth_patience = int(min(config.smooth_early_stop_patience, max(smooth_epochs - 2, 2)))
    smooth_threshold = float(config.smooth_fixed_threshold if config.smooth_fixed_threshold is not None else avg_threshold)

    print(f"Device: {config.device}")
    print(f"Source best run: {target_run_id}")
    print(f"Best group: ratio={best_ratio:.2f}, params={best_p_tag}")
    print(
        "Smoothing setup: "
        f"lr={smooth_lr:.6f}, max_epochs={smooth_epochs}, patience={smooth_patience}, "
        f"fixed_threshold={smooth_threshold:.3f}, warmup={config.smooth_warmup_epochs}, "
        f"cosine={bool(config.smooth_use_cosine_schedule)}, grad_clip={config.smooth_grad_clip_norm}"
    )
    print(f"Smooth Run ID: {smooth_run_id}")
    print(f"Smooth Run Dir: {smooth_run_dir}")

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

    r_tag = ratio_tag(best_ratio)
    p_tag = param_tag(params)
    suffix = str(config.smooth_artifact_suffix).strip() or "smooth"
    rows: list[dict[str, Any]] = []

    for seed in config.seed_list:
        split = split_dataset_for_ratio(
            dataset=dataset,
            train_ratio=best_ratio,
            val_ratio_in_train=config.val_ratio_in_train,
            batch_size=config.batch_size,
            random_state=int(seed),
            num_workers=config.num_workers,
        )

        model = build_model(split, params).to(config.device)
        ckpt = checkpoint_dir / f"wdcnn_{smooth_run_id}_{r_tag}_{p_tag}_seed{int(seed)}_{suffix}.pth"
        train_out = train_wdcnn_model(
            model=model,
            train_loader=split.train_loader,
            val_loader=split.val_loader,
            device=config.device,
            lr=smooth_lr,
            weight_decay=config.weight_decay,
            max_epochs=smooth_epochs,
            early_stop_patience=smooth_patience,
            scheduler_factor=config.lr_scheduler_factor,
            scheduler_patience=config.lr_scheduler_patience,
            min_lr=config.min_lr,
            checkpoint_path=str(ckpt),
            threshold_metric=config.threshold_metric,
            fixed_val_threshold=smooth_threshold,
            use_cosine_schedule=bool(config.smooth_use_cosine_schedule),
            warmup_epochs=int(config.smooth_warmup_epochs),
            grad_clip_norm=float(config.smooth_grad_clip_norm),
        )

        test_metrics, y_test, p_test = evaluate_wdcnn_model(
            train_out.model,
            split.test_loader,
            device=config.device,
            threshold=train_out.best_threshold,
        )

        tag = f"{r_tag}_{p_tag}_seed{int(seed)}_{smooth_run_id}_{suffix}"
        history_path = smooth_run_dir / f"history_{tag}.png"
        roc_path = smooth_run_dir / f"roc_{tag}.png"
        pr_path = smooth_run_dir / f"pr_{tag}.png"
        topn_path = smooth_run_dir / f"topn_{tag}.png"
        metrics_json_path = smooth_run_dir / f"metrics_{tag}.json"
        metrics_txt_path = smooth_run_dir / f"metrics_{tag}.txt"

        plot_training_history(train_out.history, str(history_path))
        plot_roc_pr_curves(y_test, p_test, str(roc_path), str(pr_path))
        plot_topn_precision_curve(y_test, p_test, str(topn_path), max_n=200)

        seed_payload = {
            "run_id": smooth_run_id,
            "stage": "smooth_fine",
            "source_best_run_id": target_run_id,
            "train_ratio": best_ratio,
            "seed": int(seed),
            "params": params,
            "smoothing_config": {
                "lr": smooth_lr,
                "max_epochs": smooth_epochs,
                "early_stop_patience": smooth_patience,
                "fixed_threshold": smooth_threshold,
                "warmup_epochs": int(config.smooth_warmup_epochs),
                "use_cosine_schedule": bool(config.smooth_use_cosine_schedule),
                "grad_clip_norm": float(config.smooth_grad_clip_norm),
            },
            "best_epoch": int(train_out.best_epoch),
            "best_threshold": float(train_out.best_threshold),
            "val_metrics": train_out.best_val_metrics,
            "test_metrics": test_metrics,
            "artifacts": {
                "checkpoint": str(ckpt),
                "history": str(history_path),
                "roc": str(roc_path),
                "pr": str(pr_path),
                "topn": str(topn_path),
            },
        }

        save_json(seed_payload, str(metrics_json_path))
        save_metrics_text(seed_payload, str(metrics_txt_path))

        row = {
            "run_id": smooth_run_id,
            "stage": "smooth_fine",
            "source_best_run_id": target_run_id,
            "train_ratio": best_ratio,
            "seed": int(seed),
            "alpha": int(params["alpha"]),
            "beta": int(params["beta"]),
            "gamma": int(params["gamma"]),
            "r_layers": int(params["r_layers"]),
            "lr": smooth_lr,
            "max_epochs": smooth_epochs,
            "early_stop_patience": smooth_patience,
            "best_epoch": int(train_out.best_epoch),
            "best_threshold": float(train_out.best_threshold),
            **{f"val_{k}": float(v) for k, v in train_out.best_val_metrics.items()},
            **{f"test_{k}": float(v) for k, v in test_metrics.items()},
            "metrics_json": str(metrics_json_path),
            "metrics_text": str(metrics_txt_path),
            "checkpoint": str(ckpt),
        }
        rows.append(row)

    csv_path = smooth_run_dir / f"smooth_fine_records_{smooth_run_id}.csv"
    save_records_csv(rows, str(csv_path))

    metric_keys = ["val_auc", "val_recall", "test_auc", "test_recall", "test_precision", "test_map100", "test_map200", "test_f1"]
    summary = summarize_metrics(rows, metric_keys)
    best_seed = pick_best_seed(rows)

    final_payload = {
        "run_id": smooth_run_id,
        "source_best_run_id": target_run_id,
        "selected_group": {
            "train_ratio": best_ratio,
            "param_tag": best_p_tag,
            "params": params,
        },
        "smoothing_config": {
            "lr": smooth_lr,
            "max_epochs": smooth_epochs,
            "early_stop_patience": smooth_patience,
            "fixed_threshold": smooth_threshold,
            "warmup_epochs": int(config.smooth_warmup_epochs),
            "use_cosine_schedule": bool(config.smooth_use_cosine_schedule),
            "grad_clip_norm": float(config.smooth_grad_clip_norm),
        },
        "rows_csv": str(csv_path),
        "rows": rows,
        "summary": summary,
        "best_seed_row": best_seed,
    }

    summary_json = smooth_run_dir / f"smooth_summary_{smooth_run_id}.json"
    summary_txt = smooth_run_dir / f"smooth_summary_{smooth_run_id}.txt"
    save_json(final_payload, str(summary_json))
    save_metrics_text(final_payload, str(summary_txt))

    print("=" * 88)
    print("Best-group smooth retrain finished")
    print(f"Summary JSON: {summary_json}")
    print("=" * 88)


if __name__ == "__main__":
    main()
