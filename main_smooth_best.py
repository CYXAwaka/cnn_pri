from __future__ import annotations

import argparse
import itertools
import json
import re
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import torch

from config import config
from data_process import preprocess_for_wdcnn, split_dataset_for_ratio
from engine import (
    compute_binary_metrics,
    create_run_id,
    evaluate_wdcnn_model,
    plot_roc_pr_curves,
    plot_topn_precision_curve,
    plot_training_history,
    save_json,
    save_metrics_text,
    save_records_csv,
    select_best_threshold,
    select_threshold_with_precision_floor,
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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Best-group smooth_v2 retrain with two-stage search.")
    parser.add_argument("--smoke", action="store_true", help="Run 1-seed quick smoke pass.")
    return parser.parse_args()


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


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def config_id(stage: str, lr: float, warmup: int, patience: int, pos_weight_scale: float) -> str:
    lr_code = int(round(float(lr) * 1_000_000))
    pw_code = int(round(float(pos_weight_scale) * 100))
    return f"{stage}_l{lr_code}_w{int(warmup)}_p{int(patience)}_pw{pw_code}"


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


def run_threshold_selector_selftest() -> None:
    y_true = np.asarray([1, 1, 1, 0, 0, 0], dtype=np.int64)
    y_prob = np.asarray([0.9, 0.8, 0.4, 0.7, 0.6, 0.1], dtype=np.float32)
    precision_floor = 0.30
    thr = select_threshold_with_precision_floor(y_true, y_prob, precision_floor=precision_floor, fallback_metric="f1")
    m = compute_binary_metrics(y_true, y_prob, threshold=thr)
    if m["precision"] < precision_floor:
        raise AssertionError("Precision-floor selector failed to satisfy precision floor.")

    candidates = np.linspace(0.05, 0.95, 91)
    max_recall = -1.0
    for c in candidates:
        mc = compute_binary_metrics(y_true, y_prob, threshold=float(c))
        if mc["precision"] >= precision_floor:
            max_recall = max(max_recall, float(mc["recall"]))
    if max_recall >= 0 and float(m["recall"]) + 1e-12 < max_recall:
        raise AssertionError("Precision-floor selector did not maximize recall among eligible thresholds.")

    fallback_thr = select_threshold_with_precision_floor(y_true, y_prob, precision_floor=1.01, fallback_metric="f1")
    plain_f1_thr = select_best_threshold(y_true, y_prob, metric="f1")
    if abs(float(fallback_thr) - float(plain_f1_thr)) > 1e-12:
        raise AssertionError("Fallback threshold is inconsistent with F1-optimal threshold.")

    print("[SelfTest] threshold selector passed.")


def load_global_best(source_run_id: str) -> tuple[float, dict[str, int | float], dict[str, Any]]:
    p = Path(config.result_dir) / f"run_{source_run_id}" / f"global_best_{source_run_id}.json"
    if not p.exists():
        raise FileNotFoundError(f"Missing global best file: {p}")
    payload = load_json(p)
    ratio = float(payload["global_best_ratio"])
    ptag = str(payload["global_best_param_tag"])
    parsed = parse_param_tag(ptag)
    params = {
        "alpha": int(parsed["alpha"]),
        "beta": int(parsed["beta"]),
        "gamma": int(parsed["gamma"]),
        "r_layers": int(parsed["r_layers"]),
        "dropout": float(config.dropout),
    }
    return ratio, params, payload


def baseline_global_best(payload: dict[str, Any]) -> dict[str, float]:
    r = payload.get("global_best_ratio_summary", {})
    return {
        "val_auc": float(r.get("val_auc_mean", np.nan)),
        "val_recall": float(r.get("val_recall_mean", np.nan)),
        "test_auc": float(r.get("test_auc_mean", np.nan)),
        "test_recall": float(r.get("test_recall_mean", np.nan)),
        "test_precision": float(r.get("test_precision_mean", np.nan)),
        "test_map100": float(r.get("test_map100_mean", np.nan)),
        "test_map200": float(r.get("test_map200_mean", np.nan)),
        "test_f1": float(r.get("test_f1_mean", np.nan)),
    }


def baseline_smooth_v1(run_id: str) -> dict[str, float]:
    p = Path(config.result_dir) / f"run_{run_id}" / f"smooth_summary_{run_id}.json"
    if not p.exists():
        return {
            "val_auc": float("nan"),
            "val_recall": float("nan"),
            "test_auc": float("nan"),
            "test_recall": float("nan"),
            "test_precision": float("nan"),
            "test_map100": float("nan"),
            "test_map200": float("nan"),
            "test_f1": float("nan"),
        }
    d = load_json(p).get("summary", {})
    return {
        "val_auc": float(d.get("val_auc", {}).get("mean", np.nan)),
        "val_recall": float(d.get("val_recall", {}).get("mean", np.nan)),
        "test_auc": float(d.get("test_auc", {}).get("mean", np.nan)),
        "test_recall": float(d.get("test_recall", {}).get("mean", np.nan)),
        "test_precision": float(d.get("test_precision", {}).get("mean", np.nan)),
        "test_map100": float(d.get("test_map100", {}).get("mean", np.nan)),
        "test_map200": float(d.get("test_map200", {}).get("mean", np.nan)),
        "test_f1": float(d.get("test_f1", {}).get("mean", np.nan)),
    }


def build_stage_a_configs(smoke: bool) -> list[dict[str, Any]]:
    configs: list[dict[str, Any]] = []
    for lr, warmup, patience in itertools.product(
        config.smooth_v2_stage_a_lrs,
        config.smooth_v2_stage_a_warmups,
        config.smooth_v2_stage_a_patience,
    ):
        cfg = {
            "stage": "A",
            "lr": float(lr),
            "warmup": int(warmup),
            "patience": int(patience),
            "max_epochs": int(config.smooth_v2_stage_a_max_epochs),
            "pos_weight_scale": 1.0,
        }
        cfg["config_id"] = config_id(cfg["stage"], cfg["lr"], cfg["warmup"], cfg["patience"], cfg["pos_weight_scale"])
        configs.append(cfg)
    if smoke:
        return configs[:1]
    return configs


def build_stage_b_configs(top_stage_a_cfgs: list[dict[str, Any]], smoke: bool) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    seen: set[str] = set()
    for base in top_stage_a_cfgs:
        for scale in config.smooth_v2_stage_b_pos_weight_scales:
            cfg = {
                "stage": "B",
                "lr": float(base["lr"]),
                "warmup": int(base["warmup"]),
                "patience": int(base["patience"]),
                "max_epochs": int(config.smooth_v2_stage_a_max_epochs),
                "pos_weight_scale": float(scale),
            }
            cfg["config_id"] = config_id(cfg["stage"], cfg["lr"], cfg["warmup"], cfg["patience"], cfg["pos_weight_scale"])
            if cfg["config_id"] in seen:
                continue
            seen.add(cfg["config_id"])
            out.append(cfg)
    if smoke:
        return out[:1]
    return out


def aggregate_by_config(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[str(row["config_id"])].append(row)

    summaries: list[dict[str, Any]] = []
    for cid, rs in grouped.items():
        head = rs[0]
        summary = {
            "config_id": cid,
            "stage": head["stage"],
            "lr": float(head["lr"]),
            "warmup": int(head["warmup"]),
            "patience": int(head["patience"]),
            "max_epochs": int(head["max_epochs"]),
            "pos_weight_scale": float(head["pos_weight_scale"]),
            "num_seeds": int(len(rs)),
            "val_auc_mean": float(np.mean([float(x["val_auc"]) for x in rs])),
            "val_recall_mean": float(np.mean([float(x["val_recall"]) for x in rs])),
            "val_precision_mean": float(np.mean([float(x["val_precision"]) for x in rs])),
            "val_map100_mean": float(np.mean([float(x["val_map100"]) for x in rs])),
            "val_map200_mean": float(np.mean([float(x["val_map200"]) for x in rs])),
            "test_auc_mean": float(np.mean([float(x["test_auc"]) for x in rs])),
            "test_recall_mean": float(np.mean([float(x["test_recall"]) for x in rs])),
            "test_precision_mean": float(np.mean([float(x["test_precision"]) for x in rs])),
            "test_map100_mean": float(np.mean([float(x["test_map100"]) for x in rs])),
            "test_map200_mean": float(np.mean([float(x["test_map200"]) for x in rs])),
            "test_f1_mean": float(np.mean([float(x["test_f1"]) for x in rs])),
        }
        summaries.append(summary)

    summaries.sort(
        key=lambda x: selection_tuple(
            {
                "auc": x["val_auc_mean"],
                "recall": x["val_recall_mean"],
                "map100": x["val_map100_mean"],
                "map200": x["val_map200_mean"],
                "precision": x["val_precision_mean"],
            }
        ),
        reverse=True,
    )
    return summaries


def summarize_primary_metrics(rows: list[dict[str, Any]]) -> dict[str, dict[str, float]]:
    keys = [
        "val_auc",
        "val_recall",
        "val_precision",
        "val_map100",
        "val_map200",
        "test_auc",
        "test_recall",
        "test_precision",
        "test_map100",
        "test_map200",
        "test_f1",
    ]
    return summarize_metrics(rows, keys)


def run_trial(
    dataset,
    params: dict[str, int | float],
    train_ratio: float,
    seed: int,
    trial_cfg: dict[str, Any],
    run_id: str,
    run_dir: Path,
    checkpoint_root: Path,
) -> dict[str, Any]:
    split = split_dataset_for_ratio(
        dataset=dataset,
        train_ratio=train_ratio,
        val_ratio_in_train=config.val_ratio_in_train,
        batch_size=config.batch_size,
        random_state=int(seed),
        num_workers=config.num_workers,
    )

    model = build_model(split, params).to(config.device)

    r_tag = ratio_tag(train_ratio)
    p_tag = param_tag(params)
    cfg_id = str(trial_cfg["config_id"])

    ckpt = checkpoint_root / f"wdcnn_{run_id}_{r_tag}_{p_tag}_{cfg_id}_seed{int(seed)}_smoothv2.pth"
    train_out = train_wdcnn_model(
        model=model,
        train_loader=split.train_loader,
        val_loader=split.val_loader,
        device=config.device,
        lr=float(trial_cfg["lr"]),
        weight_decay=config.weight_decay,
        max_epochs=int(trial_cfg["max_epochs"]),
        early_stop_patience=int(trial_cfg["patience"]),
        scheduler_factor=config.lr_scheduler_factor,
        scheduler_patience=config.lr_scheduler_patience,
        min_lr=config.min_lr,
        checkpoint_path=str(ckpt),
        threshold_metric=config.threshold_metric,
        fixed_val_threshold=None,
        use_cosine_schedule=True,
        warmup_epochs=int(trial_cfg["warmup"]),
        grad_clip_norm=float(config.smooth_grad_clip_norm),
        pos_weight_scale=float(trial_cfg["pos_weight_scale"]),
        precision_floor=float(config.smooth_v2_precision_floor),
    )

    test_metrics, y_test, p_test = evaluate_wdcnn_model(
        train_out.model,
        split.test_loader,
        device=config.device,
        threshold=train_out.best_threshold,
    )

    tag = f"{trial_cfg['stage']}_{cfg_id}_{r_tag}_{p_tag}_seed{int(seed)}_{run_id}"
    history_path = run_dir / f"history_{tag}.png"
    roc_path = run_dir / f"roc_{tag}.png"
    pr_path = run_dir / f"pr_{tag}.png"
    topn_path = run_dir / f"topn_{tag}.png"
    metrics_json_path = run_dir / f"metrics_{tag}.json"
    metrics_txt_path = run_dir / f"metrics_{tag}.txt"

    plot_training_history(train_out.history, str(history_path))
    plot_roc_pr_curves(y_test, p_test, str(roc_path), str(pr_path))
    plot_topn_precision_curve(y_test, p_test, str(topn_path), max_n=200)

    payload = {
        "run_id": run_id,
        "stage": str(trial_cfg["stage"]),
        "config_id": cfg_id,
        "train_ratio": float(train_ratio),
        "seed": int(seed),
        "params": params,
        "search_config": trial_cfg,
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
    save_json(payload, str(metrics_json_path))
    save_metrics_text(payload, str(metrics_txt_path))

    row = {
        "run_id": run_id,
        "stage": str(trial_cfg["stage"]),
        "config_id": cfg_id,
        "train_ratio": float(train_ratio),
        "seed": int(seed),
        "alpha": int(params["alpha"]),
        "beta": int(params["beta"]),
        "gamma": int(params["gamma"]),
        "r_layers": int(params["r_layers"]),
        "lr": float(trial_cfg["lr"]),
        "warmup": int(trial_cfg["warmup"]),
        "patience": int(trial_cfg["patience"]),
        "max_epochs": int(trial_cfg["max_epochs"]),
        "pos_weight_scale": float(trial_cfg["pos_weight_scale"]),
        "precision_floor": float(config.smooth_v2_precision_floor),
        "best_epoch": int(train_out.best_epoch),
        "best_threshold": float(train_out.best_threshold),
        **{f"val_{k}": float(v) for k, v in train_out.best_val_metrics.items()},
        **{f"test_{k}": float(v) for k, v in test_metrics.items()},
        "metrics_json": str(metrics_json_path),
        "metrics_text": str(metrics_txt_path),
        "checkpoint": str(ckpt),
    }
    return row


def metric_delta(new: dict[str, float], old: dict[str, float]) -> dict[str, float]:
    out: dict[str, float] = {}
    for k in new:
        nv = float(new.get(k, np.nan))
        ov = float(old.get(k, np.nan))
        out[k] = float(nv - ov) if not np.isnan(nv) and not np.isnan(ov) else float("nan")
    return out


def main() -> None:
    args = parse_args()
    smoke = bool(args.smoke)

    config.make_dirs()
    run_threshold_selector_selftest()

    source_run_id = str(config.smooth_target_run_id or config.resume_run_id).strip()
    train_ratio, params, global_best_payload = load_global_best(source_run_id)

    # Guardrail: keep this round fixed to best-group protocol.
    if abs(train_ratio - float(config.smooth_v2_ratio)) > 1e-9:
        print(f"[WARN] global_best ratio={train_ratio:.2f} != configured smooth_v2_ratio={config.smooth_v2_ratio:.2f}; using global_best.")
    if param_tag(params) != str(config.smooth_v2_param_tag):
        print(f"[WARN] global_best params={param_tag(params)} != configured smooth_v2_param_tag={config.smooth_v2_param_tag}; using global_best.")

    result_root = Path(config.result_dir) / str(config.smooth_v2_result_subdir).strip()
    result_root.mkdir(parents=True, exist_ok=True)
    run_id, run_dir = create_run_id(str(result_root))

    checkpoint_root = Path(config.checkpoint_dir) / str(config.smooth_v2_result_subdir).strip()
    checkpoint_root.mkdir(parents=True, exist_ok=True)

    seeds = list(config.seed_list[:1] if smoke else config.seed_list)
    print(f"Device: {config.device}")
    print(f"Source run: {source_run_id}")
    print(f"Best group: ratio={train_ratio:.2f}, params={param_tag(params)}")
    print(f"Mode: {'SMOKE' if smoke else 'FULL'}")
    print(f"Run ID: {run_id}")
    print(f"Run Dir: {run_dir}")
    print(f"Seeds: {seeds}")

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

    # Stage A
    stage_a_cfgs = build_stage_a_configs(smoke=smoke)
    stage_a_rows: list[dict[str, Any]] = []
    print(f"\n[Stage A] configs={len(stage_a_cfgs)} seeds={len(seeds)} total_runs={len(stage_a_cfgs) * len(seeds)}")
    for idx, cfg in enumerate(stage_a_cfgs, start=1):
        print(f"[Stage A {idx:03d}/{len(stage_a_cfgs):03d}] {cfg['config_id']}")
        for seed in seeds:
            row = run_trial(
                dataset=dataset,
                params=params,
                train_ratio=train_ratio,
                seed=int(seed),
                trial_cfg=cfg,
                run_id=run_id,
                run_dir=run_dir,
                checkpoint_root=checkpoint_root,
            )
            stage_a_rows.append(row)

    stage_a_summary = aggregate_by_config(stage_a_rows)
    top_k = 1 if smoke else int(config.smooth_v2_stage_a_top_k)
    top_stage_a = stage_a_summary[:top_k]
    print(f"[Stage A] Top-{len(top_stage_a)} selected for Stage B")
    for i, r in enumerate(top_stage_a, start=1):
        print(
            f"  {i}. {r['config_id']} | val_auc={r['val_auc_mean']:.4f} "
            f"val_recall={r['val_recall_mean']:.4f} val_precision={r['val_precision_mean']:.4f}"
        )

    # Stage B
    stage_b_cfgs = build_stage_b_configs(top_stage_a, smoke=smoke)
    stage_b_rows: list[dict[str, Any]] = []
    print(f"\n[Stage B] configs={len(stage_b_cfgs)} seeds={len(seeds)} total_runs={len(stage_b_cfgs) * len(seeds)}")
    for idx, cfg in enumerate(stage_b_cfgs, start=1):
        print(f"[Stage B {idx:03d}/{len(stage_b_cfgs):03d}] {cfg['config_id']}")
        for seed in seeds:
            row = run_trial(
                dataset=dataset,
                params=params,
                train_ratio=train_ratio,
                seed=int(seed),
                trial_cfg=cfg,
                run_id=run_id,
                run_dir=run_dir,
                checkpoint_root=checkpoint_root,
            )
            stage_b_rows.append(row)

    stage_b_summary = aggregate_by_config(stage_b_rows)
    final_summary_source = stage_b_summary if stage_b_summary else stage_a_summary
    if not final_summary_source:
        raise RuntimeError("No completed trials found.")

    best_cfg = final_summary_source[0]
    best_cfg_id = str(best_cfg["config_id"])
    final_seed_rows = [r for r in (stage_b_rows if stage_b_rows else stage_a_rows) if str(r["config_id"]) == best_cfg_id]
    final_seed_rows.sort(
        key=lambda r: selection_tuple(
            {
                "auc": float(r["val_auc"]),
                "recall": float(r["val_recall"]),
                "map100": float(r["val_map100"]),
                "map200": float(r["val_map200"]),
                "precision": float(r["val_precision"]),
            }
        ),
        reverse=True,
    )

    final_stats = summarize_primary_metrics(final_seed_rows)
    final_means = {
        "val_auc": float(final_stats["val_auc"]["mean"]),
        "val_recall": float(final_stats["val_recall"]["mean"]),
        "test_auc": float(final_stats["test_auc"]["mean"]),
        "test_recall": float(final_stats["test_recall"]["mean"]),
        "test_precision": float(final_stats["test_precision"]["mean"]),
        "test_map100": float(final_stats["test_map100"]["mean"]),
        "test_map200": float(final_stats["test_map200"]["mean"]),
        "test_f1": float(final_stats["test_f1"]["mean"]),
    }

    gb = baseline_global_best(global_best_payload)
    sv1 = baseline_smooth_v1(str(config.smooth_v1_run_id))
    comparison = {
        "global_best": gb,
        "smooth_v1": sv1,
        "smooth_v2_best": final_means,
        "delta_vs_global_best": metric_delta(final_means, gb),
        "delta_vs_smooth_v1": metric_delta(final_means, sv1),
    }

    # Save artifacts
    stage_a_records_csv = run_dir / f"stage_a_records_{run_id}.csv"
    stage_b_records_csv = run_dir / f"stage_b_records_{run_id}.csv"
    stage_a_summary_csv = run_dir / f"stage_a_summary_{run_id}.csv"
    stage_b_summary_csv = run_dir / f"stage_b_summary_{run_id}.csv"
    final_rows_csv = run_dir / f"final_best_rows_{run_id}.csv"
    summary_json = run_dir / f"smooth_v2_summary_{run_id}.json"
    summary_txt = run_dir / f"smooth_v2_summary_{run_id}.txt"

    save_records_csv(stage_a_rows, str(stage_a_records_csv))
    save_records_csv(stage_b_rows, str(stage_b_records_csv))
    save_records_csv(stage_a_summary, str(stage_a_summary_csv))
    save_records_csv(stage_b_summary, str(stage_b_summary_csv))
    save_records_csv(final_seed_rows, str(final_rows_csv))

    payload = {
        "run_id": run_id,
        "mode": "smoke" if smoke else "full",
        "source_best_run_id": source_run_id,
        "selected_group": {
            "train_ratio": float(train_ratio),
            "param_tag": param_tag(params),
            "params": params,
        },
        "precision_floor": float(config.smooth_v2_precision_floor),
        "stage_a_search_space": {
            "lrs": config.smooth_v2_stage_a_lrs,
            "warmups": config.smooth_v2_stage_a_warmups,
            "patience": config.smooth_v2_stage_a_patience,
            "max_epochs": int(config.smooth_v2_stage_a_max_epochs),
        },
        "stage_b_search_space": {
            "pos_weight_scales": config.smooth_v2_stage_b_pos_weight_scales,
            "base_top_k_from_stage_a": int(top_k),
        },
        "stage_a_records_csv": str(stage_a_records_csv),
        "stage_b_records_csv": str(stage_b_records_csv),
        "stage_a_summary_csv": str(stage_a_summary_csv),
        "stage_b_summary_csv": str(stage_b_summary_csv),
        "final_best_rows_csv": str(final_rows_csv),
        "stage_a_top_configs": top_stage_a,
        "stage_b_top_config": best_cfg,
        "final_best_seed_rows": final_seed_rows,
        "final_best_summary": final_stats,
        "baseline_comparison": comparison,
    }
    save_json(payload, str(summary_json))
    save_metrics_text(payload, str(summary_txt))

    print("\n" + "=" * 88)
    print("Smooth-v2 search finished")
    print(f"Best config: {best_cfg_id}")
    print(f"Best val_auc={best_cfg['val_auc_mean']:.4f}, val_recall={best_cfg['val_recall_mean']:.4f}")
    print(f"Summary JSON: {summary_json}")
    print("=" * 88)


if __name__ == "__main__":
    main()
