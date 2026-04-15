from __future__ import annotations

import csv
import json
import re
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import torch

from config import config
from data_process import preprocess_for_wdcnn, split_dataset_for_ratio
from engine import (
    compute_binary_metrics,
    create_run_id,
    plot_roc_pr_curves,
    plot_topn_precision_curve,
    plot_training_history,
    save_json,
    save_metrics_text,
    save_records_csv,
    select_best_threshold,
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


CKPT_PATTERN = re.compile(
    r"^wdcnn_"
    r"(?P<run>\d{8}_\d{6}_v\d{3})_"
    r"(?P<ratio>r\d{2})_"
    r"(?P<ptag>a\d+_b\d+_g\d+_r\d+)_"
    r"seed(?P<seed>\d+)_"
    r"(?P<stage>coarse|fine)\.pth$"
)


INT_KEYS = {
    "seed",
    "alpha",
    "beta",
    "gamma",
    "r_layers",
    "best_epoch",
}


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


def run_and_dir() -> tuple[str, Path]:
    resume_id = str(config.resume_run_id or "").strip()
    if resume_id:
        run_dir = Path(config.result_dir) / f"run_{resume_id}"
        run_dir.mkdir(parents=True, exist_ok=True)
        return resume_id, run_dir
    run_id, run_dir = create_run_id(config.result_dir)
    return run_id, run_dir


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


def checkpoint_path(
    checkpoint_dir: Path,
    run_id: str,
    r_tag: str,
    p_tag: str,
    seed: int,
    stage: str,
) -> Path:
    return checkpoint_dir / f"wdcnn_{run_id}_{r_tag}_{p_tag}_seed{seed}_{stage}.pth"


def fine_artifact_paths(run_dir: Path, run_id: str, r_tag: str, p_tag: str, seed: int) -> dict[str, Path]:
    tag = f"{r_tag}_{p_tag}_seed{seed}_{run_id}"
    return {
        "history": run_dir / f"history_{tag}.png",
        "roc": run_dir / f"roc_{tag}.png",
        "pr": run_dir / f"pr_{tag}.png",
        "topn": run_dir / f"topn_{tag}.png",
        "metrics_json": run_dir / f"metrics_{tag}.json",
        "metrics_text": run_dir / f"metrics_{tag}.txt",
    }


def task_key(stage: str, train_ratio: float, params: dict[str, int | float], seed: int) -> tuple[str, str, int, str]:
    return (ratio_tag(train_ratio), param_tag(params), int(seed), stage)


def _to_number_if_needed(key: str, value: Any) -> Any:
    if value is None:
        return value
    if isinstance(value, (int, float)):
        return value
    if isinstance(value, str):
        s = value.strip()
        if s == "":
            return value
        if key in INT_KEYS:
            try:
                return int(float(s))
            except Exception:
                return value
        if key == "train_ratio" or key == "best_threshold" or key.startswith("val_") or key.startswith("test_"):
            try:
                return float(s)
            except Exception:
                return value
    return value


def normalize_row_types(row: dict[str, Any]) -> dict[str, Any]:
    return {k: _to_number_if_needed(k, v) for k, v in row.items()}


def load_csv_records(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    out: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for raw in reader:
            out.append(normalize_row_types(dict(raw)))
    return out


def row_to_task_key(stage: str, row: dict[str, Any]) -> tuple[str, str, int, str]:
    params = {
        "alpha": int(row["alpha"]),
        "beta": int(row["beta"]),
        "gamma": int(row["gamma"]),
        "r_layers": int(row["r_layers"]),
    }
    return task_key(stage, float(row["train_ratio"]), params, int(row["seed"]))


def upsert_row(
    rows: list[dict[str, Any]],
    row_index: dict[tuple[str, str, int, str], int],
    key: tuple[str, str, int, str],
    row: dict[str, Any],
) -> None:
    if key in row_index:
        rows[row_index[key]] = row
    else:
        row_index[key] = len(rows)
        rows.append(row)


def parse_checkpoint_keys(checkpoint_dir: Path, run_id: str) -> dict[str, set[tuple[str, str, int, str]]]:
    by_stage: dict[str, set[tuple[str, str, int, str]]] = {"coarse": set(), "fine": set()}
    for p in checkpoint_dir.glob(f"wdcnn_{run_id}_*.pth"):
        m = CKPT_PATTERN.match(p.name)
        if not m:
            continue
        if m.group("run") != run_id:
            continue
        key = (
            m.group("ratio"),
            m.group("ptag"),
            int(m.group("seed")),
            m.group("stage"),
        )
        by_stage[m.group("stage")].add(key)
    return by_stage


@torch.no_grad()
def predict_probs(model: torch.nn.Module, dataloader, device: str) -> tuple[np.ndarray, np.ndarray]:
    model.eval()
    labels: list[np.ndarray] = []
    probs: list[np.ndarray] = []

    for x1, x2, y in dataloader:
        x1 = x1.to(device)
        x2 = x2.to(device)
        y = y.to(device)
        logits, _ = model(x1, x2)
        p = torch.sigmoid(logits)
        labels.append(y.detach().cpu().numpy())
        probs.append(p.detach().cpu().numpy())

    y_true = np.concatenate(labels, axis=0).astype(np.int64)
    y_prob = np.concatenate(probs, axis=0).astype(np.float32)
    return y_true, y_prob


def evaluate_on_loader(
    model: torch.nn.Module,
    dataloader,
    device: str,
    threshold: float,
) -> tuple[dict[str, float], np.ndarray, np.ndarray]:
    y_true, y_prob = predict_probs(model, dataloader, device)
    metrics = compute_binary_metrics(y_true, y_prob, threshold=threshold)
    return metrics, y_true, y_prob


def evaluate_with_best_threshold(
    model: torch.nn.Module,
    dataloader,
    device: str,
    threshold_metric: str,
) -> tuple[dict[str, float], np.ndarray, np.ndarray, float]:
    y_true, y_prob = predict_probs(model, dataloader, device)
    threshold = select_best_threshold(y_true, y_prob, metric=threshold_metric)
    metrics = compute_binary_metrics(y_true, y_prob, threshold=threshold)
    return metrics, y_true, y_prob, float(threshold)


def make_coarse_row(
    run_id: str,
    train_ratio: float,
    seed: int,
    params: dict[str, int | float],
    val_metrics: dict[str, float],
    best_threshold: float,
    best_epoch: int,
    ckpt_path: Path,
    resume_source: str,
) -> dict[str, Any]:
    return {
        "run_id": run_id,
        "stage": "coarse",
        "train_ratio": float(train_ratio),
        "seed": int(seed),
        "alpha": int(params["alpha"]),
        "beta": int(params["beta"]),
        "gamma": int(params["gamma"]),
        "r_layers": int(params["r_layers"]),
        "best_epoch": int(best_epoch),
        "best_threshold": float(best_threshold),
        "checkpoint": str(ckpt_path),
        "resume_source": resume_source,
        **{f"val_{k}": float(v) for k, v in val_metrics.items()},
    }


def make_fine_row(
    run_id: str,
    train_ratio: float,
    seed: int,
    params: dict[str, int | float],
    best_epoch: int,
    best_threshold: float,
    val_metrics: dict[str, float],
    test_metrics: dict[str, float],
    metrics_json_path: Path,
    metrics_text_path: Path,
    ckpt_path: Path,
    resume_source: str,
) -> dict[str, Any]:
    row: dict[str, Any] = {
        "run_id": run_id,
        "stage": "fine",
        "train_ratio": float(train_ratio),
        "seed": int(seed),
        "alpha": int(params["alpha"]),
        "beta": int(params["beta"]),
        "gamma": int(params["gamma"]),
        "r_layers": int(params["r_layers"]),
        "best_epoch": int(best_epoch),
        "best_threshold": float(best_threshold),
        "metrics_json": str(metrics_json_path),
        "metrics_text": str(metrics_text_path),
        "checkpoint": str(ckpt_path),
        "resume_source": resume_source,
    }
    row.update({f"val_{k}": float(v) for k, v in val_metrics.items()})
    row.update({f"test_{k}": float(v) for k, v in test_metrics.items()})
    return row


def read_fine_row_from_metrics_json(path: Path, expected_run_id: str) -> dict[str, Any] | None:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None

    if str(payload.get("run_id")) != expected_run_id:
        return None

    params = payload.get("params", {})
    if not params:
        return None

    metrics_text = path.with_suffix(".txt")
    artifacts = payload.get("artifacts", {})
    ckpt = artifacts.get("checkpoint", "")
    return make_fine_row(
        run_id=expected_run_id,
        train_ratio=float(payload.get("train_ratio", np.nan)),
        seed=int(payload.get("seed", -1)),
        params={
            "alpha": int(params.get("alpha")),
            "beta": int(params.get("beta")),
            "gamma": int(params.get("gamma")),
            "r_layers": int(params.get("r_layers")),
            "dropout": float(params.get("dropout", config.dropout)),
        },
        best_epoch=int(payload.get("best_epoch", -1)),
        best_threshold=float(payload.get("best_threshold", 0.5)),
        val_metrics={k: float(v) for k, v in payload.get("val_metrics", {}).items()},
        test_metrics={k: float(v) for k, v in payload.get("test_metrics", {}).items()},
        metrics_json_path=path,
        metrics_text_path=metrics_text,
        ckpt_path=Path(str(ckpt)) if ckpt else Path(""),
        resume_source="metrics_json",
    )


def load_fine_rows_from_run_dir(run_dir: Path, run_id: str) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for p in run_dir.glob(f"metrics_*_{run_id}.json"):
        row = read_fine_row_from_metrics_json(p, expected_run_id=run_id)
        if row is not None:
            out.append(row)
    return out


def maybe_load_state_dict(model: torch.nn.Module, ckpt_path: Path, device: str) -> bool:
    try:
        state = torch.load(str(ckpt_path), map_location=device)
        if isinstance(state, dict) and "state_dict" in state:
            state = state["state_dict"]
        model.load_state_dict(state, strict=True)
        return True
    except Exception as e:
        print(f"[WARN] checkpoint load failed: {ckpt_path} | {e}")
        return False


def write_fine_payload(
    run_id: str,
    train_ratio: float,
    seed: int,
    params: dict[str, int | float],
    best_epoch: int,
    best_threshold: float,
    val_metrics: dict[str, float],
    test_metrics: dict[str, float],
    ckpt_path: Path,
    artifacts: dict[str, Path],
) -> None:
    payload = {
        "run_id": run_id,
        "stage": "fine",
        "train_ratio": float(train_ratio),
        "seed": int(seed),
        "params": {
            "alpha": int(params["alpha"]),
            "beta": int(params["beta"]),
            "gamma": int(params["gamma"]),
            "r_layers": int(params["r_layers"]),
            "dropout": float(params["dropout"]),
        },
        "best_epoch": int(best_epoch),
        "best_threshold": float(best_threshold),
        "val_metrics": {k: float(v) for k, v in val_metrics.items()},
        "test_metrics": {k: float(v) for k, v in test_metrics.items()},
        "artifacts": {
            "checkpoint": str(ckpt_path),
            "history": str(artifacts["history"]) if artifacts.get("history") else "",
            "roc": str(artifacts["roc"]),
            "pr": str(artifacts["pr"]),
            "topn": str(artifacts["topn"]),
        },
    }
    save_json(payload, str(artifacts["metrics_json"]))
    save_metrics_text(payload, str(artifacts["metrics_text"]))


def save_incremental_outputs(
    run_id: str,
    run_dir: Path,
    coarse_records: list[dict[str, Any]],
    fine_records: list[dict[str, Any]],
    ratio_summary_records: list[dict[str, Any]],
    expected_coarse_tasks: int,
    expected_fine_tasks: int,
) -> None:
    coarse_csv = run_dir / f"coarse_records_{run_id}.csv"
    fine_csv = run_dir / f"fine_records_{run_id}.csv"
    ratio_csv = run_dir / f"ratio_summary_{run_id}.csv"
    resume_state_json = run_dir / f"resume_state_{run_id}.json"

    save_records_csv(coarse_records, str(coarse_csv))
    save_records_csv(fine_records, str(fine_csv))
    save_records_csv(ratio_summary_records, str(ratio_csv))

    coarse_done = sum(1 for r in coarse_records if str(r.get("run_id", "")) == run_id)
    fine_done = sum(1 for r in fine_records if str(r.get("run_id", "")) == run_id)

    state = {
        "run_id": run_id,
        "updated_at": datetime.now().isoformat(timespec="seconds"),
        "expected_tasks": {
            "coarse": int(expected_coarse_tasks),
            "fine": int(expected_fine_tasks),
            "total": int(expected_coarse_tasks + expected_fine_tasks),
        },
        "completed_records": {
            "coarse": int(coarse_done),
            "fine": int(fine_done),
            "total": int(coarse_done + fine_done),
        },
        "pending_records": {
            "coarse": int(max(expected_coarse_tasks - coarse_done, 0)),
            "fine": int(max(expected_fine_tasks - fine_done, 0)),
            "total": int(max(expected_coarse_tasks - coarse_done, 0) + max(expected_fine_tasks - fine_done, 0)),
        },
        "csv_paths": {
            "coarse": str(coarse_csv),
            "fine": str(fine_csv),
            "ratio_summary": str(ratio_csv),
        },
    }
    save_json(state, str(resume_state_json))


def format_metric(value: Any) -> str:
    try:
        return f"{float(value):.4f}"
    except Exception:
        return "nan"


def select_top_params_from_coarse(rows: list[dict[str, Any]], top_k: int) -> list[dict[str, int | float]]:
    ordered = sorted(
        rows,
        key=lambda x: selection_tuple(
            {
                "auc": x.get("val_auc", float("nan")),
                "recall": x.get("val_recall", float("nan")),
                "map100": x.get("val_map100", float("nan")),
                "map200": x.get("val_map200", float("nan")),
                "precision": x.get("val_precision", float("nan")),
            }
        ),
        reverse=True,
    )
    return [
        {
            "alpha": int(r["alpha"]),
            "beta": int(r["beta"]),
            "gamma": int(r["gamma"]),
            "r_layers": int(r["r_layers"]),
            "dropout": config.dropout,
        }
        for r in ordered[:top_k]
    ]


def best_param_by_val_mean(param_to_rows: dict[str, list[dict[str, Any]]]) -> tuple[str | None, list[dict[str, Any]], dict[str, float]]:
    best_tag: str | None = None
    best_rows: list[dict[str, Any]] = []
    best_mean: dict[str, float] = {}
    best_score = selection_tuple({})

    for p_tag, rows in param_to_rows.items():
        if not rows:
            continue
        mean_val = {
            "auc": float(np.mean([float(r.get("val_auc", np.nan)) for r in rows])),
            "recall": float(np.mean([float(r.get("val_recall", np.nan)) for r in rows])),
            "map100": float(np.mean([float(r.get("val_map100", np.nan)) for r in rows])),
            "map200": float(np.mean([float(r.get("val_map200", np.nan)) for r in rows])),
            "precision": float(np.mean([float(r.get("val_precision", np.nan)) for r in rows])),
        }
        score = selection_tuple(mean_val)
        if score > best_score:
            best_score = score
            best_tag = p_tag
            best_rows = rows
            best_mean = mean_val
    return best_tag, best_rows, best_mean


def upsert_ratio_summary(rows: list[dict[str, Any]], row: dict[str, Any]) -> None:
    tr = float(row["train_ratio"])
    for idx, old in enumerate(rows):
        if float(old.get("train_ratio", np.nan)) == tr:
            rows[idx] = row
            return
    rows.append(row)


def main() -> None:
    config.make_dirs()
    run_id, run_dir = run_and_dir()
    checkpoint_dir = Path(config.checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    coarse_csv = run_dir / f"coarse_records_{run_id}.csv"
    fine_csv = run_dir / f"fine_records_{run_id}.csv"
    ratio_csv = run_dir / f"ratio_summary_{run_id}.csv"

    coarse_records = load_csv_records(coarse_csv)
    fine_records = load_csv_records(fine_csv)
    ratio_summary_records = load_csv_records(ratio_csv)

    coarse_index = {row_to_task_key("coarse", r): i for i, r in enumerate(coarse_records)}
    fine_index = {row_to_task_key("fine", r): i for i, r in enumerate(fine_records)}

    # Supplement fine records from existing metrics json artifacts.
    for row in load_fine_rows_from_run_dir(run_dir, run_id):
        key = row_to_task_key("fine", row)
        upsert_row(fine_records, fine_index, key, row)

    coarse_grid = config.coarse_param_grid()
    primary_seed = int(config.seed_list[0])
    expected_coarse_tasks = len(config.train_ratios) * len(coarse_grid)
    expected_fine_tasks = len(config.train_ratios) * config.refine_top_k * len(config.seed_list)

    ckpt_keys = parse_checkpoint_keys(checkpoint_dir, run_id)
    coarse_ckpt_count = len(ckpt_keys["coarse"])
    fine_ckpt_count = len(ckpt_keys["fine"])

    print(f"Device: {config.device}")
    print(f"Run ID: {run_id}")
    print(f"Run Dir: {run_dir}")
    print(
        f"[Resume scan] coarse done={coarse_ckpt_count}/{expected_coarse_tasks}, "
        f"pending={max(expected_coarse_tasks - coarse_ckpt_count, 0)}"
    )
    print(
        f"[Resume scan] fine done={fine_ckpt_count}/{expected_fine_tasks}, "
        f"pending={max(expected_fine_tasks - fine_ckpt_count, 0)}"
    )
    print(
        f"[Resume scan] total pending={max(expected_coarse_tasks - coarse_ckpt_count, 0) + max(expected_fine_tasks - fine_ckpt_count, 0)}"
    )

    save_incremental_outputs(
        run_id=run_id,
        run_dir=run_dir,
        coarse_records=coarse_records,
        fine_records=fine_records,
        ratio_summary_records=ratio_summary_records,
        expected_coarse_tasks=expected_coarse_tasks,
        expected_fine_tasks=expected_fine_tasks,
    )

    if bool(config.resume_scan_only):
        print("Scan-only mode enabled, exiting without training.")
        return

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

    for tr in config.train_ratios:
        r_tag = ratio_tag(tr)
        print("\n" + "=" * 88)
        print(f"[Ratio {tr:.2f}] Stage-1 coarse search with resume")
        print("=" * 88)

        coarse_split = split_dataset_for_ratio(
            dataset=dataset,
            train_ratio=tr,
            val_ratio_in_train=config.val_ratio_in_train,
            batch_size=config.batch_size,
            random_state=primary_seed,
            num_workers=config.num_workers,
        )

        ratio_coarse_rows: list[dict[str, Any]] = []

        for i, params in enumerate(coarse_grid, start=1):
            p_tag = param_tag(params)
            key = task_key("coarse", tr, params, primary_seed)
            ckpt_path = checkpoint_path(checkpoint_dir, run_id, r_tag, p_tag, primary_seed, "coarse")

            print(f"[Coarse {i:03d}/{len(coarse_grid):03d}] ratio={tr:.2f} params={p_tag}")

            if bool(config.resume_skip_existing) and key in coarse_index:
                print(f"SKIP existing coarse record: {ckpt_path.name}")
                row = coarse_records[coarse_index[key]]
                ratio_coarse_rows.append(row)
                continue

            row: dict[str, Any] | None = None
            reused = False

            if bool(config.resume_skip_existing) and ckpt_path.exists():
                model = build_model(coarse_split, params).to(config.device)
                if maybe_load_state_dict(model, ckpt_path, config.device):
                    val_metrics, _, _, best_thr = evaluate_with_best_threshold(
                        model=model,
                        dataloader=coarse_split.val_loader,
                        device=config.device,
                        threshold_metric=config.threshold_metric,
                    )
                    row = make_coarse_row(
                        run_id=run_id,
                        train_ratio=tr,
                        seed=primary_seed,
                        params=params,
                        val_metrics=val_metrics,
                        best_threshold=best_thr,
                        best_epoch=-1,
                        ckpt_path=ckpt_path,
                        resume_source="checkpoint_rebuild",
                    )
                    reused = True
                    print(
                        "SKIP existing checkpoint -> rebuild val metrics | "
                        f"val_auc={format_metric(val_metrics.get('auc'))} "
                        f"val_recall={format_metric(val_metrics.get('recall'))}"
                    )

            if row is None:
                model = build_model(coarse_split, params).to(config.device)
                train_out = train_wdcnn_model(
                    model=model,
                    train_loader=coarse_split.train_loader,
                    val_loader=coarse_split.val_loader,
                    device=config.device,
                    lr=config.lr,
                    weight_decay=config.weight_decay,
                    max_epochs=config.coarse_max_epochs,
                    early_stop_patience=config.early_stop_patience,
                    scheduler_factor=config.lr_scheduler_factor,
                    scheduler_patience=config.lr_scheduler_patience,
                    min_lr=config.min_lr,
                    checkpoint_path=str(ckpt_path),
                    threshold_metric=config.threshold_metric,
                )
                row = make_coarse_row(
                    run_id=run_id,
                    train_ratio=tr,
                    seed=primary_seed,
                    params=params,
                    val_metrics=train_out.best_val_metrics,
                    best_threshold=train_out.best_threshold,
                    best_epoch=train_out.best_epoch,
                    ckpt_path=ckpt_path,
                    resume_source="trained",
                )
                print(
                    "TRAIN coarse done | "
                    f"val_auc={format_metric(train_out.best_val_metrics.get('auc'))} "
                    f"val_recall={format_metric(train_out.best_val_metrics.get('recall'))}"
                )

            upsert_row(coarse_records, coarse_index, key, row)
            ratio_coarse_rows.append(row)

            save_incremental_outputs(
                run_id=run_id,
                run_dir=run_dir,
                coarse_records=coarse_records,
                fine_records=fine_records,
                ratio_summary_records=ratio_summary_records,
                expected_coarse_tasks=expected_coarse_tasks,
                expected_fine_tasks=expected_fine_tasks,
            )

            if reused:
                continue

        top_params = select_top_params_from_coarse(ratio_coarse_rows, config.refine_top_k)
        print(f"\n[Ratio {tr:.2f}] Stage-2 fine training on top-{len(top_params)} configs")

        param_to_seed_rows: dict[str, list[dict[str, Any]]] = defaultdict(list)

        for seed in config.seed_list:
            split = split_dataset_for_ratio(
                dataset=dataset,
                train_ratio=tr,
                val_ratio_in_train=config.val_ratio_in_train,
                batch_size=config.batch_size,
                random_state=int(seed),
                num_workers=config.num_workers,
            )

            for params in top_params:
                p_tag = param_tag(params)
                key = task_key("fine", tr, params, int(seed))
                ckpt_path = checkpoint_path(checkpoint_dir, run_id, r_tag, p_tag, int(seed), "fine")
                artifacts = fine_artifact_paths(run_dir, run_id, r_tag, p_tag, int(seed))

                if bool(config.resume_skip_existing) and key in fine_index:
                    row = fine_records[fine_index[key]]
                    print(f"SKIP existing fine record: {ckpt_path.name}")
                    param_to_seed_rows[p_tag].append(row)
                    continue

                row = None

                if bool(config.resume_skip_existing) and artifacts["metrics_json"].exists():
                    loaded = read_fine_row_from_metrics_json(artifacts["metrics_json"], expected_run_id=run_id)
                    if loaded is not None:
                        row = loaded
                        row["resume_source"] = "metrics_json"
                        print(f"SKIP existing metrics json: {artifacts['metrics_json'].name}")

                if row is None and bool(config.resume_skip_existing) and ckpt_path.exists() and bool(config.resume_rebuild_metrics):
                    model = build_model(split, params).to(config.device)
                    if maybe_load_state_dict(model, ckpt_path, config.device):
                        val_metrics, _, _, best_thr = evaluate_with_best_threshold(
                            model=model,
                            dataloader=split.val_loader,
                            device=config.device,
                            threshold_metric=config.threshold_metric,
                        )
                        test_metrics, y_test, p_test = evaluate_on_loader(
                            model=model,
                            dataloader=split.test_loader,
                            device=config.device,
                            threshold=best_thr,
                        )

                        plot_roc_pr_curves(y_test, p_test, str(artifacts["roc"]), str(artifacts["pr"]))
                        plot_topn_precision_curve(y_test, p_test, str(artifacts["topn"]), max_n=200)

                        write_fine_payload(
                            run_id=run_id,
                            train_ratio=tr,
                            seed=int(seed),
                            params=params,
                            best_epoch=-1,
                            best_threshold=best_thr,
                            val_metrics=val_metrics,
                            test_metrics=test_metrics,
                            ckpt_path=ckpt_path,
                            artifacts=artifacts,
                        )

                        row = make_fine_row(
                            run_id=run_id,
                            train_ratio=tr,
                            seed=int(seed),
                            params=params,
                            best_epoch=-1,
                            best_threshold=best_thr,
                            val_metrics=val_metrics,
                            test_metrics=test_metrics,
                            metrics_json_path=artifacts["metrics_json"],
                            metrics_text_path=artifacts["metrics_text"],
                            ckpt_path=ckpt_path,
                            resume_source="checkpoint_rebuild",
                        )
                        print(
                            "SKIP existing checkpoint -> rebuild fine metrics | "
                            f"val_auc={format_metric(val_metrics.get('auc'))} "
                            f"val_recall={format_metric(val_metrics.get('recall'))}"
                        )

                if row is None:
                    model = build_model(split, params).to(config.device)
                    train_out = train_wdcnn_model(
                        model=model,
                        train_loader=split.train_loader,
                        val_loader=split.val_loader,
                        device=config.device,
                        lr=config.lr,
                        weight_decay=config.weight_decay,
                        max_epochs=config.fine_max_epochs,
                        early_stop_patience=config.early_stop_patience,
                        scheduler_factor=config.lr_scheduler_factor,
                        scheduler_patience=config.lr_scheduler_patience,
                        min_lr=config.min_lr,
                        checkpoint_path=str(ckpt_path),
                        threshold_metric=config.threshold_metric,
                    )

                    test_metrics, y_test, p_test = evaluate_on_loader(
                        model=train_out.model,
                        dataloader=split.test_loader,
                        device=config.device,
                        threshold=train_out.best_threshold,
                    )

                    plot_training_history(train_out.history, str(artifacts["history"]))
                    plot_roc_pr_curves(y_test, p_test, str(artifacts["roc"]), str(artifacts["pr"]))
                    plot_topn_precision_curve(y_test, p_test, str(artifacts["topn"]), max_n=200)

                    write_fine_payload(
                        run_id=run_id,
                        train_ratio=tr,
                        seed=int(seed),
                        params=params,
                        best_epoch=train_out.best_epoch,
                        best_threshold=train_out.best_threshold,
                        val_metrics=train_out.best_val_metrics,
                        test_metrics=test_metrics,
                        ckpt_path=ckpt_path,
                        artifacts=artifacts,
                    )

                    row = make_fine_row(
                        run_id=run_id,
                        train_ratio=tr,
                        seed=int(seed),
                        params=params,
                        best_epoch=train_out.best_epoch,
                        best_threshold=train_out.best_threshold,
                        val_metrics=train_out.best_val_metrics,
                        test_metrics=test_metrics,
                        metrics_json_path=artifacts["metrics_json"],
                        metrics_text_path=artifacts["metrics_text"],
                        ckpt_path=ckpt_path,
                        resume_source="trained",
                    )
                    print(
                        "TRAIN fine done | "
                        f"val_auc={format_metric(train_out.best_val_metrics.get('auc'))} "
                        f"val_recall={format_metric(train_out.best_val_metrics.get('recall'))}"
                    )

                upsert_row(fine_records, fine_index, key, row)
                param_to_seed_rows[p_tag].append(row)

                save_incremental_outputs(
                    run_id=run_id,
                    run_dir=run_dir,
                    coarse_records=coarse_records,
                    fine_records=fine_records,
                    ratio_summary_records=ratio_summary_records,
                    expected_coarse_tasks=expected_coarse_tasks,
                    expected_fine_tasks=expected_fine_tasks,
                )

        best_p_tag, best_rows, best_val_mean = best_param_by_val_mean(param_to_seed_rows)

        test_metric_keys = [
            "test_auc",
            "test_recall",
            "test_precision",
            "test_map100",
            "test_map200",
            "test_f1",
        ]
        val_metric_keys = [
            "val_auc",
            "val_recall",
            "val_precision",
            "val_map100",
            "val_map200",
            "val_f1",
        ]
        test_summary = summarize_metrics(best_rows, test_metric_keys)
        val_summary = summarize_metrics(best_rows, val_metric_keys)

        ratio_summary = {
            "run_id": run_id,
            "train_ratio": float(tr),
            "best_param_tag": best_p_tag or "",
            "num_seed_runs": int(len(best_rows)),
            "val_auc_mean": float(best_val_mean.get("auc", np.nan)),
            "val_recall_mean": float(best_val_mean.get("recall", np.nan)),
            "val_map100_mean": float(best_val_mean.get("map100", np.nan)),
            "val_map200_mean": float(best_val_mean.get("map200", np.nan)),
            "val_precision_mean": float(best_val_mean.get("precision", np.nan)),
            **{f"{k}_mean": float(v["mean"]) for k, v in test_summary.items()},
            **{f"{k}_std": float(v["std"]) for k, v in test_summary.items()},
            **{f"{k}_selected_mean": float(v["mean"]) for k, v in val_summary.items()},
            **{f"{k}_selected_std": float(v["std"]) for k, v in val_summary.items()},
        }
        upsert_ratio_summary(ratio_summary_records, ratio_summary)

        save_incremental_outputs(
            run_id=run_id,
            run_dir=run_dir,
            coarse_records=coarse_records,
            fine_records=fine_records,
            ratio_summary_records=ratio_summary_records,
            expected_coarse_tasks=expected_coarse_tasks,
            expected_fine_tasks=expected_fine_tasks,
        )

        print(f"[Ratio {tr:.2f}] best config={best_p_tag}")
        print(
            f"  val_auc_mean={format_metric(ratio_summary.get('val_auc_mean'))} "
            f"val_recall_mean={format_metric(ratio_summary.get('val_recall_mean'))}"
        )
        print(
            f"  test_auc_mean={format_metric(ratio_summary.get('test_auc_mean'))} "
            f"test_recall_mean={format_metric(ratio_summary.get('test_recall_mean'))}"
        )

    # Global best: choose among ratio-level best groups by validation mean AUC -> Recall -> MAP@100 -> MAP@200 -> Precision.
    global_best = None
    best_score = selection_tuple({})
    for row in ratio_summary_records:
        score = selection_tuple(
            {
                "auc": row.get("val_auc_mean", np.nan),
                "recall": row.get("val_recall_mean", np.nan),
                "map100": row.get("val_map100_mean", np.nan),
                "map200": row.get("val_map200_mean", np.nan),
                "precision": row.get("val_precision_mean", np.nan),
            }
        )
        if score > best_score:
            best_score = score
            global_best = row

    if global_best is None:
        global_best_payload = {
            "run_id": run_id,
            "message": "No completed ratio summary available.",
        }
    else:
        gb_ratio = float(global_best["train_ratio"])
        gb_tag = str(global_best["best_param_tag"])
        gb_seed_rows = []
        for r in fine_records:
            if float(r.get("train_ratio", np.nan)) != gb_ratio:
                continue
            params = {
                "alpha": int(r["alpha"]),
                "beta": int(r["beta"]),
                "gamma": int(r["gamma"]),
                "r_layers": int(r["r_layers"]),
            }
            if param_tag(params) == gb_tag:
                gb_seed_rows.append(r)

        gb_test_summary = summarize_metrics(
            gb_seed_rows,
            ["test_auc", "test_recall", "test_precision", "test_map100", "test_map200", "test_f1"],
        )
        global_best_payload = {
            "run_id": run_id,
            "selection_rule": "validation mean AUC -> validation mean Recall -> validation mean MAP@100 -> validation mean MAP@200 -> validation mean Precision",
            "global_best_ratio": gb_ratio,
            "global_best_param_tag": gb_tag,
            "global_best_ratio_summary": global_best,
            "global_best_seed_rows": gb_seed_rows,
            "global_best_test_summary": gb_test_summary,
        }

    summary_json = run_dir / f"summary_{run_id}.json"
    summary_txt = run_dir / f"summary_{run_id}.txt"
    global_json = run_dir / f"global_best_{run_id}.json"
    global_txt = run_dir / f"global_best_{run_id}.txt"

    summary_payload = {
        "run_id": run_id,
        "device": config.device,
        "train_ratios": config.train_ratios,
        "seed_list": config.seed_list,
        "coarse_search_size": len(config.coarse_param_grid()),
        "resume_run_id": config.resume_run_id,
        "resume_skip_existing": bool(config.resume_skip_existing),
        "resume_rebuild_metrics": bool(config.resume_rebuild_metrics),
        "coarse_records_csv": str(coarse_csv),
        "fine_records_csv": str(fine_csv),
        "ratio_summary_csv": str(ratio_csv),
        "ratio_summaries": ratio_summary_records,
        "global_best": global_best_payload,
    }

    save_json(global_best_payload, str(global_json))
    save_metrics_text(global_best_payload, str(global_txt))
    save_json(summary_payload, str(summary_json))
    save_metrics_text(summary_payload, str(summary_txt))

    print("\n" + "=" * 88)
    print("WDCNN resume pipeline finished")
    print(f"Global best JSON: {global_json}")
    print("=" * 88)


if __name__ == "__main__":
    main()
