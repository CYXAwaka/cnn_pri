from __future__ import annotations

from pathlib import Path

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
    return f"r{int(round(ratio * 100)):02d}"


def param_tag(params: dict[str, int | float]) -> str:
    return f"a{params['alpha']}_b{params['beta']}_g{params['gamma']}_r{params['r_layers']}"


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


def main() -> None:
    config.make_dirs()

    run_id, run_dir = create_run_id(config.result_dir)
    checkpoint_dir = Path(config.checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    print(f"Device: {config.device}")
    print(f"Run ID: {run_id}")
    print(f"Run Dir: {run_dir}")

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

    coarse_records: list[dict] = []
    fine_records: list[dict] = []
    ratio_summary_records: list[dict] = []

    coarse_grid = config.coarse_param_grid()
    primary_seed = config.seed_list[0]

    for tr in config.train_ratios:
        r_tag = ratio_tag(tr)
        print("\n" + "=" * 88)
        print(f"[Ratio {tr:.2f}] Stage-1 coarse search start")
        print("=" * 88)

        coarse_split = split_dataset_for_ratio(
            dataset=dataset,
            train_ratio=tr,
            val_ratio_in_train=config.val_ratio_in_train,
            batch_size=config.batch_size,
            random_state=primary_seed,
            num_workers=config.num_workers,
        )

        ratio_coarse: list[dict] = []
        for i, params in enumerate(coarse_grid, start=1):
            p_tag = param_tag(params)
            print(f"[Coarse {i:03d}/{len(coarse_grid):03d}] ratio={tr:.2f} params={p_tag}")

            model = build_model(coarse_split, params).to(config.device)
            ckpt_path = str(
                checkpoint_dir
                / f"wdcnn_{run_id}_{r_tag}_{p_tag}_seed{primary_seed}_coarse.pth"
            )

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
                checkpoint_path=ckpt_path,
                threshold_metric=config.threshold_metric,
            )

            row = {
                "run_id": run_id,
                "stage": "coarse",
                "train_ratio": tr,
                "seed": primary_seed,
                "alpha": params["alpha"],
                "beta": params["beta"],
                "gamma": params["gamma"],
                "r_layers": params["r_layers"],
                "best_epoch": train_out.best_epoch,
                "best_threshold": train_out.best_threshold,
                **{f"val_{k}": v for k, v in train_out.best_val_metrics.items()},
            }
            coarse_records.append(row)
            ratio_coarse.append(row)

        ratio_coarse_sorted = sorted(
            ratio_coarse,
            key=lambda x: selection_tuple(
                {
                    "auc": x.get("val_auc", float("nan")),
                    "map100": x.get("val_map100", float("nan")),
                    "map200": x.get("val_map200", float("nan")),
                    "recall": x.get("val_recall", float("nan")),
                    "precision": x.get("val_precision", float("nan")),
                }
            ),
            reverse=True,
        )

        top_param_rows = ratio_coarse_sorted[: config.refine_top_k]
        top_params = [
            {
                "alpha": int(r["alpha"]),
                "beta": int(r["beta"]),
                "gamma": int(r["gamma"]),
                "r_layers": int(r["r_layers"]),
                "dropout": config.dropout,
            }
            for r in top_param_rows
        ]

        print(f"\n[Ratio {tr:.2f}] Stage-2 fine training on top-{len(top_params)} configs")

        param_to_seed_rows: dict[str, list[dict]] = {}

        for params in top_params:
            p_tag = param_tag(params)
            param_to_seed_rows[p_tag] = []

            for seed in config.seed_list:
                split = split_dataset_for_ratio(
                    dataset=dataset,
                    train_ratio=tr,
                    val_ratio_in_train=config.val_ratio_in_train,
                    batch_size=config.batch_size,
                    random_state=seed,
                    num_workers=config.num_workers,
                )

                model = build_model(split, params).to(config.device)
                ckpt_path = str(checkpoint_dir / f"wdcnn_{run_id}_{r_tag}_{p_tag}_seed{seed}_fine.pth")

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
                    checkpoint_path=ckpt_path,
                    threshold_metric=config.threshold_metric,
                )

                test_metrics, y_test, p_test = evaluate_wdcnn_model(
                    train_out.model,
                    split.test_loader,
                    device=config.device,
                    threshold=train_out.best_threshold,
                )

                val_metrics = train_out.best_val_metrics

                tag = f"{r_tag}_{p_tag}_seed{seed}_{run_id}"
                history_path = run_dir / f"history_{tag}.png"
                roc_path = run_dir / f"roc_{tag}.png"
                pr_path = run_dir / f"pr_{tag}.png"
                topn_path = run_dir / f"topn_{tag}.png"
                metrics_text_path = run_dir / f"metrics_{tag}.txt"
                metrics_json_path = run_dir / f"metrics_{tag}.json"

                plot_training_history(train_out.history, str(history_path))
                plot_roc_pr_curves(y_test, p_test, str(roc_path), str(pr_path))
                plot_topn_precision_curve(y_test, p_test, str(topn_path), max_n=200)

                seed_payload = {
                    "run_id": run_id,
                    "stage": "fine",
                    "train_ratio": tr,
                    "seed": seed,
                    "params": params,
                    "best_epoch": train_out.best_epoch,
                    "best_threshold": train_out.best_threshold,
                    "val_metrics": val_metrics,
                    "test_metrics": test_metrics,
                    "artifacts": {
                        "checkpoint": ckpt_path,
                        "history": str(history_path),
                        "roc": str(roc_path),
                        "pr": str(pr_path),
                        "topn": str(topn_path),
                    },
                }

                save_metrics_text(seed_payload, str(metrics_text_path))
                save_json(seed_payload, str(metrics_json_path))

                row = {
                    "run_id": run_id,
                    "stage": "fine",
                    "train_ratio": tr,
                    "seed": seed,
                    "alpha": params["alpha"],
                    "beta": params["beta"],
                    "gamma": params["gamma"],
                    "r_layers": params["r_layers"],
                    "best_epoch": train_out.best_epoch,
                    "best_threshold": train_out.best_threshold,
                    **{f"val_{k}": v for k, v in val_metrics.items()},
                    **{f"test_{k}": v for k, v in test_metrics.items()},
                    "metrics_text": str(metrics_text_path),
                    "metrics_json": str(metrics_json_path),
                }
                fine_records.append(row)
                param_to_seed_rows[p_tag].append(row)

        # Pick best param by mean validation metrics across seeds.
        best_param_tag = None
        best_param_score = selection_tuple({})
        best_param_rows: list[dict] = []

        for p_tag, rows in param_to_seed_rows.items():
            mean_val = {
                "auc": float(np.mean([r.get("val_auc", np.nan) for r in rows])),
                "map100": float(np.mean([r.get("val_map100", np.nan) for r in rows])),
                "map200": float(np.mean([r.get("val_map200", np.nan) for r in rows])),
                "recall": float(np.mean([r.get("val_recall", np.nan) for r in rows])),
                "precision": float(np.mean([r.get("val_precision", np.nan) for r in rows])),
            }
            sc = selection_tuple(mean_val)
            if sc > best_param_score:
                best_param_score = sc
                best_param_tag = p_tag
                best_param_rows = rows

        metric_keys = ["test_auc", "test_map100", "test_map200", "test_recall", "test_precision", "test_f1"]
        test_summary = summarize_metrics(best_param_rows, metric_keys)

        ratio_summary = {
            "run_id": run_id,
            "train_ratio": tr,
            "best_param_tag": best_param_tag,
            "num_seed_runs": len(best_param_rows),
            **{f"{k}_mean": v["mean"] for k, v in test_summary.items()},
            **{f"{k}_std": v["std"] for k, v in test_summary.items()},
        }
        ratio_summary_records.append(ratio_summary)

        print(f"[Ratio {tr:.2f}] best config={best_param_tag}")
        for mk in metric_keys:
            mean_v = ratio_summary.get(f"{mk}_mean", float("nan"))
            std_v = ratio_summary.get(f"{mk}_std", float("nan"))
            print(f"  {mk}: mean={mean_v:.4f}, std={std_v:.4f}")

    # Save global outputs.
    coarse_csv = run_dir / f"coarse_records_{run_id}.csv"
    fine_csv = run_dir / f"fine_records_{run_id}.csv"
    ratio_csv = run_dir / f"ratio_summary_{run_id}.csv"
    summary_json = run_dir / f"summary_{run_id}.json"
    summary_txt = run_dir / f"summary_{run_id}.txt"

    save_records_csv(coarse_records, str(coarse_csv))
    save_records_csv(fine_records, str(fine_csv))
    save_records_csv(ratio_summary_records, str(ratio_csv))

    summary_payload = {
        "run_id": run_id,
        "device": config.device,
        "train_ratios": config.train_ratios,
        "seed_list": config.seed_list,
        "coarse_search_size": len(config.coarse_param_grid()),
        "coarse_records_csv": str(coarse_csv),
        "fine_records_csv": str(fine_csv),
        "ratio_summary_csv": str(ratio_csv),
        "ratio_summaries": ratio_summary_records,
    }
    save_json(summary_payload, str(summary_json))
    save_metrics_text(summary_payload, str(summary_txt))

    print("\n" + "=" * 88)
    print("WDCNN experiment pipeline finished")
    print(f"Summary JSON: {summary_json}")
    print("=" * 88)


if __name__ == "__main__":
    main()
