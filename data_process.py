from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset


@dataclass
class ProcessedDataset:
    X_1d: np.ndarray
    X_2d: np.ndarray
    y: np.ndarray
    day_count: int
    week_count: int
    padded_days: int


@dataclass
class SplitData:
    X1_train: np.ndarray
    X1_val: np.ndarray
    X1_test: np.ndarray
    X2_train: np.ndarray
    X2_val: np.ndarray
    X2_test: np.ndarray
    y_train: np.ndarray
    y_val: np.ndarray
    y_test: np.ndarray
    train_loader: DataLoader
    val_loader: DataLoader
    test_loader: DataLoader
    wide_input_dim: int
    deep_input_shape: tuple[int, int, int]


def read_meter_data(file_path: str) -> pd.DataFrame:
    path = Path(file_path)
    suffix = path.suffix.lower()
    if suffix == ".csv":
        return pd.read_csv(path)
    if suffix in {".xlsx", ".xls"}:
        return pd.read_excel(path)
    raise ValueError(f"Unsupported data format: {suffix}")


def infer_date_columns(df: pd.DataFrame, id_col: str, label_col: str) -> list[str]:
    date_cols: list[str] = []
    for col in df.columns:
        if col in {id_col, label_col}:
            continue
        try:
            pd.to_datetime(col)
            date_cols.append(col)
        except Exception:
            continue
    date_cols.sort(key=lambda x: pd.to_datetime(x))
    return date_cols


def _fill_missing_values_paper(x: np.ndarray) -> np.ndarray:
    """Paper Eq.(1): avg(neighbors) if both exist, else 0."""
    x_filled = x.copy()
    n_days = x.shape[1]

    for i in range(x.shape[0]):
        orig = x[i]
        row = x_filled[i]
        for j in range(n_days):
            if not np.isnan(orig[j]):
                continue
            left = orig[j - 1] if j - 1 >= 0 else np.nan
            right = orig[j + 1] if j + 1 < n_days else np.nan
            if not np.isnan(left) and not np.isnan(right):
                row[j] = 0.5 * (left + right)
            else:
                row[j] = 0.0

    # Defensive fallback
    return np.nan_to_num(x_filled, nan=0.0, posinf=0.0, neginf=0.0)


def _clip_outliers_paper(x: np.ndarray, k: float = 2.0) -> np.ndarray:
    """Paper Eq.(2): if x_i > avg(x)+k*std(x), clamp to that threshold."""
    out = x.copy()
    for i in range(out.shape[0]):
        row = out[i]
        mu = float(np.mean(row))
        sigma = float(np.std(row))
        if sigma <= 0:
            continue
        upper = mu + k * sigma
        row[row > upper] = upper
        out[i] = row
    return out


def _minmax_normalize_by_row(x: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    row_min = np.min(x, axis=1, keepdims=True)
    row_max = np.max(x, axis=1, keepdims=True)
    denom = np.maximum(row_max - row_min, eps)
    return (x - row_min) / denom


def _pad_to_full_weeks(
    x: np.ndarray,
    days_per_week: int,
    pad_value: float,
) -> tuple[np.ndarray, int]:
    n_days = x.shape[1]
    padded_days = int(np.ceil(n_days / days_per_week) * days_per_week)
    pad_len = padded_days - n_days
    if pad_len <= 0:
        return x, 0
    x_pad = np.pad(x, ((0, 0), (0, pad_len)), mode="constant", constant_values=pad_value)
    return x_pad, pad_len


def _to_week_matrix(x_padded: np.ndarray, days_per_week: int) -> np.ndarray:
    n_samples, total_days = x_padded.shape
    if total_days % days_per_week != 0:
        raise ValueError("Padded series length must be divisible by days_per_week")
    n_weeks = total_days // days_per_week
    x2 = x_padded.reshape(n_samples, n_weeks, days_per_week)
    return np.expand_dims(x2, axis=1).astype(np.float32)


def _print_data_report(name: str, y: np.ndarray, day_count: int, week_count: int, padded_days: int) -> None:
    total = int(len(y))
    pos = int(np.sum(y == 1))
    neg = int(np.sum(y == 0))
    print("=" * 72)
    print(f"{name}")
    print(f"样本数={total}, 正常用户={neg}, 窃电用户={pos}, 正样本占比={pos / max(total, 1):.4f}")
    print(f"天数长度={day_count}, 周数={week_count}, 补齐天数={padded_days}")
    print("=" * 72)


def preprocess_for_wdcnn(
    file_path: str,
    id_col: str,
    label_col: str,
    days_per_week: int = 7,
    fill_missing_calendar_days: bool = True,
    use_outlier_clip: bool = True,
    outlier_sigma_k: float = 2.0,
    normalize_eps: float = 1e-8,
    week_pad_value: float = 0.0,
) -> ProcessedDataset:
    df = read_meter_data(file_path)
    if id_col not in df.columns or label_col not in df.columns:
        raise ValueError(f"Missing required columns: {id_col}, {label_col}")

    date_cols = infer_date_columns(df, id_col=id_col, label_col=label_col)
    if not date_cols:
        raise ValueError("No date columns were detected in the dataset")

    # Parse and optionally align to full calendar range.
    x_df = df[date_cols].apply(pd.to_numeric, errors="coerce")
    parsed_dates = pd.to_datetime(date_cols)
    x_df.columns = parsed_dates

    if fill_missing_calendar_days:
        full_calendar = pd.date_range(parsed_dates.min(), parsed_dates.max(), freq="D")
        x_df = x_df.reindex(columns=full_calendar)

    y = df[label_col].astype(int).values.astype(np.int64)

    x = x_df.to_numpy(dtype=np.float32)
    x = _fill_missing_values_paper(x)
    if use_outlier_clip:
        x = _clip_outliers_paper(x, k=outlier_sigma_k)
    x = _minmax_normalize_by_row(x, eps=normalize_eps).astype(np.float32)

    x_padded, padded_days = _pad_to_full_weeks(x, days_per_week=days_per_week, pad_value=week_pad_value)
    x2 = _to_week_matrix(x_padded, days_per_week=days_per_week)

    dataset = ProcessedDataset(
        X_1d=x,
        X_2d=x2,
        y=y,
        day_count=x.shape[1],
        week_count=x2.shape[2],
        padded_days=padded_days,
    )
    _print_data_report(
        name="论文对齐预处理完成",
        y=dataset.y,
        day_count=dataset.day_count,
        week_count=dataset.week_count,
        padded_days=dataset.padded_days,
    )
    return dataset


def _build_dataloader(
    x1: np.ndarray,
    x2: np.ndarray,
    y: np.ndarray,
    batch_size: int,
    shuffle: bool,
    num_workers: int,
) -> DataLoader:
    ds = TensorDataset(
        torch.tensor(x1, dtype=torch.float32),
        torch.tensor(x2, dtype=torch.float32),
        torch.tensor(y, dtype=torch.float32),
    )
    return DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=False,
        drop_last=False,
    )


def split_dataset_for_ratio(
    dataset: ProcessedDataset,
    train_ratio: float,
    val_ratio_in_train: float,
    batch_size: int,
    random_state: int,
    num_workers: int = 0,
) -> SplitData:
    if train_ratio <= 0.0 or train_ratio >= 1.0:
        raise ValueError("train_ratio must be within (0, 1)")
    if val_ratio_in_train <= 0.0 or val_ratio_in_train >= 1.0:
        raise ValueError("val_ratio_in_train must be within (0, 1)")

    n = dataset.y.shape[0]
    all_idx = np.arange(n)

    train_idx, test_idx = train_test_split(
        all_idx,
        train_size=train_ratio,
        random_state=random_state,
        stratify=dataset.y,
    )

    train_sub_idx, val_idx = train_test_split(
        train_idx,
        test_size=val_ratio_in_train,
        random_state=random_state,
        stratify=dataset.y[train_idx],
    )

    x1_train = dataset.X_1d[train_sub_idx]
    x1_val = dataset.X_1d[val_idx]
    x1_test = dataset.X_1d[test_idx]

    x2_train = dataset.X_2d[train_sub_idx]
    x2_val = dataset.X_2d[val_idx]
    x2_test = dataset.X_2d[test_idx]

    y_train = dataset.y[train_sub_idx].astype(np.int64)
    y_val = dataset.y[val_idx].astype(np.int64)
    y_test = dataset.y[test_idx].astype(np.int64)

    split = SplitData(
        X1_train=x1_train,
        X1_val=x1_val,
        X1_test=x1_test,
        X2_train=x2_train,
        X2_val=x2_val,
        X2_test=x2_test,
        y_train=y_train,
        y_val=y_val,
        y_test=y_test,
        train_loader=_build_dataloader(x1_train, x2_train, y_train, batch_size, True, num_workers),
        val_loader=_build_dataloader(x1_val, x2_val, y_val, batch_size, False, num_workers),
        test_loader=_build_dataloader(x1_test, x2_test, y_test, batch_size, False, num_workers),
        wide_input_dim=x1_train.shape[1],
        deep_input_shape=tuple(x2_train.shape[1:]),
    )

    _print_data_report(
        name=f"数据划分完成（训练比例={train_ratio:.2f}，随机种子={random_state}）",
        y=np.concatenate([split.y_train, split.y_val, split.y_test]),
        day_count=dataset.day_count,
        week_count=dataset.week_count,
        padded_days=dataset.padded_days,
    )
    print(f"划分规模 => 训练集={len(split.y_train)}, 验证集={len(split.y_val)}, 测试集={len(split.y_test)}")
    return split


def count_pos_neg(labels: Sequence[int] | np.ndarray) -> tuple[int, int]:
    arr = np.asarray(labels)
    pos = int(np.sum(arr == 1))
    neg = int(np.sum(arr == 0))
    return pos, neg
