
"""
data_process.py
~~~~~~~~~~~~~~~
这个文件专门负责数据预处理。

我把它拆成了很多“小函数”，每个函数只做一件事：
1. 读数据
2. 自动识别日期列
3. 缺失值修复
4. 离群值修复
5. 归一化
6. 按周重构为 2D CNN 输入
7. 划分训练/验证/测试集
8. 构造 DataLoader

这样你以后看代码不会乱，也更符合“模块化”的要求。
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset, TensorDataset


def read_meter_data(file_path: str) -> pd.DataFrame:
    """
    读取电表数据。

    为什么这里要同时支持 csv 和 xlsx？
    因为你现在手上可能有：
    - data/raw/data.csv
    - 部分数据.xlsx
    为了避免你来回改代码，这里直接自动判断格式。
    """
    file_path = Path(file_path)
    suffix = file_path.suffix.lower()

    if suffix == ".csv":
        df = pd.read_csv(file_path)
    elif suffix in [".xlsx", ".xls"]:
        df = pd.read_excel(file_path)
    else:
        raise ValueError(f"暂不支持的文件格式: {suffix}")

    return df


def infer_date_columns(df: pd.DataFrame, id_col: str, label_col: str) -> List[str]:
    """
    自动推断“时间序列列”。

    原理：
    - 去掉 ID 列和标签列
    - 尝试把剩下的列名转成时间
    - 能转成功的就认为是时间列

    为什么要这样写？
    因为有的数据列名是字符串 '2014-01-01'
    有的数据列名是 pandas.Timestamp / datetime
    统一自动处理更稳。
    """
    date_cols = []

    for col in df.columns:
        if col in [id_col, label_col]:
            continue
        try:
            pd.to_datetime(col)
            date_cols.append(col)
        except Exception:
            # 不是日期列就跳过
            continue

    # 按真实日期先后顺序排序
    date_cols = sorted(date_cols, key=lambda x: pd.to_datetime(x))
    return date_cols


def basic_data_report(df: pd.DataFrame, feature_df: pd.DataFrame, y: np.ndarray) -> None:
    """
    打印数据体检信息。
    这个函数非常适合你调试，也适合后面写论文“数据分析与预处理”部分时查看统计信息。
    """
    total = len(y)
    pos = int((y == 1).sum())
    neg = int((y == 0).sum())

    print("=" * 60)
    print("数据体检报告")
    print("=" * 60)
    print(f"原始数据形状: {df.shape}")
    print(f"样本数: {total}")
    print(f"正常用户数(0): {neg}")
    print(f"窃电用户数(1): {pos}")
    print(f"正样本比例: {pos / total:.4f}")
    print(f"时间步数量: {feature_df.shape[1]}")
    print(f"整体缺失率: {feature_df.isna().mean().mean():.4f}")
    print(f"整体 0 值比例: {(feature_df == 0).sum().sum() / feature_df.size:.4f}")
    print("=" * 60)


def repair_missing_values_linear(X: pd.DataFrame) -> pd.DataFrame:
    """
    按“行”做线性插值，修复缺失值。

    行 = 一个用户完整的时间序列
    列 = 每一天的用电量

    为什么转置再插值？
    因为 pandas 的 interpolate 默认是沿行方向插值。
    这里我们的时间维在“列”，所以先转置，再转回来。
    """
    X = X.T.interpolate(method="linear", limit_direction="both").T
    # 如果某一整行全是 NaN，线性插值后可能仍有缺失，这里统一补 0
    X = X.fillna(0.0)
    return X


def repair_outliers_three_sigma(X: pd.DataFrame) -> pd.DataFrame:
    """
    使用 3σ 原则修复离群值。

    这里采用的是“按样本逐行修复”的思路：
    - 对每个用户自己的时间序列计算均值和标准差
    - 若某天的数值 > 均值 + 3σ，则用相邻点均值修复

    为什么不直接删除离群值？
    因为时间序列被删除点以后会破坏长度一致性。
    所以在这类任务里，“修复”通常比“删除”更合适。
    """
    X_fixed = X.copy()
    arr = X_fixed.values.astype(float)

    for i in range(arr.shape[0]):
        row = arr[i]
        mean_ = np.nanmean(row)
        std_ = np.nanstd(row)

        if np.isnan(std_) or std_ == 0:
            continue

        upper = mean_ + 3 * std_

        for j in range(len(row)):
            if row[j] > upper:
                # 用相邻点均值替代
                left = row[j - 1] if j - 1 >= 0 else row[j]
                right = row[j + 1] if j + 1 < len(row) else row[j]
                row[j] = (left + right) / 2.0

        arr[i] = row

    return pd.DataFrame(arr, columns=X.columns, index=X.index)


def minmax_normalize_by_row(X: pd.DataFrame) -> pd.DataFrame:
    """
    对每个用户自己的时间序列做 Min-Max 归一化。

    为什么按行归一化，而不是按整列归一化？
    因为这里更关注“某个用户自己的用电曲线形状”，
    而不是绝对电量值的大小。

    这也更接近窃电检测中常见的思路：
    模型学习的是“模式变化”，例如：
    - 周期性是否被破坏
    - 某些区间是否异常变低
    """
    row_min = X.min(axis=1)
    row_max = X.max(axis=1)
    X_norm = X.sub(row_min, axis=0).div((row_max - row_min + 1e-8), axis=0)
    return X_norm


def keep_valid_samples(
    X: pd.DataFrame,
    y: np.ndarray,
    missing_threshold: float
) -> Tuple[pd.DataFrame, np.ndarray]:
    """
    删除缺失率过高的样本。

    这么做的原因：
    缺失过多的样本即使被插值，也很可能已经失真，
    强行保留反而会误导模型。
    """
    missing_rate = X.isna().mean(axis=1)
    keep_mask = missing_rate <= missing_threshold
    X_keep = X.loc[keep_mask].reset_index(drop=True)
    y_keep = y[keep_mask.values]
    return X_keep, y_keep


def trim_days_to_full_weeks(
    X: pd.DataFrame,
    days_per_week: int,
    target_weeks: int | None = None
) -> pd.DataFrame:
    """
    将天级数据裁剪成“完整周”。

    例如：
    - 1035 天 -> 147 周 + 6 天
    - 只保留完整的 147×7 = 1029 天

    如果设置了 target_weeks=147，则最终保留 147 周。
    这样就能与你参考论文中的 147×7 输入保持一致。
    """
    total_days = X.shape[1]
    full_weeks = total_days // days_per_week

    if target_weeks is not None:
        full_weeks = min(full_weeks, target_weeks)

    keep_days = full_weeks * days_per_week
    X_trim = X.iloc[:, :keep_days].copy()
    return X_trim


def build_week_image(X: pd.DataFrame, days_per_week: int = 7) -> np.ndarray:
    """
    将 1D 时间序列重构为 2D 周矩阵。

    输入：
        [样本数, 天数]
    输出：
        [样本数, 1, 周数, 7]

    举例：
        1029 天 -> 147 周
        输出 shape = [N, 1, 147, 7]

    这一步非常关键，因为论文中的 CNN 就是吃这种 2D 输入的。
    """
    values = X.values.astype(np.float32)
    n_samples, n_days = values.shape

    if n_days % days_per_week != 0:
        raise ValueError("天数不能被 7 整除，请先裁剪成完整周。")

    n_weeks = n_days // days_per_week
    values_2d = values.reshape(n_samples, n_weeks, days_per_week)

    # 增加通道维，供 Conv2D 使用
    values_2d = np.expand_dims(values_2d, axis=1)  # [N, 1, weeks, 7]
    return values_2d


def random_oversample(
    X: np.ndarray,
    y: np.ndarray,
    random_state: int = 42
) -> Tuple[np.ndarray, np.ndarray]:
    """
    训练集随机过采样，只在训练集上使用。

    为什么要做这一步？
    因为 SGCC 数据严重不平衡，正常用户远多于窃电用户。
    如果不处理，模型很容易学成“永远预测正常”，
    然后得到一个看起来很高、实际上没意义的 Accuracy。

    这里采用最朴素也最稳的方式：
    - 找到少数类
    - 有放回抽样复制它，直到和多数类数量一致
    """
    rng = np.random.default_rng(random_state)

    classes, counts = np.unique(y, return_counts=True)
    if len(classes) != 2:
        return X, y

    major_class = classes[np.argmax(counts)]
    minor_class = classes[np.argmin(counts)]

    major_idx = np.where(y == major_class)[0]
    minor_idx = np.where(y == minor_class)[0]

    if len(minor_idx) == 0:
        return X, y

    extra_minor_idx = rng.choice(minor_idx, size=len(major_idx) - len(minor_idx), replace=True)
    balanced_idx = np.concatenate([major_idx, minor_idx, extra_minor_idx])
    rng.shuffle(balanced_idx)

    return X[balanced_idx], y[balanced_idx]


@dataclass
class PreparedData:
    """
    用 dataclass 把所有预处理结果打包，避免 main.py 里变量满天飞。
    """
    X_train_img: np.ndarray
    X_val_img: np.ndarray
    X_test_img: np.ndarray
    y_train: np.ndarray
    y_val: np.ndarray
    y_test: np.ndarray
    train_loader: DataLoader
    val_loader: DataLoader
    test_loader: DataLoader
    input_shape: Tuple[int, int, int]


def build_dataloader(
    X: np.ndarray,
    y: np.ndarray,
    batch_size: int,
    shuffle: bool
) -> DataLoader:
    """
    把 numpy 数据打包成 PyTorch 的 DataLoader。
    """
    dataset = TensorDataset(
        torch.tensor(X, dtype=torch.float32),
        torch.tensor(y, dtype=torch.long)
    )
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)


def preprocess_for_cnn_lg(
    file_path: str,
    id_col: str,
    label_col: str,
    missing_threshold: float,
    use_outlier_repair: bool,
    days_per_week: int,
    target_weeks: int,
    train_ratio: float,
    val_ratio: float,
    test_ratio: float,
    batch_size: int,
    random_state: int,
    use_random_oversample: bool = True,
) -> PreparedData:
    """
    这是 CNN-LG 的完整预处理主流程。

    整体步骤：
    1. 读数据
    2. 找日期列
    3. 删掉缺失过高样本
    4. 线性插值
    5. 可选：3σ 修复离群值
    6. 按行归一化
    7. 裁剪成完整周
    8. 转成 [N, 1, 周数, 7] 的 2D 输入
    9. 分层划分训练/验证/测试
    10. 只对训练集做过采样
    """
    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6, "数据划分比例之和必须等于 1"

    df = read_meter_data(file_path)

    if id_col not in df.columns or label_col not in df.columns:
        raise ValueError(f"数据中缺少必要列：{id_col} 或 {label_col}")

    date_cols = infer_date_columns(df, id_col=id_col, label_col=label_col)
    if len(date_cols) == 0:
        raise ValueError("没有识别到日期列，请检查列名格式。")

    X = df[date_cols].apply(pd.to_numeric, errors="coerce")
    y = df[label_col].astype(int).values

    basic_data_report(df, X, y)

    # 删除缺失过高样本
    X, y = keep_valid_samples(X, y, missing_threshold)

    # 缺失值修复
    X = repair_missing_values_linear(X)

    # 离群值修复
    if use_outlier_repair:
        X = repair_outliers_three_sigma(X)

    # 归一化
    X = minmax_normalize_by_row(X)

    # 裁剪成完整周，并尽量贴近论文输入 147×7
    X = trim_days_to_full_weeks(X, days_per_week=days_per_week, target_weeks=target_weeks)

    # 转成 CNN 2D 输入
    X_img = build_week_image(X, days_per_week=days_per_week)

    # 先划测试集：论文里是 50% 测试
    X_train_val, X_test, y_train_val, y_test = train_test_split(
        X_img,
        y,
        test_size=test_ratio,
        stratify=y,
        random_state=random_state,
    )

    # 再从剩下的部分划验证集
    # 在 train+val 中，验证集占比 = val / (train + val)
    val_ratio_in_train_val = val_ratio / (train_ratio + val_ratio)

    X_train, X_val, y_train, y_val = train_test_split(
        X_train_val,
        y_train_val,
        test_size=val_ratio_in_train_val,
        stratify=y_train_val,
        random_state=random_state,
    )

    # 只对训练集做过采样
    if use_random_oversample:
        X_train, y_train = random_oversample(X_train, y_train, random_state=random_state)

    train_loader = build_dataloader(X_train, y_train, batch_size=batch_size, shuffle=True)
    val_loader = build_dataloader(X_val, y_val, batch_size=batch_size, shuffle=False)
    test_loader = build_dataloader(X_test, y_test, batch_size=batch_size, shuffle=False)

    input_shape = tuple(X_train.shape[1:])  # [C, H, W]

    print(f"CNN 输入形状（单样本）: {input_shape}")
    print(f"训练集: {len(y_train)}，验证集: {len(y_val)}，测试集: {len(y_test)}")

    return PreparedData(
        X_train_img=X_train,
        X_val_img=X_val,
        X_test_img=X_test,
        y_train=y_train,
        y_val=y_val,
        y_test=y_test,
        train_loader=train_loader,
        val_loader=val_loader,
        test_loader=test_loader,
        input_shape=input_shape,
    )
