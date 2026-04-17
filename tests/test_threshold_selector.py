from __future__ import annotations

import unittest

import numpy as np

from engine import compute_binary_metrics, select_best_threshold, select_threshold_with_precision_floor


class ThresholdSelectorTest(unittest.TestCase):
    """
    阈值选择策略单元测试：
    - 验证 precision_floor 约束是否生效
    - 验证无可行阈值时是否正确回退到 f1 最优阈值
    """

    def test_precision_floor_recall_priority(self) -> None:
        # 构造一个可满足 precision_floor 的样本集合。
        y_true = np.asarray([1, 1, 1, 0, 0, 0], dtype=np.int64)
        y_prob = np.asarray([0.9, 0.8, 0.4, 0.7, 0.6, 0.1], dtype=np.float32)
        precision_floor = 0.30

        # 执行“精确率下限约束 + 召回优先”的阈值搜索。
        thr = select_threshold_with_precision_floor(
            y_true=y_true,
            y_prob=y_prob,
            precision_floor=precision_floor,
            fallback_metric="f1",
        )
        chosen = compute_binary_metrics(y_true, y_prob, threshold=thr)

        # 断言 1：最终选择阈值必须满足 precision 下限。
        self.assertGreaterEqual(chosen["precision"], precision_floor)

        # 断言 2：在所有满足 precision_floor 的阈值中，召回率应达到最大。
        candidates = np.linspace(0.05, 0.95, 91)
        max_recall = -1.0
        for c in candidates:
            m = compute_binary_metrics(y_true, y_prob, threshold=float(c))
            if m["precision"] >= precision_floor:
                max_recall = max(max_recall, float(m["recall"]))
        self.assertGreaterEqual(chosen["recall"] + 1e-12, max_recall)

    def test_fallback_to_f1_when_no_eligible_threshold(self) -> None:
        # 构造一个不可能满足 precision_floor=1.01 的样本集合。
        y_true = np.asarray([1, 1, 0, 0, 0], dtype=np.int64)
        y_prob = np.asarray([0.9, 0.4, 0.8, 0.7, 0.6], dtype=np.float32)

        fallback_thr = select_threshold_with_precision_floor(
            y_true=y_true,
            y_prob=y_prob,
            precision_floor=1.01,
            fallback_metric="f1",
        )
        f1_thr = select_best_threshold(y_true=y_true, y_prob=y_prob, metric="f1")

        # 当无可行阈值时，必须等价于 f1 最优阈值。
        self.assertAlmostEqual(float(fallback_thr), float(f1_thr), places=12)


if __name__ == "__main__":
    unittest.main()
