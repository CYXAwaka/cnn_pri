
"""
main_cnn_lg.py
~~~~~~~~~~~~~~
这是 CNN-LG 的总入口。

这个文件刻意写得比较“流程化”，方便你理解：
1. 加载配置
2. 做数据预处理
3. 建 CNN 特征提取器
4. 用 CNN 做预训练
5. 提取 64 维特征
6. 训练 LightGBM
7. 评估并保存结果

你可以把这个文件理解为“总调度中心”。
"""

from pathlib import Path

import torch

# 对于部分 Windows / CPU 环境，PyTorch 在线程数过多时可能出现训练非常慢的情况。
# 这里主动限制线程数，让小项目运行更稳定。
torch.set_num_threads(1)
try:
    torch.set_num_interop_threads(1)
except Exception:
    pass

from config import config
from data_process import preprocess_for_cnn_lg
from engine import (
    evaluate_lightgbm,
    extract_features,
    plot_training_history,
    save_lightgbm_model,
    save_metrics_text,
    train_cnn_feature_extractor,
    train_lightgbm_classifier,
)
from src.models.cnn_lg_model import CNNFeatureExtractor, CNNPretrainClassifier


def main():
    # 1. 创建输出目录
    config.make_dirs()

    print(f"当前设备: {config.device}")

    # 2. 数据预处理
    prepared = preprocess_for_cnn_lg(
        file_path=config.data_path,
        id_col=config.id_col,
        label_col=config.label_col,
        missing_threshold=config.missing_threshold,
        use_outlier_repair=config.use_outlier_repair,
        days_per_week=config.days_per_week,
        target_weeks=config.target_weeks,
        train_ratio=config.train_ratio,
        val_ratio=config.val_ratio,
        test_ratio=config.test_ratio,
        batch_size=config.batch_size,
        random_state=config.random_state,
        use_random_oversample=config.use_random_oversample,
    )

    # 3. 创建 CNN 特征提取器
    feature_extractor = CNNFeatureExtractor(
        input_shape=prepared.input_shape,
        in_channels=config.in_channels,
        conv_channels=config.conv_channels,
        kernel_size=config.conv_kernel_size,
        pool1_kernel_size=config.pool1_kernel_size,
        pool1_stride=config.pool1_stride,
        pool2_kernel_size=config.pool2_kernel_size,
        pool2_stride=config.pool2_stride,
        fc_dim=config.fc_dim,
        dropout=config.dropout,
    )
    model = CNNPretrainClassifier(feature_extractor, num_classes=config.num_classes)
    model = model.to(config.device)

    # 4. CNN 预训练
    best_cnn_path = str(Path(config.checkpoint_dir) / "best_cnn_pretrain.pth")
    model, history = train_cnn_feature_extractor(
        model=model,
        train_loader=prepared.train_loader,
        val_loader=prepared.val_loader,
        device=config.device,
        lr=config.cnn_lr,
        weight_decay=config.weight_decay,
        epochs=config.cnn_epochs,
        patience=config.early_stop_patience,
        checkpoint_path=best_cnn_path,
    )

    # 5. 提取特征
    X_train_feat, y_train = extract_features(model, prepared.train_loader, config.device)
    X_val_feat, y_val = extract_features(model, prepared.val_loader, config.device)
    X_test_feat, y_test = extract_features(model, prepared.test_loader, config.device)

    print(f"提取后的特征维度: {X_train_feat.shape[1]}")

    # 6. 训练 LightGBM
    lgbm_model = train_lightgbm_classifier(
        X_train_feat=X_train_feat,
        y_train=y_train,
        params=config.lgbm_params,
    )

    # 7. 评估
    val_metrics = evaluate_lightgbm(lgbm_model, X_val_feat, y_val)
    test_metrics = evaluate_lightgbm(lgbm_model, X_test_feat, y_test)

    print("\n========== 验证集指标 ==========")
    for k, v in val_metrics.items():
        print(f"{k}: {v}")

    print("\n========== 测试集指标 ==========")
    for k, v in test_metrics.items():
        print(f"{k}: {v}")

    # 8. 保存模型和结果
    save_lightgbm_model(lgbm_model, str(Path(config.checkpoint_dir) / "best_lightgbm.pkl"))
    plot_training_history(history, str(Path(config.result_dir) / "cnn_pretrain_history.png"))
    save_metrics_text(
        {"val_metrics": val_metrics, "test_metrics": test_metrics},
        str(Path(config.result_dir) / "cnn_lg_metrics.txt")
    )

    print("\nCNN-LG 训练和评估已完成。")


if __name__ == "__main__":
    main()
