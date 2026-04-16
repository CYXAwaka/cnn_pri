# WDCNN 单配置复现版

本项目已简化为“固定参数复现 Stage A(1/18)”流程，目标是稳定复现如下训练设定：

- 训练比例：`0.80`
- 随机种子：`42`
- 模型参数：`a90_b120_g20_r3`（`dropout=0.2`）
- 训练参数：`lr=4e-4`、`warmup=1`、`train_epochs=7`
- 学习率轨迹：按 `scheduler_total_epochs=20` 计算余弦调度，但训练在第 7 轮结束
- 阈值策略：验证集按 `precision>=0.30` 约束选阈值（无可行阈值时回退到 F1 最优阈值）

## 目录说明

- `config.py`：固定复现实验参数
- `data_process.py`：数据预处理与数据集划分
- `src/models/wdcnn_model.py`：Wide + Deep CNN 模型结构
- `engine.py`：训练、评估、指标与绘图
- `main_smooth_best.py`：唯一实验入口

## 运行方式

在项目根目录执行：

```bash
python main_smooth_best.py
```

## 输出产物

运行后会在 `results/repro_stageA001/run_*/` 生成：

- `checkpoint`
- 训练曲线图 `history`
- ROC/PR/TopN 图
- `metrics_*.json/.txt`
- `summary_*.json/.txt`

## 说明

- 控制台日志与关键说明已改为中文。
- 历史 `checkpoints/results` 产物保留，不会被删除。
