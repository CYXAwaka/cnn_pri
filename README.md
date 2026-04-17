``# WDCNN 

---

## 1. 项目架构总览

### 1.1 代码文件职责

- `config.py`  
  全局配置中心：路径、预处理参数、模型参数、训练参数、日志语言、阈值策略等。

- `data_process.py`  
  数据读取与预处理模块：日期识别、缺失恢复、异常截断、归一化、1D/2D 双输入构建、分层划分 DataLoader。

- `src/models/wdcnn_model.py`  
  模型结构定义：`PaperG1G2Transform`、`WideBranch`、`DeepBranch`、`WideDeepCNN`。

- `engine.py`  
  训练与评估引擎：训练循环、阈值搜索、指标计算、选优与早停、画图、JSON/TXT/CSV 落盘。

- `main_smooth_best.py`  
  主入口：串联配置、数据、模型、训练、测试、产物保存。

- `tests/test_threshold_selector.py`  
  阈值选择逻辑单测，验证 precision 下限约束与回退逻辑正确。

---

## 2. 数据与模型流程

### 2.1 数据预处理流程（`data_process.py`）

预处理按以下顺序执行：

1``. 读取数据（csv/xlsx/xls）。  
2. 自动识别日期列并按时间升序重排。  
3. 可选：补齐完整日历日期列（避免时间断档）。  
4. 缺失值恢复：  
   - 左右邻居都存在 -> 邻居均值  
   - 否则 -> 0  
5. 异常值处理：`x > mean + k*std` 时上截断。  
6. 每个用户做 Min-Max 归一化。  
7. 同时构建两路输入：  
   - `X_1d`：日序列（给 Wide 分支）  
   - `X_2d`：周矩阵 `[1, week_count, 7]`（给 Deep 分支）  
8. 分层切分 `train/val/test` 并生成 DataLoader。

### 2.2 模型结构（`src/models/wdcnn_model.py`）

模型由三部分组成：

1. `WideBranch`  
   对 1D 序列做全连接提取全局模式，输出 `alpha` 维特征。

2. `DeepBranch`  
   对 2D 周矩阵先做 `g1/g2` 固定趋势增强，再经多层卷积+池化+全连接，输出 `beta` 维特征。

3. `Fusion Head`  
   拼接 `[wide_feat, deep_feat]` 后映射到 1 维 logit，完成二分类。

---

## 3. 训练、验证、测试流程

### 3.1 训练主循环（`engine.py::train_wdcnn_model`）

每个 epoch 执行：

1. 训练集前向/反向，得到 `train_loss` 与训练分数。  
2. 验证集前向，得到 `val_loss` 与验证分数。  
3. 验证阈值选择（优先级）：  
   - 若配置固定阈值 -> 直接用  
   - 否则若设置 `precision_floor` -> 在满足 `precision>=floor` 的候选阈值里选 `recall` 最大（平分再看 F1）  
   - 否则按 `threshold_metric`（默认 F1）搜索  
4. 计算每轮指标并打印日志。  
5. 按选优规则更新“最佳模型”。  
6. 调度学习率（cosine 或 plateau）。  
7. 若启用早停且长期无提升，则停止训练。

### 3.2 选优规则（当前默认）

验证集排序优先级：

`AUC -> Recall -> MAP@100 -> MAP@200 -> Precision`

即 AUC 优先，AUC 相同再比较 Recall，以此类推。

### 3.3 测试阶段

训练结束后：

1. 回载最佳 checkpoint。  
2. 使用“最佳验证轮的阈值”在测试集评估。  
3. 输出测试指标与图像结果。

---

## 4. 指标体系与作用说明

### 4.1 AUC（ROC-AUC）

- 含义：模型将正样本排在负样本之前的总体能力。  
- 适用：总体区分能力评估。  
- 特点：对类别不平衡相对不敏感。

### 4.2 Recall（召回率）

- 公式：`TP / (TP + FN)`  
- 含义：真实窃电用户里，被模型抓到的比例。  
- 业务意义：漏检风险控制核心指标。

### 4.3 Precision（精确率）

- 公式：`TP / (TP + FP)`  
- 含义：模型判为窃电的样本里，真正窃电的比例。  
- 业务意义：人工核查成本控制核心指标。

### 4.4 F1

- 公式：`2 * Precision * Recall / (Precision + Recall)`  
- 含义：精确率与召回率的调和平均。  
- 作用：常用于阈值搜索回退指标（平衡误报与漏报）。

### 4.5 MAP@100 / MAP@200

- 含义：按分数降序取 Top-N 后，对正样本位置的精确率做平均。  
- 直观理解：排名越靠前越准，MAP 越高。  
- 业务意义：非常适合“优先核查前 N 户”的场景。

### 4.6 Top-N Precision 曲线

- 定义：`P@N = Top-N 中正样本数 / N`。  
- 作用：观察“只查前 N 个”时命中率如何随 N 变化。

---

## 5. 输出产物说明

每次运行都会在 `results/repro_stageA001/run_<run_id>/` 下产出：

- `history_*.png`：训练过程总览图（Loss/AUC/MAP/Recall/Precision/LR）
- `roc_*.png`：ROC 曲线
- `pr_*.png`：PR 曲线
- `topn_*.png`：Top-N Precision 曲线
- `metrics_*.json/.txt`：完整指标与配置
- `summary_*.json/.txt`：摘要信息

权重文件保存在：

- `checkpoints/repro_stageA001/wdcnn_<run_id>_seed42.pth`

---

## 6. 运行方式

在项目根目录执行：

```bash
python main_smooth_best.py
```

---

## 7. 常改配置建议

在 `config.py` 中优先调整：

- 数据划分：`train_ratio`、`val_ratio_in_train`
- 模型容量：`alpha / beta / gamma / r_layers / dropout`
- 训练策略：`lr / train_epochs / warmup_epochs / grad_clip_norm`
- 不平衡处理：`pos_weight_scale`
- 阈值偏好：`precision_floor`、`threshold_metric`
- 日志语言：`chinese_log`

---

## 8. 测试

阈值逻辑单测：

```bash
python -m unittest tests/test_threshold_selector.py
```

如果你想继续扩展测试，建议优先增加：

- 预处理边界测试（缺失、异常、非整周补齐）
- 指标公式一致性测试（MAP@N/Top-N）
- 端到端冒烟测试（小样本 + 少 epoch）
