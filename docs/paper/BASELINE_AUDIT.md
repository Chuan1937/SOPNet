# 基线模型审查（G1）

审查对象：`sopnet/training/baselines.py`、`sopnet/training/metrics.py`、
`scripts/train_baselines.py`、`scripts/run_robustness.py`，对照 SeisPolarity 官方
实现（`/home/yuan/code/SeisPolarity`）。

## 1. 各模型协议核对

| 模型 | 官方声明 | 我们当前实现 | 结论 |
|---|---|---|---|
| Ross (SCSN) | 400 样本，3 类 `{0:Up, 1:Down, 2:Unknown}` | 400 样本；UP→0/DOWN→1/UNKNOWN→2 | ✅ 一致 |
| RPNet | 400 样本，2 类，**索引 1 = Up，0 = Down**（`build_picks` 注释） | 训练目标 DOWN=1、解码 1→DOWN | ❌ 与官方类别顺序相反（内部自洽，但应改为官方顺序后重训） |
| EQPolarity | 600 样本，2 类 `{0:Up, 1:Down}`，单 logit | 模型以 `input_length=200` 构建并训练 | ❌ 输入长度与官方不符；需按 600 重建并重训 |
| CFM | 160 样本，2 类，单 logit | 160 样本 | ✅ 一致（预处理差异见下） |
| DiTingMotion | 128 样本，2 通道；第 2 通道 = **一阶差分取符号**（`DifferentialFeatures`：`np.diff`，前向差分，**在开头补 1 个 0**，再取 `sign`） | 第 2 通道 = `torch.gradient`（中心差分、连续值、无符号） | ❌ 必须修正并重训 |

## 2. 必须修正的问题

### 2.1 DiTingMotion 第二通道（严重）

官方定义（`seispolarity/generate/augmentation.py::DifferentialFeatures`）：

```
diff = np.diff(x)            # 前向差分，长度 N-1
diff = np.pad(diff, (1, 0))  # 开头补一个 0，恢复长度 N
ch2  = np.sign(diff)         # 取符号
输入 = concat([x, ch2])      # (2, N)
```

修正 `prepare_baseline_input`：前向差分 + 开头补零 + 取符号，然后重训
DiTingMotion。

### 2.2 RPNet 类别顺序

官方 `build_picks` 明确 "output index 1 is UP, 0 is DOWN"。改为训练目标
`UP→1`、解码 `1→UP`（与官方一致），并重训 rpnet。当前 checkpoint 在重训前
不得与新代码混用。

### 2.3 EQPolarity 输入长度

官方声明 600 样本（6 s @ 100 Hz）。我们的缓存窗口恰为 600 样本，因此可以直接
喂全窗口。需以 `input_length=600` 构建模型并重训（Transformer 序列长度 ×3，
预计 15-30 h，含早停可能更短）。

### 2.4 Unknown 的二分类处理（评估协议）

`binary_confusion()` 目前把所有非 UP 预测（含 Unknown）都记为 Down 类：
对 3 类模型，预测 Unknown 会被当作 Down 参与指标计算。计划要求：

- **主结果（strict U/D）**：用模型自身的 U/D 得分做二分类决策——
  3 类模型比较 U 与 D 的 logit（忽略 Unknown 类）；2 类模型用其 U/D 概率。
- **补充结果（拒判）**：报告模型自己的 Unknown 预测率（3 类模型 argmax 为
  Unknown 的比例；SOPNet 为置信度低于验证集阈值的比例）。

实现：新增 `strict_ud` 解码与 `unknown_rate` 指标，主表使用 strict U/D，
拒判率与三分类指标放入补充材料。

## 3. 其他核对项（已确认无需修改）

- Ross 3 类映射与官方一致。
- CFM 输入 160、单 logit sigmoid，与官方一致。
- 所有基线都在统一缓存上从零重训，未使用官方权重，无测试集泄漏。
- 数据划分按 event_key + 波形哈希去重，六个模型共享同一份测试清单。

## 4. 需要重训的模型与原因

| 模型 | 原因 | 预计耗时 |
|---|---|---|
| DiTingMotion | 第 2 通道定义修正 | ~2 h |
| RPNet | 类别顺序改为官方约定 | ~2 h |
| EQPolarity | 输入长度改为官方 600 | 15-30 h |
| Ross | strict U/D 主指标下建议重训（原早停基于含 Unknown 的 F1） | ~1-2 h |
| CFM | 无需重训（协议已一致，50 epoch 完成） | — |

重训全部使用相同的训练集、优化器、增强与早停 patience（6），并记录
`baseline_protocol.csv`（模型、输入长度、通道、输出约定、checkpoint、
训练轮数）。

## 5. 待办清单

- [x] 审查各模型协议与官方实现
- [ ] 修正 DiTingMotion 输入通道
- [ ] 修正 RPNet 类别顺序
- [ ] 新增 strict U/D 解码与 unknown_rate 指标
- [ ] EQPolarity 以 600 输入重建
- [ ] 重训四个基线并重生成主结果表
- [ ] 逐源结果表 `per_source_results.csv`
- [ ] 保存逐样本预测到 `outputs/paper/predictions/`
