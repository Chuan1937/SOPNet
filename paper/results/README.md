# SOPNet 论文结果包

本目录是 SOPNet 论文的**可公开发布结果包**：只包含聚合指标、正式图件与溯源
信息；不包含地震波形、模型权重、训练缓存、访问凭据或受限数据集内容。

- **主模型**：SOPNet（Signed-Onset Field Network，带符号起始场网络）
- **对照模型**：CFM、RPNet、Ross、EQPolarity、DiTingMotion（共 5 个）
- **主指标**：strict U/D（只在已知极性样本上评估，用每个模型自身的 U/D 得分判定）
- **规范测试集**：777,720 条测试波形，其中 327,231 条已知 U/D
- **鲁棒性固定子集**：30,000 条测试波形，其中 12,520 条已知 U/D（seed 20261004）
- **消融版本**：C 组续训完成后的终版（A/B/C/D = 0.9746 / 0.9752 / 0.9781 / 0.9782）
- **结果版本**：对应源码提交 `1188a68f10dda6b9c10a0cba8316790818a7466f`

```text
paper/results/
├── README.md
├── tables/
│   ├── main_results.csv
│   ├── per_source_results.csv
│   ├── baseline_protocol.csv
│   ├── ablation_results.csv
│   ├── bootstrap.csv
│   ├── snr_all.csv
│   └── p_shift_all.csv
├── figures/
│   ├── robustness_heatmaps.pdf
│   ├── robustness_heatmaps.svg
│   └── robustness_heatmaps.png
└── provenance/
    └── experiment_manifest.json
```

`tables/` 中的 CSV 是 `outputs/paper/tables/` 对应文件的逐字节副本；每个文件的
SHA-256 记录在 `provenance/experiment_manifest.json`。`outputs/` 本身不入库。

---

## 1. 实验协议

### 1.1 数据与划分

五个公开数据集（SCSN、DiTing、PNW、INSTANCE、TXED）按物理极性统一为
`Up = +1`、`Down = -1`、`Unknown = 0`，事件级划分（seed `20261004`）。

| 数据源 | 测试集已知 U/D 样本数 |
|---|---:|
| SCSN | 253,194 |
| DiTing | 64,048 |
| PNW | 9,742 |
| TXED | 143 |
| INSTANCE | 104 |
| **合计** | **327,231** |

测试集总样本 777,720，测试事件 111,588。TXED 与 INSTANCE 的已知 U/D 测试
样本极少，逐源结果必须结合样本数阅读（`per_source_results.csv`）。

### 1.2 严格 U/D 指标

- 只在**已知极性**（±1）样本上计算，Unknown 样本不进入分母。
- SOPNet：极性取 `sign(f(argmax |f|))`；主指标不使用置信度阈值。
- 基线：使用各模型自身的 U/D 得分/输出；三分类模型（Ross、DiTingMotion）在
  主指标中若输出 Unknown 也按错误计；Unknown 率仅作补充列。
- `macro_f1`：Up/Down 两类的 threshold-free macro-F1，不随置信度阈值变化。
- `precision` / `recall` / `f1`：以 **Up** 为正类。
- 验证集调出的置信度阈值（SOPNet 为 0.95）只用于选择性预测/覆盖率等补充
  分析，不进入主表。

### 1.3 鲁棒性

- 六个模型使用**同一固定 30,000 条测试子集**（seed `20261004`）。
- 噪声：Clean（不加深斯白噪声）、20、10、5、0、−5 dB。
- P 到时偏移：−200、−100、−50、−30、−20、−10、0、+10、+20、+30、+50、
  +100、+200 ms（13 点细网格，包含原粗网格的全部点）。
- 每个模型保持其原生输入长度/窗口（SOPNet 400；基线 600），但接收完全相同的
  样本、噪声与偏移。`snr_all.csv` 的 `accuracy` 是已知 U/D 样本上的严格准确率，
  判定规则为 `ud_score >= 0 → Up`。
- 鲁棒性结果没有计算置信区间，因此图件不绘制误差棒。

### 1.4 消融

四个变体均为 no-jitter、known-only 损失作用域、同一骨干/数据/训练配置，测试集
327,231 条已知 U/D。A/B 在 50 轮调度完成前停止（运行元数据未记录实际完成
轮次），C 已续训至 50 轮，D 复用最终 SOPNet；见第 6 节限制 9。

| 变体 | 说明 |
|---|---|
| A | 二分类分类头（共享编码器） |
| B | 带符号起始场，无一致性损失 |
| C | B + 极性一致性损失（λ_pol = 0.5） |
| D | C + 反转一致性损失（λ_inv = 0.1，完整模型） |

C 行为续训至收敛的终版（best epoch 49，共 50 轮），不是早期欠训练结果。
D 复用最终 SOPNet checkpoint。

---

## 2. CSV 字段

所有比率均为 `[0, 1]` 小数；图件中的百分比 = CSV 值 × 100。

### `main_results.csv`（6 行）

| 字段 | 含义 |
|---|---|
| `model` | 模型显示名 |
| `accuracy` | 已知 U/D 样本严格准确率 |
| `macro_f1` | Up/Down threshold-free macro-F1 |
| `precision`, `recall` | 以 Up 为正类 |
| `mcc` | Matthews 相关系数 |
| `n_known` | 已知 U/D 样本数（327,231） |
| `n` | 测试集总样本数（777,720） |
| `unknown_rate` | 全体样本中被判为 Unknown / 低于选择性阈值的比例（补充） |

### `per_source_results.csv`（30 行 = 6 模型 × 5 数据源）

字段：`model, source, n_known, accuracy, macro_f1, mcc`。每个模型的 `n_known`
之和为 327,231。

### `baseline_protocol.csv`（6 行）

字段：`name, factory, input_length, input_channels, output, derivative,
window_length, epochs, epochs_run, best_epoch, batch_size, seed, parameters,
checkpoint, predictions, model, note`。

该表保留运行记录原貌，不补造未记录字段：

- CFM 是从遗留 checkpoint 按同一 strict U/D 协议**重评估**，`protocol.json`
  未记录训练轮次，因此 `epochs_run` / `best_epoch` / `batch_size` / `seed`
  为空；原始 `config.json` 保留在本地实验目录，不随本结果包发布。
- SOPNet 条目是本文模型（非基线）；`name` 为空是原表结构使然，
  `model` 列给出 `SOPNet`。
- EQPolarity 的协议记录为 `epochs_run = 5`、`best_epoch = 5`。
- `checkpoint` 与 `predictions` 是仓库相对路径；对应权重与逐样本预测**不公开**。

### `ablation_results.csv`（4 行）

字段：`experiment, accuracy, macro_f1, precision, recall, mcc, n_known`。
`experiment` 取值为 `A_cls_ud`、`B_field`、`C_polarity`、`D_full`。

### `bootstrap.csv`（5 行）

事件级聚类配对 bootstrap（整事件重采样，1000 次，95% CI，seed `20261004`）：

| 字段 | 含义 |
|---|---|
| `baseline` | 对照模型键名 |
| `sopnet_macro_f1`, `baseline_macro_f1` | 观测到的 macro-F1 |
| `delta` | SOPNet − baseline |
| `ci_low`, `ci_high` | 95% 百分位区间 |
| `n_resamples`, `confidence` | 1000 / 0.95 |
| `p_improvement` | 重采样中 delta > 0 的比例 |
| `n_groups` | 重采样的事件数（63,044） |

### `snr_all.csv`（36 行 = 6 模型 × 6 噪声档）

字段：`model, accuracy, precision, recall, f1, mcc, support, macro_f1_ud, n,
n_known, snr_db`。

- `snr_db = inf` 表示 **Clean**（不添加噪声），不是 0 dB。
- `support = n_known = 12,520`；`n = 30,000`。
- `accuracy` 为严格 U/D 准确率；`macro_f1_ud` 为 Up/Down macro-F1。

### `p_shift_all.csv`（78 行 = 6 模型 × 13 偏移点）

字段与 `snr_all.csv` 相同，但最后一列为 `shift_seconds`（秒）。

- `0.01` 表示 10 ms；图件横轴以 ms 显示。
- 负值表示参考 P 窗口相对标注向前移动，正值反之。
- 13 点细网格已包含原粗网格的 ±200/±100/±50 ms；没有单独的旧粗网格文件，
  原始结果未被覆盖或丢弃。

---

## 3. 模型列表

| 键名 | 显示名 | 角色 | 输入长度 | 参数量 |
|---|---|---|---:|---:|
| `sopnet` | SOPNet | 本文带符号场模型（最终配方） | 400 | 1,440,417 |
| `cfm` | CFM | 卷积极性分类基线（遗留 checkpoint，strict 重评估） | 160 | 410,981 |
| `rpnet` | RPNet | 稳健 P 波极性基线 | 400 | 6,044,338 |
| `ross` | Ross | 三分类（U/D/Unknown）深度学习基线 | 400 | 3,665,731 |
| `eqpolarity` | EQPolarity | CCT 极性分类基线 | 600 | 1,159,601 |
| `diting_motion` | DiTingMotion | 运动特征极性基线 | 128（双通道） | 30,146 |

本结果包的评估未加载任何预训练权重。CFM 的 checkpoint 早于统一重训轮次，
但使用完全相同的测试集与 strict U/D 协议重评估。checkpoint 路径与 SHA-256
见 `provenance/experiment_manifest.json`（权重本身不发布）。

---

## 4. 图件生成

`figures/robustness_heatmaps.{pdf,svg,png}` 由以下命令一次生成：

```bash
python scripts/plot_paper_robustness_heatmaps.py \
  --snr paper/results/tables/snr_all.csv \
  --p-shift paper/results/tables/p_shift_all.csv \
  --output-dir paper/results/figures
```

图件规格：

- **(a) Noise robustness**：行 = 六个模型（按主表强弱顺序），列 = Clean、
  20、10、5、0、−5 dB。
- **(b) Sensitivity to P-arrival shifts**：同样的行，列 = 实际 13 点偏移网格
  （−200 至 +200 ms）。
- 两面板共享同一颜色标尺（40–100%），颜色与标注数字一一对应。
- 每个单元格标注严格 U/D 准确率，保留一位小数（百分数）。
- 缺失的模型−条件组合显示为 `NA`；脚本不会插值或补造数值。
- 配色为低饱和蓝绿顺序色带，对常见色觉缺陷友好；未绘制任何无计算依据的
  误差棒/置信区间。
- 字体：Nimbus Roman（Times New Roman 的度量兼容衬线字体；回退
  Times New Roman / DejaVu Serif），主要字号 9–10 pt。
- 成品宽度 17.0 cm；PNG 600 dpi（4015 px 宽），PDF/SVG 为矢量；SVG 文字
  转路径，脱离本机字体也能保持版式一致。
- 图中不含数学公式。

脚本只读取上述两份 CSV；每个数字都能追溯到对应 CSV 行。`pytest` 中的
`test_heatmap_data_matches_csv` 会逐格核对矩阵与 CSV 值。

---

## 5. 核验与溯源

```bash
# 数据完整性、口径一致性、图件逐格可追溯、manifest 哈希
python -m pytest tests/test_paper_results.py -q

# 代码质量
ruff check scripts/plot_paper_robustness_heatmaps.py tests/test_paper_results.py

# 文件校验和
sha256sum paper/results/tables/*.csv \
         paper/results/figures/robustness_heatmaps.*
```

`tests/test_paper_results.py` 覆盖：

- 七份 CSV 可读、必需字段存在、比率范围合法；
- 六个模型在 SNR（6 档）与 P 偏移（13 点）网格上各恰有一条记录；
- Clean 与零偏移的严格准确率逐模型完全一致；
- 逐源 `n_known` 之和 = 327,231；主表/消融/ bootstrap 相互一致；
- D 行与主表 SOPNet 行一致；C 行使用收敛后的终版数值；
- 图件矩阵逐格匹配 CSV；缺失值保留为 NaN 并渲染为 NA；
- manifest 中每份 CSV 的 SHA-256 与字节数匹配当前文件。

`provenance/experiment_manifest.json` 记录：源码提交、checkpoint 路径与
SHA-256、训练/数据/自助法种子、测试与子集样本数、噪声与偏移设置、各 CSV
与图件的 SHA-256，以及结果生成/核验命令。

---

## 6. 已知限制

1. **鲁棒性只覆盖 30,000 条固定子集**（12,520 条已知 U/D），不是完整测试集；
   因此 Clean 行的 0.9768 与主表完整测试集的 0.9782 不完全相同，这是子集
   抽样差异，不是口径变化。
2. **未计算鲁棒性置信区间**，图件与 CSV 均不包含误差棒；模型间差异应结合
   主表的事件级 bootstrap 结果阅读。
3. **P 偏移下所有模型快速退化**：±10 ms 已出现明显下降，±50 ms 时六个模型
   仅约 0.59–0.71，±100 ms 接近随机。该现象是固定窗口分类范式的系统性问题，
   不应解读为某一模型独有缺陷。
4. **消融 C 是续训终版**（best epoch 49 / 50 轮），不再使用早期欠训练数值；
   A/B 的 `best_epoch` 未被运行日志完整记录，因此 manifest 不臆造该字段。
5. **CFM 的 checkpoint 早于统一重训轮次**，仅按 strict 协议重评估；
   `baseline_protocol.csv` 中其训练轮次为空。
6. **EQPolarity 协议记录为 5 轮**（best epoch 5）；该事实如实保留，不做额外
   方法学解读。
7. **TXED（143）与 INSTANCE（104）测试已知样本极少**，逐源数字的方差很大，
   不应单独用于模型排序。
8. 主指标只覆盖已知 U/D 样本；Unknown 拒绝/选择性预测是补充分析，不等价于
   完整的开集识别评估。
9. **A/B 消融运行在 50 轮调度完成前停止**，且运行元数据没有记录实际完成轮次
   （`metrics.json` 缺失，`best_epoch` 未写入）。本包发布的是磁盘上的
   best-validation checkpoint；C 已从 epoch 6 续训并完成 50 轮，D 复用最终
   SOPNet。A/B 的绝对值应按“未完全收敛”解读，A/B 与 C/D 的差距不能全部
   归因于损失项。此限制不影响 CSV 的真实性，但影响消融归因的强度。

---

## 7. 发布范围与许可

- 本目录只发布聚合统计与图件；不发布原始波形、模型权重、缓存、逐样本预测、
  绝对路径或访问凭据。
- 底层数据集（SCSN、DiTing、PNW、INSTANCE、TXED）仍受各自原始许可与引用
  要求约束；使用本结果包中的聚合数字时请同时引用对应数据集与论文。
- 仓库代码遵循根目录 `LICENSE`；本结果包的图件与表格用于论文展示与复核。
