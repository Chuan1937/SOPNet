# 结果核验文档（RESULT_VERIFICATION）

论文中每个数字必须能追溯到"checkpoint + 测试样本 + 命令"。本文件随 G1-G3
推进持续更新；状态列：`已验证` / `待生成` / `待复算`。

## 指标口径

- **主指标（strict U/D）**：在已知极性（±1）测试样本上，用模型自身 U/D 得分
  判定的准确率、macro-F1（Up/Down 两类平均）、precision、recall、MCC。
  三分类基线的 "unknown" 预测在主指标中按错误计；unknown 率单独作为补充。
- **SOPNet 场模型**：极性 = sign(f(argmax|f|))；阈值只用于补充的选择性指标，
  不参与主指标。
- **P 位置误差**：固定中心窗口下的内部一致性诊断，不当作独立拾取能力。

## 指标口径（重要）

- `macro_f1_ud`（主表/消融/论文引用）：**无阈值 strict 口径**——全部已知
  U/D 样本，用模型自身 U/D 得分判定。
- `macro_f1_ud_selected` / `coverage*` / `covered_accuracy*`：验证集调阈后的
  选择性操作点，仅作补充分析，不进入主表。
- 2026-10-09 修复记录：此前 `evaluate_field` 会用阈值化版本覆盖
  `macro_f1_ud`（B 组曾被压低至 0.9436）；修复后 B/D 在收尾阶段重评估刷新。

## 已验证数字

### 数据集统计（Table 1）

| 量 | 值 | 来源 |
|---|---|---|
| 总波形 | 7,781,561 | `outputs/cache_v1/manifest.parquet` 聚合 |
| 划分（train/val/test） | 6,230,917 / 772,924 / 777,720 | 同上（事件级划分，seed 20261004） |
| 测试事件数 | 111,588 | 同上 |
| 测试已知 U/D | 327,231 | 同上 |
| 逐源测试已知 U/D | SCSN 253,194；DiTing 64,048；PNW 9,742；TXED 143；INSTANCE 104 | 同上 |

### SOPNet 最终模型（`outputs/runs/sopnet_nojitter_36/best.pt`，seed 36，best epoch 43）

测试集 `test_metrics.json`（`scripts/evaluate.py --split test --auto-threshold`）：

| 指标 | 值 |
|---|---|
| n / n_known | 777,720 / 327,231 |
| known accuracy | 0.978196 |
| macro-F1 (U/D) | 0.978042（20:43 口径修复后刷新） |
| precision / recall | 0.977001 / 0.975405 |
| binary F1 | 0.976202 |
| MCC | 0.956085 |
| P MAE | 6.39 样本（63.9 ms），median 70 ms，p95 110 ms |
| 阈值 | 0.95 |
| coverage（全体/已知） | 0.9804 / 0.9948 |
| covered accuracy（已知） | 0.9806 |
| ECE（Platt，known-only） | 0.0307（a=4.602, b=-0.674） |

复现命令：

```bash
python scripts/evaluate.py \
  --checkpoint outputs/runs/sopnet_nojitter_36/best.pt \
  --cache-dir outputs/cache_v1 --split test --auto-threshold --examples 12 \
  --save-predictions outputs/paper/predictions/sopnet_test.npz
```

### CFM（strict 协议重评估，无重训）

`outputs/runs/baseline_cfm/test_metrics.json`：

| 指标 | 值 |
|---|---|
| accuracy | 0.972909 |
| macro-F1 (U/D) | 0.972718 |
| precision / recall | 0.971138 / 0.969733 |
| MCC | 0.945437 |
| unknown 率 | 0.0（原生二分类） |
| 参数量 / 输入长度 | 410,981 / 160 |

### G1 重训基线（2026-10-09，strict 协议）

重训命令（串行队列 `outputs/runs/g1_pipeline.log`）：
`python scripts/train_baselines.py --cache-dir outputs/cache_v1 --baselines <name>
--epochs 50 --batch-size 512/1024 --num-workers 8 --window-length 600 --seed 36`
（diting_motion 使用 batch 1024 加速；其余 512）

| 模型 | 完成时间 | test acc | macro-F1 (U/D) | MCC | unknown 率 | 参数量 / 输入 |
|---|---|---|---|---|---|---|
| DiTingMotion | 01:24 | 0.963414 | 0.963144 | 0.926301 | 0.5927 | 30,146 / 128（双通道） |
| Ross | 02:53 | 0.969428 | 0.969208 | 0.938421 | 0.5989 | 3,665,731 / 400 |
| RPNet | 10:50 | 0.972823 | 0.972633 | 0.945266 | 0.0 | 6,044,338 / 400 |
| EQPolarity | 进行中（10:51 起） | — | — | — | — | — / 600 |

复现评估命令（任意基线）：

```bash
python scripts/evaluate_baselines.py --baselines <name> --num-workers 8
```

逐样本预测：`outputs/paper/predictions/{name}_{val,test}.npz`
（字段：sample_ids, labels, ud_score, native_class, unknown_flag, confidence）。

### 图件

| 图 | 文件 | 生成命令 |
|---|---|---|
| Fig 1 架构 | `outputs/paper/figures/fig1_architecture.{pdf,png}` | `python scripts/make_fig_architecture.py` |
| Fig 2 定性示例 | `outputs/paper/figures/fig2_examples.{pdf,png}` | `python scripts/make_fig_examples.py --device cpu`（固定 seed 11，前 6000 测试样本中选取） |

## 待生成数字（自动收尾链 / G2 / G3）

| 产物 | 依赖 | 状态 |
|---|---|---|
| `outputs/paper/tables/main_results.csv` | G1 六模型 | ✅ 已验证（20:43 刷新） |
| `outputs/paper/tables/per_source_results.csv` | 六模型逐样本预测 | ✅ 已验证 |
| `outputs/paper/tables/baseline_protocol.csv` | 各基线 protocol.json | ✅ 已验证 |
| `outputs/paper/tables/bootstrap.csv` | 事件级聚类配对 bootstrap | ✅ 已验证（5 组 CI 均排除 0） |
| `outputs/paper/tables/ablation_results.csv` | G2 A/B/C/D | ✅ 已验证（C 完整收敛：0.9781 / 0.9779 / 0.9558） |
| `outputs/paper/tables/snr_all.csv`、`p_shift_all.csv` | G3 六模型鲁棒性（含 13 点 P 偏移细网格） | ✅ 已验证 |
| Fig 3/4（SNR/P 偏移曲线） | 同上 | ✅ 已产出 |
| Fig 5（校准/选择性预测） | SOPNet 预测 | ✅ 已产出 |

## 核验规则

1. 论文正文/表格中的每个数字必须出现在本文件或由本文件列出的命令重新生成。
2. 旧协议（非 strict）数字一律不得进入论文；旧 checkpoint 已归档
   `outputs/retired/`。
3. 任何重算若与已记录数字不一致，以重算为准并在此更新，同时检查
   `docs/paper/EXPERIMENT_LOG.md` 是否记录了原因。

---

## 论文结果发布路径（paper/results/，2026-10-20）

最终可发布汇总已整理到 `paper/results/`，作为 GitHub `main` 上的正式结果包：

```text
paper/results/
├── README.md
├── tables/{main_results,per_source_results,baseline_protocol,
│           ablation_results,bootstrap,snr_all,p_shift_all}.csv
├── figures/robustness_heatmaps.{pdf,svg,png}
└── provenance/experiment_manifest.json
```

- `tables/*.csv` 是 `outputs/paper/tables/` 对应文件的逐字节副本；七份文件的
  行数、字段和 SHA-256 记录在 `experiment_manifest.json`。
- P 偏移细网格（±10/20/30 ms 与 ±50/100/200 ms，共 13 点）已合并在
  `p_shift_all.csv`；没有单独的旧粗网格文件，原始细网格结果未被覆盖。
- 消融表使用 C 组续训终版（best epoch 49 / 50 轮），不是早期欠训练的
  0.9736 版本。
- A/B 消融在 50 轮调度完成前停止，运行元数据未记录实际完成轮次；其
  checkpoint 为磁盘上的 best-validation checkpoint。该限制已写入
  `paper/results/README.md` 与 manifest，不将 A/B 的绝对值宣称为完全收敛。

### 数据一致性核验（2026-10-20）

| 检查项 | 结果 |
|---|---|
| 七份 CSV 可读、必需字段存在、比率范围合法 | ✅ |
| 六模型 × 6 SNR 档：每模型每档恰一条记录（36 行） | ✅ |
| 六模型 × 13 P 偏移点：每模型每点恰一条记录（78 行） | ✅ |
| Clean 与零偏移严格准确率逐模型一致（最大绝对差） | ✅ 0.0 |
| 主表 n / n_known | ✅ 777,720 / 327,231 |
| 逐源 n_known 之和（每模型） | ✅ 327,231 |
| D 行与主表 SOPNet 行一致 | ✅ |
| C 行使用续训终版（0.9781 / 0.9779 / 0.9558） | ✅ |
| 图中每格与 CSV 对应行逐值一致 | ✅ `test_heatmap_values_are_traceable_to_csv` |
| manifest 中各 CSV / 图件 SHA-256 与字节数 | ✅ `test_manifest_hashes_and_metadata` |
| 无绝对路径、波形、权重或凭据进入结果包 | ✅ |

复现命令：

```bash
python scripts/plot_paper_robustness_heatmaps.py \
  --snr paper/results/tables/snr_all.csv \
  --p-shift paper/results/tables/p_shift_all.csv \
  --output-dir paper/results/figures

python -m pytest tests/test_paper_results.py -q
ruff check scripts/plot_paper_robustness_heatmaps.py tests/test_paper_results.py
sha256sum paper/results/tables/*.csv paper/results/figures/robustness_heatmaps.*
```

图件规格：17.0 cm 宽、白底、Nimbus Roman（Times New Roman 度量兼容）9–10 pt、
两面板共享 40–100% 色标、低饱和蓝绿色带、缺失值标 NA、无无依据误差棒；
输出 PDF/SVG/PNG（PNG 600 dpi）。PDF 已确认嵌入 Nimbus Roman Type0 子集。

发布提交：`e3476bd09d062ac3d30b3789d08d657bf9befd41`（结果包首次发布）；
`experiment_manifest.json` 的 `publication_commit` 字段记录该提交 SHA。
