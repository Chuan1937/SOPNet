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
| macro-F1 (U/D) | 0.977931 |
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

### 图件

| 图 | 文件 | 生成命令 |
|---|---|---|
| Fig 1 架构 | `outputs/paper/figures/fig1_architecture.{pdf,png}` | `python scripts/make_fig_architecture.py` |
| Fig 2 定性示例 | `outputs/paper/figures/fig2_examples.{pdf,png}` | `python scripts/make_fig_examples.py --device cpu`（固定 seed 11，前 6000 测试样本中选取） |

## 待生成数字（自动收尾链 / G2 / G3）

| 产物 | 依赖 | 状态 |
|---|---|---|
| `outputs/paper/tables/main_results.csv` | G1 五基线重训完成 | 待生成（收尾链） |
| `outputs/paper/tables/per_source_results.csv` | 六模型逐样本预测 | 待生成（收尾链） |
| `outputs/paper/tables/baseline_protocol.csv` | 各基线 protocol.json | 待生成（收尾链） |
| `outputs/paper/tables/bootstrap.csv` | 事件级聚类配对 bootstrap | 待生成（收尾链） |
| `outputs/paper/tables/ablation_results.csv` | G2 A/B/C 训练 + 测试评估 | 待生成（G2） |
| `outputs/paper/tables/snr_all.csv`、`p_shift_all.csv` | G3 六模型鲁棒性 | 待生成（G3） |
| Fig 3/4（SNR/P 偏移曲线） | 同上 | 待生成（G3） |
| Fig 5（校准/选择性预测） | SOPNet val/test 预测 | 待生成（G3 或 G4） |

## 核验规则

1. 论文正文/表格中的每个数字必须出现在本文件或由本文件列出的命令重新生成。
2. 旧协议（非 strict）数字一律不得进入论文；旧 checkpoint 已归档
   `outputs/retired/`。
3. 任何重算若与已记录数字不一致，以重算为准并在此更新，同时检查
   `docs/paper/EXPERIMENT_LOG.md` 是否记录了原因。
