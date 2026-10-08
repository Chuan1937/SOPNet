# SOPNet v1 结果汇总

最终模型：**SOPNet 带符号场，无 P jitter，known-only 场损失**
（`configs/experiments/sopnet_nojitter.yaml`，checkpoint
`outputs/runs/sopnet_nojitter_36/best.pt`，seed 36，50 epoch，1.44 M 参数）。

## 统一 test split（777,720 条样本，其中 327,231 条有 U/D 标签）

| 模型 | U/D acc | F1 | MCC |
|---|---:|---:|---:|
| **SOPNet（无 jitter）** | **0.9782** | **0.9762** | **0.9561** |
| cfm | 0.9729 | 0.9704 | 0.9454 |
| eqpolarity | 0.9728 | 0.9703 | 0.9453 |
| rpnet | 0.9694 | 0.9665 | 0.9384 |
| ross | 0.8983 | 0.8772 | 0.8056 |
| diting_motion | 0.8935 | 0.8719 | 0.7945 |

所有基线都在完全相同的统一训练集（6,230,917 条样本）上从零重训；没有使用任何
官方预训练权重，因此不存在测试集污染。

按数据源准确率（SOPNet）：SCSN 0.9791、DiTing 0.9795、PNW 0.9527、
INSTANCE 0.9712、TXED 0.5874（n=143，低信噪比子集）。

P 波定位：MAE 6.4 样本（64 ms），中位 7（70 ms），p95 11（110 ms）。

## 配对 bootstrap（macro-F1 U/D，1000 次重采样）

| 对比 | delta | 95% 置信区间 | p(提升) |
|---|---:|---|---:|
| vs cfm | +0.0053 | [0.0049, 0.0057] | 1.0 |
| vs eqpolarity | +0.0054 | [0.0050, 0.0058] | 1.0 |
| vs rpnet | +0.0088 | [0.0084, 0.0093] | 1.0 |
| vs ross | +0.1105 | [0.1096, 0.1114] | 1.0 |
| vs diting_motion | +0.1116 | [0.1106, 0.1126] | 1.0 |

## 鲁棒性

- **SNR**（干净 → −5 dB）：0.978 → 0.961（20 dB）→ 0.936（10）→ 0.909（5）
  → 0.862（0）→ 0.778（−5 dB）。退化平缓。
- **P 窗口偏移**：0 s 时 0.978，但 ±0.1 s 即掉到 ~0.49-0.56，更大偏移约 ~0.50。

### 讨论要点（精度 vs 平移容忍度）

带 P 位置 jitter 的训练会迫使场模型同时求解定位与极性：代价是早期收敛慢约
0.5-7 个点、最终精度低约 0.5 个点（jitter 版 epoch 2 为 0.932，无 jitter 版为
0.978），但换来极性读出对窗口平移的容忍。发布的 v1 模型为追求最高精度刻意去掉
了 jitter，因此要求 P 到时的精度在几个样本以内；P 偏移鲁棒性是其主要局限，
部署时可与 SOPNet 自身的场峰值或专用拾取器配合来缓解。该权衡在论文讨论中明确
报告。

## 产物

`outputs/paper/tables/{main_results,bootstrap,complexity,p_shift_sopnet,snr_sopnet}.csv`，
`outputs/paper/figures/{fig_p_shift_sopnet,fig_snr_sopnet}.{pdf,png}`。
