# 实验协议

> **v1 说明**：本文件描述完整实验协议。v1 最终模型采用无 jitter + known-only 场损失
> 配方（`configs/experiments/sopnet_nojitter.yaml`，单 seed 36），消融 A-E 未运行；
> 最终数字与 P 偏移鲁棒性讨论见 `docs/results_v1.md`。

## 流程与命令

```bash
# 0. 环境
nvidia-smi
python -c "import torch; print(torch.__version__, torch.cuda.is_available())"

# 1. 原始数据审计（含标签实测验证）
python scripts/inspect_data.py --data-root /mnt/d/AI_Seismic_Data --verify

# 2. 仅元数据的 manifest（778 万行，约 20 秒）
python scripts/build_manifest.py --data-root /mnt/d/AI_Seismic_Data

# 2b. 可选 I/O 基准测试
python scripts/benchmark_io.py --data-root /mnt/d/AI_Seismic_Data

# 3. 全量缓存（约 19 GB，单盘友好的 worker 池）
python scripts/build_cache.py --data-root /mnt/d/AI_Seismic_Data \
    --manifest outputs/manifest_raw.parquet --cache-dir outputs/cache_v1 \
    --workers 8 --shard-size 50000

# 4. 冒烟测试（任何全量运行前必须通过）
python scripts/train.py --config configs/experiments/sopnet_full.yaml \
    --cache-dir outputs/smoke/cache_v1 --output outputs/smoke/run_sopnet \
    --limit-train 20000 --limit-val 5000 --epochs 3 --batch-size 1024

# 5. 小规模超参搜索（20% 数据，约 10 epoch）
#    sigma 取 {5,10,15}；jitter 取 {0.4,0.8,1.0 s}；lambda_inv 取 {0,0.05,0.1}
#    jitter 边界以样本为单位：+-0.4 s -> [160, 240]，
#    +-0.8 s -> [120, 280]，+-1.0 s -> [100, 300]
for sigma in 5 10 15; do
  for jitter in "160 240" "120 280" "100 300"; do
    for linv in 0 0.05 0.1; do
      python scripts/train.py --config configs/experiments/sopnet_full.yaml \
          --cache-dir outputs/cache_v1 --seed 36 --epochs 10 --limit-train 40000 \
          --sigma $sigma --jitter $jitter --lambda-inv $linv \
          --output outputs/sweep/s${sigma}_j${jitter// /_}_i${linv}
    done
  done
done

# 6. 全量训练，三个 seed（或：bash scripts/run_seeds.sh 512 50）
for seed in 36 2026 3407; do
  python scripts/train.py --config configs/experiments/sopnet_full.yaml \
      --cache-dir outputs/cache_v1 --output outputs/runs/sopnet_full_$seed \
      --epochs 50 --batch-size 512 --seed $seed
done

# 中断的训练从 <output>/last.pt 续跑（模型、优化器、调度器、epoch、历史）。
# run_seeds.sh 在 last.pt 存在时自动加 --resume：
python scripts/train.py --config configs/experiments/sopnet_full.yaml \
    --cache-dir outputs/cache_v1 --output outputs/runs/sopnet_full_36 \
    --epochs 50 --batch-size 512 --seed 36 --resume

# 7-11. 其余实验放进一个后台流水线（基线 -> 消融 ->
# 重评估 -> 鲁棒性 -> bootstrap -> 表格）：
nohup setsid bash scripts/run_full_pipeline.sh 512 1024 50 50 \
    > outputs/runs/pipeline2.log 2>&1 &

# ...或分步执行：
# 7. 相同划分上的基线（50 epoch，早停 patience 6；
#    每个基线都会写出 val + test 指标）
python scripts/train_baselines.py --cache-dir outputs/cache_v1 \
    --baselines ross rpnet eqpolarity cfm diting_motion \
    --epochs 50 --batch-size 512 --seed 36

# 8. 消融 A-D（E 复用 outputs/runs/sopnet_full_36）
python scripts/run_ablation.py --cache-dir outputs/cache_v1 \
    --experiments A_cls B_field C_jitter D_polarity --epochs 50 --batch-size 1024

# 9. test 评估（仅在已知 U/D 样本上计算 Platt 校准后的 ECE）
python scripts/evaluate.py --checkpoint outputs/runs/sopnet_full_36/best.pt \
    --cache-dir outputs/cache_v1 --split test --auto-threshold --examples 12

# 10. 鲁棒性与对最优基线的配对 bootstrap
python scripts/run_robustness.py --cache-dir outputs/cache_v1 \
    --checkpoint sopnet=outputs/runs/sopnet_full_36/best.pt
python scripts/run_bootstrap.py --cache-dir outputs/cache_v1 \
    --checkpoint outputs/runs/sopnet_full_36/best.pt

# 11. 论文表格
python scripts/make_paper_tables.py --runs-root outputs/runs outputs/ablation
```

### 校准语义

场模型输出的是峰值幅度而不是概率，因此 ECE 先经 Platt 校准
（`fit_platt_scaling`，在验证集上拟合），并且只在有标签的 U/D 测试样本上评估。
选择性预测指标（`coverage_known`、`covered_accuracy_known`）同样排除 X 样本；
全样本的 `coverage`/`covered_accuracy` 仅保留用于 U/D/X 阈值分析。

## 冒烟测试验收门槛

在暂存缓存上运行（`outputs/smoke/cache_v1`，75,656 条样本，覆盖全部五个数据源）。
3 个 epoch 的参考运行记录：

| 检查项 | 要求 | 实测 |
|---|---|---|
| CUDA | 设备可用 | RTX 4060 Ti，torch 2.5.1+cu124 |
| 损失 | 下降 | 场损失 0.442 -> 0.373（val 0.411 -> 0.368） |
| NaN | 无 | 无 |
| 场峰值 | 单个带符号峰 | Up 平均峰 +0.394；Down −0.327 |
| 极性 | 符号正确 | 3 个 epoch 后 71%（Up）/ 67%（Down） |
| P 到时 | 接近参考 | 中位 5-6 样本（50-60 ms），82% 在 20 样本以内 |

`outputs/smoke/run_sopnet/` 包含 `train.log`、`metrics.json`、`best.pt`、
`last.pt`、`val_metrics.json` 和 `examples.png/pdf`（12 个随机验证样本，含波形、
目标场、预测场、真实/预测 P、置信度）。高精度需要完整的 50 epoch 训练；冒烟门槛
检验的是学习信号的正确性，而不是最终性能。

## 指标

主指标：Macro-F1。同时报告：已知极性子集上的 Accuracy、Precision、Recall、MCC，
以及应用验证集调出的置信度阈值后的 `macro_f1_udx` 和覆盖率/覆盖-准确率曲线。
SOPNet 还报告 P 到时 MAE、中位数和 95 分位误差（样本数与毫秒）以及期望校准误差。

## 消融矩阵

| 实验 | field | jitter | 极性损失 | 反转一致性 |
|---|---:|---:|---:|---:|
| A SOPNet-Cls | x | x | x | x |
| B 带符号场 | ✓ | x | x | x |
| C + P jitter | ✓ | ✓ | x | x |
| D + 极性损失 | ✓ | ✓ | ✓ | x |
| E 完整 SOPNet | ✓ | ✓ | ✓ | ✓ |

配置位于 `configs/experiments/`；`scripts/run_ablation.py` 依次执行并写出
`outputs/paper/tables/ablation.csv`。

## 鲁棒性

只在统一 test split 上评估，绝不按数据源拆分：

- **P 窗口偏移**：0、±0.1、±0.2、±0.5、±1.0 秒。
- **噪声**：干净、20、10、5、0、−5 dB。

`scripts/run_robustness.py` 写出 `p_shift_*.csv`、`snr_*.csv` 和 600-dpi 的
PDF/PNG 图。v1 的 P 偏移结果较差（±0.1 s 即掉到随机水平），已作为
"精度 vs 平移容忍度"的权衡写入论文讨论，见 `docs/results_v1.md`。

## 统计与可复现性

- 最终模型：seed 36、2026、3407；报告 mean ± std。
- SOPNet 对最优基线：在固定 test 集上做 1000 次配对 bootstrap
  （`sopnet.evaluation.bootstrap`），报告 ΔF1 及其 95% 置信区间。
- 每次运行都记录 git commit、完整 YAML 配置、seed、Python/PyTorch/CUDA 版本、
  GPU 型号、训练时长、参数量和峰值显存（`metrics.json` + `config_resolved.yaml`）。
- checkpoint 与缓存都留在 `outputs/` 下（gitignored）。

## 论文产物

```
outputs/paper/tables/{main_results, ablation, p_shift, snr, complexity, bootstrap}.csv
outputs/paper/figures/{fig_signed_label, fig_prediction_examples, fig_p_shift,
                       fig_snr, fig_confusion_matrix, fig_calibration}.pdf|.png
```

复杂度和主结果表由各次运行的 `metrics.json` 汇总；图由 `evaluate.py`、
`run_robustness.py` 和 `run_ablation.py` 直接输出。
