# Experiment protocol

## Pipeline and commands

```bash
# 0. environment
nvidia-smi
python -c "import torch; print(torch.__version__, torch.cuda.is_available())"

# 1. audit raw data (+ empirical label verification)
python scripts/inspect_data.py --data-root /mnt/d/AI_Seismic_Data --verify

# 2. metadata-only manifest (7.78 M rows, ~20 s)
python scripts/build_manifest.py --data-root /mnt/d/AI_Seismic_Data

# 2b. optional I/O benchmark
python scripts/benchmark_io.py --data-root /mnt/d/AI_Seismic_Data

# 3. full cache (~19 GB, single disk-friendly worker pool)
python scripts/build_cache.py --data-root /mnt/d/AI_Seismic_Data \
    --manifest outputs/manifest_raw.parquet --cache-dir outputs/cache_v1 \
    --workers 8 --shard-size 50000

# 4. smoke test (must pass before any full run)
python scripts/train.py --config configs/experiments/sopnet_full.yaml \
    --cache-dir outputs/smoke/cache_v1 --output outputs/smoke/run_sopnet \
    --limit-train 20000 --limit-val 5000 --epochs 3 --batch-size 1024

# 5. small hyper-parameter search (20 % data, ~10 epochs)
#    sigma in {5,10,15}; jitter in {0.4,0.8,1.0 s}; lambda_inv in {0,0.05,0.1}

# 6. full training, three seeds
for seed in 36 2026 3407; do
  python scripts/train.py --config configs/experiments/sopnet_full.yaml \
      --cache-dir outputs/cache_v1 --output outputs/runs/sopnet_full_$seed \
      --epochs 50 --batch-size 1024 --seed $seed
done

# 7. baselines on identical splits
python scripts/train_baselines.py --cache-dir outputs/cache_v1 \
    --baselines ross rpnet eqpolarity cfm diting_motion

# 8. test evaluation + ablation + robustness
python scripts/evaluate.py --checkpoint outputs/runs/sopnet_full_36/best.pt \
    --cache-dir outputs/cache_v1 --split test --auto-threshold --examples 12
python scripts/run_ablation.py --cache-dir outputs/cache_v1
python scripts/run_robustness.py --cache-dir outputs/cache_v1 \
    --checkpoint sopnet=outputs/runs/sopnet_full_36/best.pt
```

## Smoke-test acceptance gate

Run on the staged cache (`outputs/smoke/cache_v1`, 75,656 samples covering all
five sources). The 3-epoch reference run records:

| check | requirement | observed |
|---|---|---|
| CUDA | device available | RTX 4060 Ti, torch 2.5.1+cu124 |
| loss | decreasing | field loss 0.442 -> 0.373 (val 0.411 -> 0.368) |
| NaN | none | none |
| field peak | single signed peak | Up mean peak +0.394; Down −0.327 |
| polarity | correct sign | 71 % (Up) / 67 % (Down) after 3 epochs |
| P onset | close to reference | median 5-6 samples (50-60 ms), 82 % within 20 samples |

`outputs/smoke/run_sopnet/` contains `train.log`, `metrics.json`, `best.pt`,
`last.pt`, `val_metrics.json` and `examples.png/pdf` (12 random validation
samples with waveform, target field, predicted field, true/predicted P,
confidence). Reaching high accuracy requires the full 50-epoch run; the smoke
gate is about correctness of the learning signal, not final performance.

## Metrics

Primary: Macro-F1. Also reported: Accuracy, Precision, Recall, MCC on the
known-polarity subset, plus `macro_f1_udx` after applying the validation-tuned
confidence threshold and coverage/accuracy-versus-coverage. SOPNet additionally
reports P-arrival MAE, median and 95th percentile error (samples and ms), and
expected calibration error.

## Ablation matrix

| Experiment | field | jitter | polarity loss | inversion |
|---|---:|---:|---:|---:|
| A SOPNet-Cls | x | x | x | x |
| B signed field | ✓ | x | x | x |
| C + P jitter | ✓ | ✓ | x | x |
| D + polarity loss | ✓ | ✓ | ✓ | x |
| E full SOPNet | ✓ | ✓ | ✓ | ✓ |

Configs live in `configs/experiments/`; `scripts/run_ablation.py` executes them
and writes `outputs/paper/tables/ablation.csv`.

## Robustness

Unified test split only, never broken down by source:

- **P-window shift**: 0, ±0.1, ±0.2, ±0.5, ±1.0 s.
- **Noise**: clean, 20, 10, 5, 0, −5 dB.

`scripts/run_robustness.py` writes `p_shift_*.csv`, `snr_*.csv` and 600-dpi
PDF/PNG figures.

## Statistics and reproducibility

- Final models: seeds 36, 2026, 3407; report mean ± std.
- SOPNet vs best baseline: paired bootstrap over 1000 resamples of the fixed
  test set (`sopnet.evaluation.bootstrap`), reporting ΔF1 and its 95 % CI.
- Every run records git commit, full YAML config, seed, Python/PyTorch/CUDA
  versions, GPU model, training time, parameter count and peak GPU memory
  (`metrics.json` + `config_resolved.yaml`).
- Checkpoints and caches stay under `outputs/` (gitignored).

## Paper artefacts

```
outputs/paper/tables/{main_results, ablation, p_shift, snr, complexity, bootstrap}.csv
outputs/paper/figures/{fig_signed_label, fig_prediction_examples, fig_p_shift,
                       fig_snr, fig_confusion_matrix, fig_calibration}.pdf|.png
```

Complexity and main-results tables are assembled from per-run `metrics.json`
files; figures are emitted directly by `evaluate.py`, `run_robustness.py` and
`run_ablation.py`.