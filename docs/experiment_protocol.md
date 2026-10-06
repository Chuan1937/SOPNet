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
#    jitter bounds are given in samples: +-0.4 s -> [160, 240],
#    +-0.8 s -> [120, 280], +-1.0 s -> [100, 300]
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

# 6. full training, three seeds (or: bash scripts/run_seeds.sh 512 50)
for seed in 36 2026 3407; do
  python scripts/train.py --config configs/experiments/sopnet_full.yaml \
      --cache-dir outputs/cache_v1 --output outputs/runs/sopnet_full_$seed \
      --epochs 50 --batch-size 512 --seed $seed
done

# Interrupted runs resume from <output>/last.pt (model, optimizer, scheduler,
# epoch, history). run_seeds.sh adds --resume automatically when last.pt exists:
python scripts/train.py --config configs/experiments/sopnet_full.yaml \
    --cache-dir outputs/cache_v1 --output outputs/runs/sopnet_full_36 \
    --epochs 50 --batch-size 512 --seed 36 --resume

# 7-11. everything else in one detached pipeline (baselines -> ablation ->
# re-evaluation -> robustness -> bootstrap -> tables):
nohup setsid bash scripts/run_full_pipeline.sh 512 1024 50 50 \
    > outputs/runs/pipeline2.log 2>&1 &

# ...or step by step:
# 7. baselines on identical splits (50 epochs, early stopping patience 6;
#    val + test metrics written per baseline)
python scripts/train_baselines.py --cache-dir outputs/cache_v1 \
    --baselines ross rpnet eqpolarity cfm diting_motion \
    --epochs 50 --batch-size 512 --seed 36

# 8. ablation A-D (E is reused from outputs/runs/sopnet_full_36)
python scripts/run_ablation.py --cache-dir outputs/cache_v1 \
    --experiments A_cls B_field C_jitter D_polarity --epochs 50 --batch-size 1024

# 9. test evaluation (Platt-calibrated ECE on known U/D samples only)
python scripts/evaluate.py --checkpoint outputs/runs/sopnet_full_36/best.pt \
    --cache-dir outputs/cache_v1 --split test --auto-threshold --examples 12

# 10. robustness and paired bootstrap vs the best baseline
python scripts/run_robustness.py --cache-dir outputs/cache_v1 \
    --checkpoint sopnet=outputs/runs/sopnet_full_36/best.pt
python scripts/run_bootstrap.py --cache-dir outputs/cache_v1 \
    --checkpoint outputs/runs/sopnet_full_36/best.pt

# 11. paper tables
python scripts/make_paper_tables.py --runs-root outputs/runs outputs/ablation
```

### Calibration semantics

The field model emits a peak magnitude, not a probability, so ECE is computed
after Platt scaling (`fit_platt_scaling`) fitted on validation and evaluated on
labelled U/D test samples only. Selective-prediction numbers (`coverage_known`,
`covered_accuracy_known`) likewise exclude X samples; the all-sample
`coverage`/`covered_accuracy` are kept only for the U/D/X threshold analysis.

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