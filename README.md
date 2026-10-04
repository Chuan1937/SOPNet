# SOPNet

**SOPNet (Signed-Onset Polarity Network)** determines the P-wave first-motion polarity
of a single vertical-component seismogram by predicting a **signed onset field**
instead of a discrete class label.

Given a 4 s window `x(t)` sampled at 100 Hz, the network predicts

```
f(x) : R^400 -> [-1, 1]^400
```

which is trained against the signed Gaussian target

```
g(t) = s * exp(-(t - t_P)^2 / (2 * sigma^2)),   s = +1 (Up), -1 (Down), 0 (Unknown)
```

P onset, polarity, and confidence are all read from one output field:

```
t_hat = argmax_t |f(x)(t)|,   s_hat = sign(f(x)(t_hat)),   C = max_t |f(x)(t)|
```

## Unified training data

Five datasets (SCSN, TXED, INSTANCE, DiTing, PNW) are logically merged into one
manifest. `source` is retained for label mapping and auditing only and is never
given to the model. Canonical labels are physical: `Up = +1`, `Down = -1`,
`Unknown = 0`.

| source | raw labels | canonical mapping | native rate | component order |
|---|---|---|---|---|
| SCSN | 0/1/2 | 0=D, 1=U, 2=X | 100 Hz | flat `X` |
| TXED | U/D/unknown | U=+1, D=-1, else 0 | 100 Hz | ZNE |
| INSTANCE | positive/negative/undecidable | pos=+1, neg=-1, else 0 | 100 Hz | ZNE |
| DiTing | U/D/R/C/other | U,C=+1; D,R=-1; else 0 | 50 Hz | Z (index 0) |
| PNW | positive/negative/undecidable | pos=+1, neg=-1, else 0 | 100 Hz | ENZ (Z index 2) |

The DiTing `R`/`C` assignment is verified empirically in `docs/data_protocol.md`.

## Pipeline

```
raw data (read-only /mnt/d/AI_Seismic_Data)
  -> manifest.parquet           (metadata only, no waveform copies)
  -> sharded HDF5 cache         (600 samples @ 100 Hz, P at sample 300)
  -> UnifiedPolarityDataset     (random 400-sample crop, signed-field targets)
  -> SOPNet / SOPNet-Cls
  -> baselines, ablation, robustness, bootstrap
```

## Quick start

```
pip install -e .
python scripts/inspect_data.py --data-root /mnt/d/AI_Seismic_Data
python scripts/build_manifest.py --data-root /mnt/d/AI_Seismic_Data
python scripts/build_cache.py --data-root /mnt/d/AI_Seismic_Data --workers 8
python scripts/train.py --config configs/experiments/sopnet_full.yaml \
    --limit-train 20000 --limit-val 5000 --epochs 3
```

## Layout

```
sopnet/{data,models,losses,training,evaluation,utils}
configs/{data,model,train,experiments}
scripts/    tests/    docs/
outputs/    (gitignored)
```

## License

BSD-3-Clause. See `LICENSE`.
