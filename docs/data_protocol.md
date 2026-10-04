# Data protocol

Raw data is mounted read-only at `/mnt/d/AI_Seismic_Data`. Original files are
never modified or copied in full.

## Unified dataset

| source | file(s) | native rate | component order | Z index | p_pick |
|---|---|---|---|---|---|
| SCSN | `scsn/scsn_p_2000_2017_6sec_0.5r_fm_combined.hdf5` | 100 Hz | flat `X (N,600)` | - | 300 (fixed) |
| TXED | `txed/Texd.hdf5` + `txed/Texd_filtered.csv` | 100 Hz | ZNE | 0 | `trace_p_arrival_sample` |
| INSTANCE | `Instance/INSTANCE.hdf5` + `Instance/INSTANCE_filtered.csv` | 100 Hz | ZNE | 0 | `trace_P_arrival_sample` |
| PNW | `pnw/pnw.hdf5` + `pnw/pnw.csv` | 100 Hz | ENZ | 2 | `trace_P_arrival_sample` |
| DiTing 1.0 | `谛听1.0-50HZ/Diting50hz/DiTing330km_part_{0..27}.{csv,hdf5}` | 50 Hz | Z (index 0) | 0 | `p_pick` |

Counts after scanning (seed-independent, metadata only): **7,781,561 samples**.

| source | U | D | X | total |
|---|---:|---:|---:|---:|
| SCSN | 1,108,531 | 1,417,416 | 2,321,301 | 4,847,248 |
| TXED | 742 | 640 | 9,310 | 10,692 |
| INSTANCE | 626 | 369 | 3,969 | 4,964 |
| PNW | 58,775 | 49,113 | 76,021 | 183,909 |
| DiTing | 345,875 | 295,150 | 2,093,723 | 2,734,748 |
| **total** | **1,514,549** | **1,762,688** | **4,504,324** | **7,781,561** |

## Canonical labels

`source` is used only for parsing and auditing; models never see it. Canonical
polarity is physical: `Up = +1`, `Down = -1`, `Unknown = 0`.

| source | raw | canonical | evidence |
|---|---|---|---|
| SCSN | `0` | Down | signed post-P mean −3.7e-4 (n=66 k) |
| SCSN | `1` | Up | signed post-P mean +2.6e-4 (n=67 k) |
| SCSN | `2` | Unknown | signed post-P mean ≈ 0 |
| TXED | `U/D/unknown` | +1/−1/0 | file metadata |
| INSTANCE | `positive/negative/undecidable` | +1/−1/0 | file metadata |
| PNW | `positive/negative/undecidable` | +1/−1/0 | file metadata |
| DiTing | `U` / `C` | Up | first-break positive 0.64 / 0.72 |
| DiTing | `D` / `R` | Down | first-break positive 0.39 / 0.42 |
| DiTing | other / missing | Unknown | no polarity code |

Two pitfalls were resolved by inspecting the real data:

1. **SCSN uses `0 = Down, 1 = Up`**, while the other datasets use
   `0 = Up, 1 = Down`. Raw integer concatenation across sources is therefore
   forbidden; `sopnet/data/canonical.py` enforces explicit per-source maps.
2. **DiTing `R` and `C` are real polarity codes**, not noise. The published
   DiTing polarity total is 641,025, exactly `U + R + D + C`; waveform
   first-break signs show `C` behaves like `U` (compression) and `R` like `D`
   (rarefaction). Mapping them recovers 593,554 otherwise lost labels.

Additional findings baked into the readers:

- INSTANCE must be indexed by `trace_name`; `trace_name_original` points at other
  records (verified by pick alignment on the vertical component).
- TXED/INSTANCE are ZNE (not the SeisPolarity default ENZ); PNW is ENZ.
  The component index comes from each file's `data_format/component_order`.
- SCSN's post-P mean is small but statistically unambiguous; per-trace first-break
  checks confirm the mapping.

## Preprocessing

```
component Z -> resample to 100 Hz -> demean -> linear detrend
-> zero-phase 1-45 Hz Butterworth (SOS) -> crop
-> max-abs normalisation (sign preserving)
```

Cache window: 600 samples (6 s), P at sample 300. Training window: 400 samples
(4 s), randomly cropped so P lies in samples 120-280. Only a short segment
around the pick is resampled for 50 Hz sources, so DiTing's 180 s traces are
never fully processed.

## Cache layout

```
outputs/cache_v1/
  manifest.parquet    sample_id, source, source_path, trace_id, row_index,
                      p_pick, sampling_rate, component, raw_label,
                      canonical_label, event_key, split_native,
                      waveform_hash, split
  index.parquet       manifest_index, shard, shard_index, sample_id, waveform_hash
  shard_00000.h5 ...  X [N,600] float32 (lzf), p_pick int16, label int8,
                      sample_index int64
  meta.json           config, counts, failures
```

Shards hold 50 k samples (~120 MB each, ~19 GB total). `manifest_index`
guarantees positional alignment between the manifest and the cache; a
regression test compares every cached window against its raw source.

## Splits

Fixed seed `20261004`, ratios 80/10/10. Samples are grouped by `event_key`
(SCSN `evids`, PNW `event_id`, INSTANCE `source_id`, TXED event name, DiTing
`part:ev_id`). Waveform duplicates, detected by a 64-bit blake2b hash of the
standardised 600-sample window, are unioned across events so identical traces
never span splits. `check_split_leakage` verifies both constraints; the current
full-manifest audit reports zero event leaks and zero duplicate leaks.

## Read-only guarantee

No script writes into `/mnt/d/AI_Seismic_Data`; all derived artefacts live under
`outputs/`. HDF5 files are opened read-only (`h5py.File(..., "r")`).