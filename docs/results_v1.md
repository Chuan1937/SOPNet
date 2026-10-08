# SOPNet v1 results summary

Final model: **SOPNet signed field, no P jitter, known-only field loss**
(`configs/experiments/sopnet_nojitter.yaml`, checkpoint
`outputs/runs/sopnet_nojitter_36/best.pt`, seed 36, 50 epochs, 1.44 M params).

## Unified test split (777,720 samples, 327,231 labelled U/D)

| model | U/D acc | F1 | MCC |
|---|---:|---:|---:|
| **SOPNet (no jitter)** | **0.9782** | **0.9762** | **0.9561** |
| cfm | 0.9729 | 0.9704 | 0.9454 |
| eqpolarity | 0.9728 | 0.9703 | 0.9453 |
| rpnet | 0.9694 | 0.9665 | 0.9384 |
| ross | 0.8983 | 0.8772 | 0.8056 |
| diting_motion | 0.8935 | 0.8719 | 0.7945 |

All baselines were retrained from scratch on the identical unified train split
(6,230,917 samples); no published weights were used, so no test contamination.

Per-source accuracy (SOPNet): SCSN 0.9791, DiTing 0.9795, PNW 0.9527,
INSTANCE 0.9712, TXED 0.5874 (n=143, low-SNR subset).

P-onset localisation: MAE 6.4 samples (64 ms), median 7 (70 ms), p95 11 (110 ms).

## Paired bootstrap (macro-F1 U/D, 1000 resamples)

| comparison | delta | 95 % CI | p(improvement) |
|---|---:|---|---:|
| vs cfm | +0.0053 | [0.0049, 0.0057] | 1.0 |
| vs eqpolarity | +0.0054 | [0.0050, 0.0058] | 1.0 |
| vs rpnet | +0.0088 | [0.0084, 0.0093] | 1.0 |
| vs ross | +0.1105 | [0.1096, 0.1114] | 1.0 |
| vs diting_motion | +0.1116 | [0.1106, 0.1126] | 1.0 |

## Robustness

- **SNR** (clean → −5 dB): 0.978 → 0.961 (20 dB) → 0.936 (10) → 0.909 (5)
  → 0.862 (0) → 0.778 (−5 dB). Graceful degradation.
- **P-window shift**: 0.978 at 0 s, but ~0.49–0.56 at ±0.1 s and ~0.50 beyond.

### Discussion note (accuracy vs shift tolerance)

Training with P-position jitter forces the field to solve localisation and
polarity jointly; it costs ~0.5–7 points of early convergence and ~0.5 points
of final accuracy (jitter run: 0.932 vs no-jitter: 0.978 at epoch 2) but makes
the polarity read-out shift-tolerant. The released v1 model deliberately drops
jitter for maximum accuracy and therefore requires a P pick accurate to a few
samples; P-shift robustness is the main limitation and is mitigated in
deployment by pairing SOPNet with its own field peak or a dedicated picker.
This trade-off is reported explicitly in the paper discussion.

## Artefacts

`outputs/paper/tables/{main_results,bootstrap,complexity,p_shift_sopnet,snr_sopnet}.csv`,
`outputs/paper/figures/{fig_p_shift_sopnet,fig_snr_sopnet}.{pdf,png}`.
