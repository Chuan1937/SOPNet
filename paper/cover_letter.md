# Cover Letter (draft)

**To:** Editor-in-Chief, *Applied Geophysics* (Springer)
**Subject:** Submission of research article — "SOPNet: A Signed-Onset Field
Network for Automatic P-Wave First-Motion Polarity Determination"

Dear Editor,

We are pleased to submit our manuscript "SOPNet: A Signed-Onset Field Network
for Automatic P-Wave First-Motion Polarity Determination" for consideration as
a research article in *Applied Geophysics*.

First-motion polarity is the key observable linking single-station waveforms
to earthquake focal mechanisms. While deep-learning models now determine
polarity with high accuracy, the field suffers from two practical problems
that our work addresses. First, almost all published polarity models treat the
problem as a categorical (up/down/unknown) classification task, discarding the
continuous, signed nature of the first motion and providing no direct,
differentiable link between onset localisation and the polarity decision.
Second, benchmark comparability is poor — models are trained on different
datasets with different splits and unknown-label policies — and robustness to
noise and to errors in the reference P arrival is rarely quantified.

Our contributions are:

- **A signed-onset-field formulation.** SOPNet is a compact 1-D
  encoder–decoder that predicts a continuous signed field over the waveform
  window; the P onset is the field maximum and the polarity its sign. The
  field is trained only on records with known polarity, with a
  polarity-consistency loss and a physical inversion-consistency constraint
  f(−x) = −f(x).
- **A unified, event-level benchmark.** We harmonise five open datasets
  (SCSN, DiTing, INSTANCE, PNW, TXED; 7.78M waveforms, 1.12M events) and
  retrain five public baselines (Ross, DiTingMotion, RPNet, EQPolarity, CFM)
  from scratch under their official input protocols on identical splits,
  compared with a strict up/down decision metric and event-clustered paired
  bootstrap statistics.
- **A quantitative robustness study.** All six models are evaluated under
  additive noise and reference-P shifts on a fixed test subset, and the
  accuracy–tolerance trade-off of fixed-window polarity classification is
  reported explicitly rather than concealed.

SOPNet reaches 97.8% accuracy and 0.956 MCC on 327,231 known-polarity test
waveforms, outperforming all five retrained baselines with bootstrap intervals
excluding zero, while degrading gracefully under noise. We believe the
combination of a physically motivated representation, a reproducible benchmark
and an honest robustness analysis fits the scope of *Applied Geophysics* and
will interest readers working on automated focal-mechanism determination and
machine-learning seismology.

The manuscript is original, has not been published previously and is not under
consideration elsewhere. All datasets used are openly available; code,
configurations and the benchmark manifest will be released upon acceptance.
The authors declare no competing interests. \TBD{author contributions and
funding acknowledgements}

Thank you for your consideration.

Sincerely,
\TBD{corresponding author, affiliation, contact}
