# SOPNet method

## Task formulation

Most polarity pickers treat first-motion determination as a discrete
classification problem. SOPNet instead predicts a **signed onset field**

```
f : R^T -> [-1, 1]^T,        T = 400 samples (4 s at 100 Hz)
```

trained against the signed Gaussian target

```
g(t) = s * exp(-(t - t_P)^2 / (2 sigma^2)),
s = +1 (Up), -1 (Down), 0 (Unknown)
```

with `sigma = 10` samples by default. P onset, polarity and confidence all come
from one output:

```
t_hat = argmax_t |f(x)(t)|,   s_hat = sign(f(x)(t_hat)),   C = max_t |f(x)(t)|
```

The representation has three practical advantages:

1. Polarity and P localisation are solved jointly; a confident polarity without
   a localised onset is impossible by construction.
2. `C` is a native confidence score, enabling selective prediction and a
   validation-tuned unknown threshold instead of a hard third class.
3. Robustness to a shifted reference window improves because the network has to
   find the onset rather than assume a fixed position.

## Architecture

A compact 1-D multi-scale encoder-decoder (1.44 M parameters):

```
input [1, 400]
  multi-scale stem (k = 5, 11, 21)        -> 48 x 400
  residual encoder 400 -> 200 -> 100 -> 50 -> 64/96/128
  dilated bottleneck d = 1, 2, 4, 8       -> 192 x 50
  decoder with skip connections 50 -> 400
  Conv1d(1) + tanh                        -> signed field [1, 400]
```

`SOPNet-Cls` shares exactly the same encoder, applies global average pooling
and a linear classifier, and serves as the capacity-matched ablation.

## Losses

```
L = L_field + lambda_pol * L_pol + lambda_inv * L_inv
```

- **Weighted signed field loss** (`beta = 8`):

  ```
  L_field = mean_t (1 + beta |g(t)|) * SmoothL1(f(t), g(t))
  ```

- **Polarity consistency** (no extra head): with `w = softmax(kappa |f|)` and
  `q = sum_t w_t f_t`,

  ```
  L_pol = log(1 + exp(-gamma * s * q))     (known samples only)
  ```

- **Inversion consistency** (applied to ~25 % of batches):

  ```
  L_inv = mean |f(-x) + f(x)|
  ```

Default weights: `lambda_pol = 0.5`, `lambda_inv = 0.1`.

## Training protocol

AdamW (`lr = 1e-3`, `wd = 1e-4`), cosine decay with 5 % warmup, gradient
clipping 1.0, AMP, batch 1024, early stopping on validation U/D accuracy
(patience 10). Training windows are randomly cropped so the P onset falls
uniformly in samples 120-280, preventing the network from memorising a fixed
onset position. Augmentations: SNR-based Gaussian noise (0-30 dB, p = 0.35),
polarity inversion (U <-> D, X unchanged, p = 0.5) and mild amplitude scaling.