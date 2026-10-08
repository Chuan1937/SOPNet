# SOPNet

**SOPNet（Signed-Onset Polarity Network，带符号起始场极性网络）** 通过预测一个
**带符号的起始场（signed onset field）** 来判定单分量垂直向地震记录的 P 波初动极性，
而不是输出离散类别标签。

给定 100 Hz 采样的 4 秒窗口 `x(t)`，网络预测

```
f(x) : R^400 -> [-1, 1]^400
```

以带符号高斯场为训练目标：

```
g(t) = s * exp(-(t - t_P)^2 / (2 * sigma^2)),   s = +1（上）, -1（下）, 0（未知）
```

P 波到时、极性与置信度全部从一个输出场读取：

```
t_hat = argmax_t |f(x)(t)|,   s_hat = sign(f(x)(t_hat)),   C = max_t |f(x)(t)|
```

## 统一训练数据

五个数据集（SCSN、TXED、INSTANCE、DiTing、PNW）在逻辑上合并为一份 manifest。
`source` 字段只用于标签映射与审计，绝不输入模型。规范标签是物理意义上的：
`Up = +1`、`Down = -1`、`Unknown = 0`。

| 数据源 | 原始标签 | 规范映射 | 原生采样率 | 分量顺序 |
|---|---|---|---|---|
| SCSN | 0/1/2 | 0=D, 1=U, 2=X | 100 Hz | flat `X` |
| TXED | U/D/unknown | U=+1, D=-1, 其他 0 | 100 Hz | ZNE |
| INSTANCE | positive/negative/undecidable | pos=+1, neg=-1, 其他 0 | 100 Hz | ZNE |
| DiTing | U/D/R/C/other | U,C=+1; D,R=-1; 其他 0 | 50 Hz | Z（索引 0） |
| PNW | positive/negative/undecidable | pos=+1, neg=-1, 其他 0 | 100 Hz | ENZ（Z 索引 2） |

DiTing 的 `R`/`C` 映射已通过实测验证，见 `docs/data_protocol.md`。

## 流程

```
原始数据（只读 /mnt/d/AI_Seismic_Data）
  -> manifest.parquet           （仅元数据，不复制波形）
  -> 分片 HDF5 缓存              （600 样本 @ 100 Hz，P 位于第 300 个样本）
  -> UnifiedPolarityDataset     （随机 400 样本裁剪，带符号场目标）
  -> SOPNet / SOPNet-Cls
  -> 基线、消融、鲁棒性、bootstrap
```

## 快速开始

```
pip install -e .
python scripts/inspect_data.py --data-root /mnt/d/AI_Seismic_Data
python scripts/build_manifest.py --data-root /mnt/d/AI_Seismic_Data
python scripts/build_cache.py --data-root /mnt/d/AI_Seismic_Data --workers 8
python scripts/train.py --config configs/experiments/sopnet_full.yaml \
    --limit-train 20000 --limit-val 5000 --epochs 3
```

## 目录结构

```
sopnet/{data,models,losses,training,evaluation,utils}
configs/{data,model,train,experiments}
scripts/    tests/    docs/
outputs/    （gitignored）
```

## 许可

BSD-3-Clause，见 `LICENSE`。
