# 数据协议

原始数据以只读方式挂载于 `/mnt/d/AI_Seismic_Data`。绝不修改原文件，也不整体复制。

## 统一数据集

| 数据源 | 文件 | 原生采样率 | 分量顺序 | Z 索引 | p_pick |
|---|---|---|---|---|---|
| SCSN | `scsn/scsn_p_2000_2017_6sec_0.5r_fm_combined.hdf5` | 100 Hz | flat `X (N,600)` | - | 300（固定） |
| TXED | `txed/Texd.hdf5` + `txed/Texd_filtered.csv` | 100 Hz | ZNE | 0 | `trace_p_arrival_sample` |
| INSTANCE | `Instance/INSTANCE.hdf5` + `Instance/INSTANCE_filtered.csv` | 100 Hz | ZNE | 0 | `trace_P_arrival_sample` |
| PNW | `pnw/pnw.hdf5` + `pnw/pnw.csv` | 100 Hz | ENZ | 2 | `trace_P_arrival_sample` |
| DiTing 1.0 | `谛听1.0-50HZ/Diting50hz/DiTing330km_part_{0..27}.{csv,hdf5}` | 50 Hz | Z（索引 0） | 0 | `p_pick` |

扫描后的计数（与随机种子无关，仅元数据）：**7,781,561 条样本**。

| 数据源 | U | D | X | 合计 |
|---|---:|---:|---:|---:|
| SCSN | 1,108,531 | 1,417,416 | 2,321,301 | 4,847,248 |
| TXED | 742 | 640 | 9,310 | 10,692 |
| INSTANCE | 626 | 369 | 3,969 | 4,964 |
| PNW | 58,775 | 49,113 | 76,021 | 183,909 |
| DiTing | 345,875 | 295,150 | 2,093,723 | 2,734,748 |
| **合计** | **1,514,549** | **1,762,688** | **4,504,324** | **7,781,561** |

## 规范标签

`source` 仅用于解析与审计，模型永远看不到它。规范极性是物理意义上的：
`Up = +1`、`Down = -1`、`Unknown = 0`。

| 数据源 | 原始值 | 规范标签 | 依据 |
|---|---|---|---|
| SCSN | `0` | Down | P 后带符号均值 −3.7e-4（n=66 k） |
| SCSN | `1` | Up | P 后带符号均值 +2.6e-4（n=67 k） |
| SCSN | `2` | Unknown | P 后带符号均值 ≈ 0 |
| TXED | `U/D/unknown` | +1/−1/0 | 文件元数据 |
| INSTANCE | `positive/negative/undecidable` | +1/−1/0 | 文件元数据 |
| PNW | `positive/negative/undecidable` | +1/−1/0 | 文件元数据 |
| DiTing | `U` / `C` | Up | 初动为正的比例 0.64 / 0.72 |
| DiTing | `D` / `R` | Down | 初动为正的比例 0.39 / 0.42 |
| DiTing | 其他 / 缺失 | Unknown | 无极性编码 |

通过检查真实数据解决了两个陷阱：

1. **SCSN 采用 `0 = Down, 1 = Up`**，而其他数据集采用 `0 = Up, 1 = Down`。
   因此禁止跨源直接拼接原始整数标签；`sopnet/data/canonical.py` 强制使用
   逐源的显式映射。
2. **DiTing 的 `R` 和 `C` 是真实的极性编码**，不是噪声。已发布的 DiTing 极性
   总数为 641,025，恰好等于 `U + R + D + C`；波形初动符号显示 `C` 的行为与
   `U` 相同（压缩），`R` 与 `D` 相同（膨胀）。正确映射后找回了 593,554 个
   原本会丢失的标签。

写入读取器的其他结论：

- INSTANCE 必须用 `trace_name` 索引；`trace_name_original` 指向其他记录
  （已用垂直向分量上的到时对齐验证）。
- TXED/INSTANCE 是 ZNE（不是 SeisPolarity 默认的 ENZ）；PNW 是 ENZ。
  分量索引来自各文件的 `data_format/component_order`。
- SCSN 的 P 后均值很小但统计上明确；逐条初动检查确认了该映射。

## 预处理

```
Z 分量 -> 重采样到 100 Hz -> 去均值 -> 线性去趋势
-> 1-45 Hz 零相位 Butterworth（SOS）-> 裁剪
-> 最大绝对值归一化（保号）
```

缓存窗口：600 个样本（6 秒），P 位于第 300 个样本。训练窗口：400 个样本
（4 秒），随机裁剪使 P 落在第 120-280 个样本之间。对 50 Hz 数据源只在到时
附近重采样一小段，因此 DiTing 的 180 秒长记录从不整段处理。

## 缓存结构

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

每个分片容纳 5 万条样本（约 120 MB/分片，合计约 19 GB）。`manifest_index`
保证 manifest 与缓存的位置对齐；回归测试会逐条比对缓存窗口与原始数据。

## 数据划分

固定种子 `20261004`，比例 80/10/10。样本按 `event_key` 分组
（SCSN `evids`、PNW `event_id`、INSTANCE `source_id`、TXED 事件名、DiTing
`part:ev_id`）。波形重复项（对标准化后的 600 样本窗口计算 64 位 blake2b 哈希
检测）在事件之间做并查集合并，保证相同波形绝不跨划分。`check_split_leakage`
同时校验这两条约束；当前全量 manifest 审计报告零事件泄漏、零重复泄漏。

## 只读保证

任何脚本都不写入 `/mnt/d/AI_Seismic_Data`；所有衍生文件都放在 `outputs/` 下。
HDF5 文件一律以只读方式打开（`h5py.File(..., "r")`）。
