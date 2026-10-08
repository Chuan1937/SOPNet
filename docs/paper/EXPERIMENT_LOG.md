# 实验日志（G1-G4）

记录每个阶段的启动/完成时间、命令、产物与异常。

## 2026-10-08

### G1 基线协议审查与重训（进行中）

- [x] 审查 5 个基线协议，发现并修正 3 处偏差（见 `BASELINE_AUDIT.md`）：
  DiTingMotion 第二通道改为官方"一阶差分取符号（开头补零）"；RPNet 类别顺序改为
  官方（索引 1 = Up）；EQPolarity 输入改为官方 600 样本。
- [x] 新增 strict U/D 主指标（用模型自身 U/D 得分判定）与 unknown 拒判率补充指标。
- [x] CFM 无需重训：21:08 完成 strict 协议重评估（test F1 0.9704，MCC 0.9454，
  macro-F1 U/D 0.9727）。
- [x] 旧协议 checkpoint 归档到 `outputs/retired/`（rpnet、diting、ross、eqpolarity）。
- [ ] 重训队列（21:08 启动，单 GPU 串行）：
  1. diting_motion（官方差分通道，batch 1024）——进行中
  2. ross（strict U/D 早停监控）
  3. rpnet（官方类别顺序）
  4. eqpolarity（官方 600 输入，预计 17-30 h）
- [ ] 重训完成后生成 `main_results.csv`、`per_source_results.csv`、
  `baseline_protocol.csv` 与逐样本预测（`outputs/paper/predictions/`）——
  已由自动收尾链 `g1_finish.sh` 接管。

日志：`outputs/runs/g1_pipeline.log`。

### 下一步（与 G1 并行，纯代码）

- [x] G2：新增 `classify_ud` 任务（SOPNet-Cls 二分类、known-only 掩码 CE），
  准备 A/B/C 三个消融配置（D 复用 `sopnet_nojitter_36`）；CPU 冒烟验证三组配置
  端到端训练成功（1 epoch/2000 样本）。
- [x] G3：`run_robustness.py` 扩展为六模型统一评估（固定子集、相同噪声；
  偏移网格 −0.2~+0.2 s，SNR clean~−5 dB；输出合并表与六模型曲线）。
- [x] 统计：事件级聚类配对 bootstrap（按 event_key 整组重采样）。
- [x] `run_full_pipeline.sh` 改为严格失败退出模式（任何步骤失败即中止）。
- [x] 逐样本预测保存（SOPNet `--save-predictions`）+ `make_main_table.py`
  （main_results / per_source_results / baseline_protocol / 消融测试表）。

### 自动收尾链

- `g1_finish.sh`（22:20 挂起）：等待重训队列结束后自动执行
  1. 校验无失败步骤且 5 个基线 test_metrics + 预测齐备；
  2. SOPNet 测试集重评估并保存逐样本预测；
  3. 汇总论文表；
  4. 聚类配对 bootstrap。
  日志：`outputs/runs/g1_finish.log`。
