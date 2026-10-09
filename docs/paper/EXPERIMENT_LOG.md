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
  1. diting_motion（官方差分通道，batch 1024）——完成 01:24
     （test acc 0.9634 / macro-F1 0.9631 / MCC 0.9263）
  2. ross（strict U/D 早停监控）——完成 02:53
     （test acc 0.9694 / macro-F1 0.9692 / MCC 0.9384）
  3. rpnet（官方类别顺序）——完成 10:50
     （test acc 0.9728 / macro-F1 0.9726 / MCC 0.9453）
  4. eqpolarity（官方 600 输入）——进行中（10:51 起；CCT 计算量大，
     首轮约 30+ 分钟，预计 10-30 h）
- [ ] 重训完成后生成 `main_results.csv`、`per_source_results.csv`、
  `baseline_protocol.csv` 与逐样本预测（`outputs/paper/predictions/`）——
  已由自动收尾链 `g1_finish.sh` 接管。

主表实时预览（strict U/D，n_known=327,231）：

| 模型 | acc | macro-F1 | MCC |
|---|---|---|---|
| SOPNet | 0.9782 | 0.9779 | 0.9561 |
| CFM | 0.9729 | 0.9727 | 0.9454 |
| RPNet | 0.9728 | 0.9726 | 0.9453 |
| Ross | 0.9694 | 0.9692 | 0.9384 |
| DiTingMotion | 0.9634 | 0.9631 | 0.9263 |
| EQPolarity | 待训练完成 | | |

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

### G4 论文骨架（22:30-23:40，与 G1 并行）

- [x] `paper/manuscript/manuscript.tex` + `references.bib`：英文初稿骨架
  （摘要/引言/相关工作/方法/数据协议/结果占位/讨论/结论），LaTeX+BibTeX
  编译通过、引用零 undefined；五基线文献出处已核实（RPNet/EQPolarity
  卷页标注投稿前复检）。
- [x] Table 1（数据集统计）用真实数字：7,781,561 波形 / 1,117,379 事件；
  测试 777,720 / 已知极性 327,231 / 111,588 事件。
- [x] Figure 1 架构图：`scripts/make_fig_architecture.py`（无 GPU 依赖）。
- [x] Figure 2 定性示例：`scripts/make_fig_examples.py`（CPU 推理，含
  低置信正确与高置信错误各一例）。
- [x] Figure 5 校准/选择性预测：`scripts/make_fig_calibration.py`
  （CFM 预测文件冒烟通过；待 SOPNet 预测 npz 生成后出正式图）。
- [x] `docs/paper/RESULT_VERIFICATION.md`：数字追溯文档（已验证/待生成清单、
  复现命令、核验规则）。

### G3 前置修复（23:40）

- 冒烟发现并修复数据管线 bug：`p_shift_samples` 对 `window_length=600` 的输入被
  `np.clip(start, 0, 600-W)` 裁掉（W=600 时恒等于 0），**导致基线模型的 P 偏移
  实验完全无效**。修复为：窗口起点允许越界，缓存外样本零填充（SOPNet 400 窗口
  行为不变）。新增回归测试 `test_p_shift_applies_to_full_cache_window`。
  修复后冒烟：CFM 在 +0.2 s 偏移下 0.975 → 0.40，与 SOPNet 量级一致。

## 2026-10-09

### G1 收尾（EQPolarity 决策）

- 15:18 停止 EQPolarity 训练（已完成 5 轮，val U/D 0.9656 趋于平台；
  用户决策，不标注早停，按正常结果入表）。
- 15:20 启动"G1 收尾 + G2 + G3"串行链（`/tmp/opencode/closeout_g2_g3.sh`，
  日志 `outputs/runs/closeout_g2_g3.log`，失败即中止）：
  1. EQPolarity `best.pt`（第 5 轮）strict 重评估 + 协议补全
     （epochs_run=5，与 ross=21、rpnet=45 同为常规字段）；
  2. SOPNet 测试集重评估并保存逐样本预测；
  3. 主表 / 逐源表 / 协议表 + 事件聚类配对 bootstrap；
  4. G2：A/B/C 消融训练（各 50 轮、batch 1024，约 14 h/个）+ 测试评估 + 消融表；
  5. G3：六模型鲁棒性（30k 固定子集，SNR + P 偏移）+ Fig5 校准图。

预计串行总时长约 45 h（G2 占 ~42 h）。
