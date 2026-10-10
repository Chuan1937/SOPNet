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

### G2-A 早停（用户决策）

- 16:37 停止 A 组训练（best epoch 7，val U/D 0.9752；epoch 8 起在 ±0.001
  内波动，进入平台）。主链按设计记录 `FAILED ablation` 并中止。
- 16:38 启动接力链 `/tmp/opencode/g2_continue.sh`（日志
  `outputs/runs/g2_continue.log`，失败即中止）：
  1. 评估 A `best.pt`（test 指标入消融表）；
  2. 训练 B（完整场模型）→ 评估 B；
  3. 训练 C（场 + 极性损失）→ 评估 C；
  4. 汇总表格 → G3 六模型鲁棒性 → Fig5 校准图。
- 参照：D 组 val 最好 0.9786（场模型对分类器 A 的差距 ~0.3 个点，
  即消融要展示的场表示增益）。

### G2-B 早停（用户决策）

- 18:43 停止 B 组训练（best epoch 8，val U/D 0.9756；第 9 轮起不再刷新）。
- 接力链 `/tmp/opencode/g2_continue_c.sh`（日志 `outputs/runs/g2_continue_c.log`）：
  评估 B → 训练 C → 评估 C → 汇总表格 → G3 鲁棒性 → Fig5。

### 指标口径修复（19:00）

- 发现 `evaluate_field` 在传入阈值时用"阈值化 macro-F1"覆盖了
  `macro_f1_ud`（B 组覆盖 90.8% 时被压成 0.9436；D 组因覆盖率 99.5%
  仅差 1e-4，此前 bootstrap 与主表的 1e-4 差异即源于此）。
- 修复：`macro_f1_ud` 保持**无阈值 strict 口径**；阈值化版本改名
  `macro_f1_ud_selected`（选择性操作点，仅作补充）。全部测试通过。
- 收尾阶段用修正后的代码重评估 B 与 SOPNet-D 并重建表格；C 的测试评估
  将自动使用修正口径。

### G2/G3 收尾完成（20:43）

- 消融最终（测试集 strict U/D，n_known=327,231；A/B/C 为平台/手动早停，
  D 为完整 50 轮）：
  - A 纯分类器 0.9746 / 0.9745 / 0.9489（e7 停）
  - B 场 0.9752 / 0.9751 / 0.9501（e8 停）
  - C 场+极性 0.9736 / 0.9734 / 0.9469（e6 停，欠训练，方案待用户定夺）
  - D 完整 0.9782 / 0.9780 / 0.9561
- 指标口径修复后 SOPNet macro-F1 = 0.97804（与 bootstrap 完全一致）。
- G3 鲁棒性（30k 固定子集）：SNR 下 SOPNet 全档第一（clean 0.9768 →
  −5 dB 0.7795）；P 偏移下六模型均在 ±0.05 s 崩至 0.59-0.71，±0.1 s 后
  接近随机——固定中心窗口对参考 P 精度是硬性依赖（讨论中如实呈现）。
- Fig3（SNR）、Fig4（P 偏移）、Fig5（校准/选择性）全部产出。
- 表格产物：`outputs/paper/tables/{main_results, per_source_results,
  baseline_protocol, ablation_results, bootstrap, snr_all, p_shift_all}.csv`。

### 参考文献核验（Crossref，21:0x）

- 全部条目经 Crossref API 校验，修正 5 处：Ross DOI（→10.1029/2017JB015251）、
  PolarCAP DOI（→10.1016/j.aiig.2022.08.001）、PNW 出处（EarthArXiv→Seismica
  2(1), doi:10.26443/seismica.v2i1.368）、DiTing 作者（Chen, Shi）、TXED 期号
  （95(3)）。补全 EQPolarity/RPNet/CFM/DiTingMotion 完整作者列表与卷页。
- 编译复验：BibTeX 零告警，正文零未定义引用。

### 待办补充

- P 偏移细网格（±10/20/30 ms）已加入 `run_robustness.py --p-shifts`，
  重跑中；完成后更新 Fig4 与正文容忍带数字。
- C 组续训（resume from last.pt，用户批准）：细网格完成后启动。

### C 组续训完成（10-10 08:57）

- C 从 e7 恢复、跑满 50 轮：best epoch 49（val 0.97844 / macro-F1 0.97831），
  测试 0.9781 / 0.9779 / 0.9558（ECE 0.0305）。
- 消融终版：A 0.9746 → B 0.9752 → C 0.9781 → D 0.9782；结论更新为
  **极性一致性损失贡献全部一致性增益（+0.29pp），反转损失中性（+0.01pp）**。
- 手稿表 3、5.3 节与讨论"Why a field"已按终版改写；早停脚注已移除。
