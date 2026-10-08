# SOPNet 论文实验计划（Applied Geophysics）

目标：以现有 v1 结果（test U/D acc 0.9782）为基础，补齐关键实验，形成可投稿的
方法论文。四个验收关口 G1-G4。

## 阶段与验收

| 关口 | 内容 | 验收标准 |
|---|---|---|
| G1 主结果可信 | 基线协议审查与修正、必要重训、统一主结果与逐源结果、逐样本预测留存 | 每个数字可追溯到 checkpoint 与测试样本，六模型协议一致 |
| G2 方法贡献成立 | 四组无 jitter 消融（A Cls / B Field / C +极性损失 / D 完整）、最终模型多种子 | 消融表完整，创新声明有证据 |
| G3 鲁棒性完整 | 六模型 SNR + P 偏移（固定子集）、事件级配对统计、5 图 4 表 | 图表与文字一致，扰动实验样本固定 |
| G4 论文可投稿 | 英文初稿、引用核查、格式与投稿材料 | 结果可核验，符合期刊要求 |

## 实验清单

1. **G1**：修正 DiTingMotion 第二通道（一阶差分取符号、开头补零）；RPNet 类别
   顺序改为官方（索引 1 = Up）；EQPolarity 输入长度改为官方 600；strict U/D
   主指标 + unknown 拒判补充指标；重训 ross / rpnet / diting_motion /
   eqpolarity；生成 `main_results.csv`、`per_source_results.csv`、
   `baseline_protocol.csv`、逐样本预测。
2. **G2**：四组消融全部无 jitter、统一主干与优化器：
   - A `SOPNet-Cls`（U/D 二分类，known-only，匹配 v1 损失口径）
   - B Signed Field（known-only 场损失）
   - C B + 极性一致性损失
   - D C + 反转一致性 = 最终 SOPNet（复用 `sopnet_nojitter_36`）
   报告 Accuracy / Macro-F1 / MCC / 参数量；关键对照多种子（资源允许）。
3. **G3**：六模型鲁棒性，固定跨源已知 U/D 子集：
   - SNR：clean、20、10、5、0、−5 dB（相同噪声、可复现）
   - P 偏移：−0.20、−0.10、−0.05、0、+0.05、+0.10、+0.20 s
   - 事件级聚类配对 bootstrap（按 event_key 重采样）
   - 输出 Figure 3/4（六模型曲线）、表格
4. **G4**：论文结构 4,500-5,500 词；5 图 4 表；Cover Letter；引用核查。

## 写作边界

- 不宣称首次提出逐时刻相位表示；不把固定中心窗口的 P MAE 当作独立拾取能力。
- 任务定义为"给定参考 P 到时的初动极性判定"，与基线任务一致。
- P 偏移鲁棒性弱写入讨论（精度 vs 平移容忍度权衡）。

## 产物目录

```
docs/paper/{EXPERIMENT_PLAN,BASELINE_AUDIT,EXPERIMENT_LOG,RESULT_VERIFICATION}.md
paper/manuscript/{manuscript.tex,references.bib}
outputs/paper/{tables,figures,predictions,run_manifests}/
```
