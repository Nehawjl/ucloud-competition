# 展示结果索引

各项数字与答辩主 PPT 的 8 页结构对应。性能目录保留 AIPerf 逐请求记录、原生汇总、测试命令、阶段时间及服务配置；精度目录保存各模型的 IFBench100 结果文件，EvalScope 原生报告和逐题记录一并归档。

## 框架初筛（第 2、3 页）

每个目录包含 C1、C2、C4、C8 四档，吞吐和延迟来自同一组请求。

| 框架 | 文件 |
|---|---|
| vLLM (Official) | [screening/official](performance/screening/official/) |
| q27 | [screening/q27](performance/screening/q27/) |
| NInfer | [screening/ninfer](performance/screening/ninfer/) |
| HyperQwen | [screening/hyperqwen-native](performance/screening/hyperqwen-native/) |

## 优化时间线（第 4 页）

| 节点 | 优化前，同并发 | 优化后 |
|---|---|---|
| C4 起点 | — | [原生 C4](performance/screening/hyperqwen-native/c4/) |
| C8 | [原生 C8](performance/screening/hyperqwen-native/c8/) | [W4A8 + MTP3](performance/timeline/c8-after/) |
| C10 | [CG32 / Prefill 2048](performance/timeline/c10-before/) | [CG40 / Prefill 1024](performance/timeline/c10-after/) |
| C12 | [W4A8 + MTP3](performance/timeline/c12-before/) | [W4A4 + MTP2 / Prefill 1280](performance/timeline/c12-after/) |
| C14 | [BF16 KV](performance/timeline/c14-before/) | [FP8 KV + SM89 调优 + 融合](performance/final/c14/) |

## 最新框架（第 1、5 页）

W4A4 + MTP2 + FP8 KV + 融合，各档 8 预热、100 正式请求，8192/512 token。

[C1](performance/final/c1/) · [C2](performance/final/c2/) · [C4](performance/final/c4/) · [C8](performance/final/c8/) · [C9](performance/final/c9/) · [C10](performance/final/c10/) · [C11](performance/final/c11/) · [C12](performance/final/c12/) · [C13](performance/final/c13/) · [C14](performance/final/c14/) · [C15](performance/final/c15/)

逐请求延迟和 token 数见 `profile_export.jsonl`；`benchmark_phase=warmup` 不进入正式统计。吞吐、测量时长见 `profile_export_aiperf.json`。排队、Prefill、Decode、MTP 与 KV 指标见 `server_metrics_export.json`，采集记录见 `server_metrics.json` 和 `context/server_metrics*.prom`。

## IFBench100（第 6、7 页）

| 模型配置 | 结果 |
|---|---|
| FP8 + BF16 A + FP8 KV | [official](accuracy/official/summary.json) |
| W4A16 + BF16 KV | [w4a16-bf16kv](accuracy/w4a16-bf16kv/summary.json) |
| W4A8 + BF16 KV | [w4a8-bf16kv](accuracy/w4a8-bf16kv/summary.json) |
| W4A4 + BF16 KV | [w4a4-bf16kv](accuracy/w4a4-bf16kv/summary.json) |
| W4A4 + MTP2 + FP8 KV + 融合 | [w4a4-mtp2-fp8kv-fused](accuracy/w4a4-mtp2-fp8kv-fused/summary.json) |

EvalScope 的 `ifbench.json` 保留原始精度；`reviews.jsonl` 包含题目、模型回答和逐题评分。PPT 展示值作了小数位舍入。

## 模型取舍（第 7 页）

| 模型配置 | 性能结果 | 精度结果 |
|---|---|---|
| FP8 baseline | [Official C2](performance/screening/official/c2/) | [IFBench100](accuracy/official/summary.json) |
| W4A16 + BF16 KV | [原生 C4](performance/screening/hyperqwen-native/c4/) | [IFBench100](accuracy/w4a16-bf16kv/summary.json) |
| W4A8 + BF16 KV | [C10](performance/models/w4a8-c10/) | [IFBench100](accuracy/w4a8-bf16kv/summary.json) |
| W4A4 + BF16 KV | [C12](performance/models/w4a4-bf16kv-c12/) | [IFBench100](accuracy/w4a4-bf16kv/summary.json) |
| W4A4 + FP8 KV + 融合 | [C14](performance/final/c14/) | [IFBench100](accuracy/w4a4-mtp2-fp8kv-fused/summary.json) |

## 文件校验

[manifest.json](manifest.json) 记录各项对应的实验编号、文件路径和 SHA-256。原生输出不改写；其中的绝对路径是采集时的实验机路径。新机器使用仓库复现脚本生成自己的运行目录。

在仓库根目录运行，无需 GPU：

```bash
python3 scripts/verify_evidence.py
```
