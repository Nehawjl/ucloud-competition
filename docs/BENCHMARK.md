# Benchmark 报告

## 1. 测试条件

单张 RTX 4090（49140 MiB），Ubuntu 22.04.5，驱动 610.57.04，CUDA 13.0，Python 3.12.14。最终框架为 vLLM 0.31.0 + 固定版 HyperQwen，使用 Humming 0.1.16。

性能测试使用 AIPerf 0.13.0：每档 8 个预热请求、100 个正式请求，闭环并发；输入 8192 token、输出 512 token，seed 42，temperature 0，ignore_eos。预热不进入统计；P95 使用逐请求数据计算，网络时间计入延迟。异常与中断保留独立状态，不以部分请求代替完整档位。

- TTFT：首 token 延迟。
- TPOT：每个请求 `(E2E−TTFT)/(输出 token 数−1)`，再对请求取 P95。
- 吞吐：正式测量区间内输出 token/s。
- 双 SLO：TTFT P95 ≤10 s，且 TPOT P95 ≤40 ms/token。

以下为与答辩 PPT 一致的展示结果。

各表对应的结果文件、逐请求/逐题明细、测试命令与服务配置见[展示结果索引](../evidence/README.md)。

## 2. 框架初筛

| 框架 | 版本 | 模型 / 格式 | C1 tok/s | C2 tok/s | C4 tok/s | C8 tok/s |
|---|---|---|---:|---:|---:|---:|
| vLLM (Official) | 0.31.0 | Qwen3.8-27B-FP8 / safetensors | 26.19 | 42.98 | 65.94 | 66.00 |
| q27 | v0.14.5 | Qwen3.8-27B-MTP-q27 / INT4、INT8 混合，.q27 | 116.07 | 119.18 | 116.95 | 120.44 |
| NInfer | 0.6.1-rtx3090 | Qwen3.8-27B-NInfer / Q4、Q5 为主，.ninfer | 71.85 | 84.08 | 88.90 | 100.59 |
| HyperQwen | - | W4A16-AutoRound fast variant / safetensors | 97.11 | 120.93 | 138.93 | 143.56 |

延迟 P95（TTFT 秒 / TPOT 毫秒）：

| 框架 | C1 | C2 | C4 | C8 |
|---|---|---|---|---|
| q27 | 3.11 / 2.68 | 6.19 / 10.36 | 12.05 / 27.57 | 24.73 / 50.97 |
| NInfer | 3.83 / 8.03 | 5.00 / 18.19 | 5.77 / 42.11 | 14.59 / 84.72 |
| HyperQwen | 2.88 / 5.71 | 3.33 / 16.54 | 5.96 / 29.33 | 10.001 / 62.48 |

q27、NInfer 的双 SLO 通过档位为 C1/C2；HyperQwen 为 C1/C2/C4。选择 HyperQwen 继续优化。这是各框架实际部署配置的比较，不将整体差异归因于单个 kernel。

## 3. 最新框架

量化配置：**权重 INT4、激活 INT4（动态对称 group-128）、KV cache FP8**。最大上下文 65536 token，MTP2 greedy，Humming 线性层，Triton SM89 attention；`max-num-batched-tokens=1536`、`max-num-scheduled-tokens=1408`、`stream-interval=2`，关闭 prefix caching。

| C | 输出 token/s | TTFT P95 s | TPOT P95 ms | E2E P95 s | 双 SLO |
|---:|---:|---:|---:|---:|:---:|
| 1 | 108.592 | 1.039 | 7.783 | 5.032 | 通过 |
| 2 | 181.605 | 1.255 | 9.850 | 6.180 | 通过 |
| 4 | 263.373 | 2.582 | 14.613 | 9.123 | 通过 |
| 8 | 334.720 | 3.328 | 23.580 | 15.443 | 通过 |
| 9 | 342.987 | 4.493 | 25.950 | 16.750 | 通过 |
| 10 | 355.325 | 5.425 | 25.972 | 18.079 | 通过 |
| 11 | 362.120 | 6.488 | 31.301 | 20.483 | 通过 |
| 12 | 370.132 | 7.353 | 35.588 | 22.530 | 通过 |
| 13 | 376.798 | 8.447 | 35.178 | 24.778 | 通过 |
| 14 | 371.708 | 9.899 | 38.348 | 27.488 | 通过 |
| 15 | 387.138 | 10.562 | 39.605 | 27.506 | 未通过 |

## 4. IFBench100

EvalScope 1.13.0，IFBench 前 100 题，`eval_batch_size=2`，`max_tokens=32768`。保留四项原生指标：

| 配置 | Prompt strict | Instruction strict | Prompt loose | Instruction loose |
|---|---:|---:|---:|---:|
| FP8 + BF16 A + FP8 KV | 80.00% | 80.18% | 87.00% | 87.39% |
| W4A8 + BF16 KV | 80.00% | 81.98% | 88.00% | 89.19% |
| W4A4 + BF16 KV | 77.00% | 77.48% | 87.00% | 87.39% |
| W4A4 + FP8 KV | | | | |
| W4A4 + MTP2 + FP8 KV + 融合 | 73.00% | 74.80% | 84.00% | 85.60% |

## 5. 候选取舍

| 模型 | 推理框架 | 并发 | 输出 token/s | 双 SLO | IFBench100 Prompt strict |
|---|---|---:|---:|:---:|---:|
| FP8 baseline | vLLM (Official) | 2 | 42.985 | 通过 | 80.00% |
| W4A16 + BF16 KV | HyperQwen（原生） | 4 | 138.929 | 通过 | 80.00% |
| W4A8 + BF16 KV | HyperQwen（原生） | 10 | 246.820 | 通过 | 80.00% |
| W4A4 + BF16 KV | HyperQwen（优化版） | 12 | 334.514 | 通过 | 77.00% |
| W4A4 + FP8 KV + 融合 | HyperQwen（优化版） | 14 | 371.708 | 通过 | 73.00% |

速度优先：优化版 HyperQwen，W4A4 + MTP2 + FP8 KV + 融合。

精度优先：原生 HyperQwen，W4A8 + BF16 KV。
