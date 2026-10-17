# 技术与优化说明

## 1. 问题与路线

目标是在单张 RTX 4090 上提高 8K 长输入的并发推理能力，同时满足 TTFT P95 ≤10 s、TPOT P95 ≤40 ms/token。

先复现官方 FP8 + vLLM baseline，再比较 q27、NInfer、HyperQwen。HyperQwen 原生 W4A16 + MTP4 在 C4 同时满足两项 SLO，且 C2/C4/C8 吞吐最高，因此以它作为优化起点。

主要瓶颈是长 Prefill 对 Decode 的干扰，以及并发升高后的矩阵乘、attention 和排队开销。优化沿用现成模型权重和框架能力，逐步调整激活量化、MTP 深度、CUDA Graph 覆盖、Prefill 预算与 KV/attention 路径。

## 2. 并发优化过程

下表在同一并发下比较上个配置与优化后配置。结果格式为“输出 token/s / TTFT P95 秒 / TPOT P95 毫秒”。

| 阶段 | 上个配置的结果 | 改动 | 优化后的结果 |
|---|---|---|---|
| C4 至 C8 | C8：143.563 / 10.001 / 62.482 | 激活 A16 改为动态 INT8，MTP4 改为 MTP3 | C8：236.996 / 6.368 / 31.098 |
| C8 至 C10 | C10：237.146 / 11.130 / 31.813 | CG 32 改为 40，Prefill chunk 2048 改为 1024 | C10：245.052 / 8.815 / 37.240 |
| C10 至 C12 | C12：247.243 / 12.165 / 45.115 | Marlin W4A8 改为 Humming W4A4，接入编译兼容补丁；MTP3 改为 2，Prefill 调到 1280 | C12：316.725 / 8.707 / 38.496 |
| C12 至 C14 | C14：339.153 / 9.966 / 42.131 | FP8 KV + Triton attention，SM89 launch 参数调优，接入 SiLU/INT4 量化融合 | C14：371.708 / 9.899 / 38.348 |

### C4 至 C8：降低 Prefill 和投机验证开销

INT4 权重不变，使用 HyperQwen 的动态 INT8 activation 路径加速线性层；同时减少 MTP 草稿数，降低每步验证成本。激活精度与投机深度一起选择，不单独追求更多草稿 token。

### C8 至 C10：覆盖真实验证宽度，缩短 Prefill 占用

该阶段不修改模型、量化方式或 CUDA kernel，只调整两个性能参数：

1. MTP3 每请求最多验证 4 个 token，C10 需要覆盖 `10 × (3+1) = 40`，因此 CG 从 32 提高到 40。
2. 闭环测试每完成一个请求就加入一个新的 8192-token prompt。Prefill chunk 从 2048 降到 1024，缩短单次占用 GPU 的时间，减少对 Decode 的干扰。

服务并发槽同步设为 10，作为该档测试的容量配置。

### C10 至 C12：接入 W4A4

保留原 INT4 group-128 权重，为线性层加入动态对称 group-128 INT4 激活，选择 Humming 后端。修复 Humming 设备信息对象在 torch.compile tracing 中的兼容问题；Embedding 保留上游配置。

MTP 改为 2，平衡草稿收益和验证成本；继续扫描 Prefill 预算。此阶段不重新训练模型或重新校准权重。

### C12 至 C14：优化缓存、attention 和激活路径

切换 FP8 KV 与 Triton attention，启用已有的 FP8 split-KV verify 路径。对 SM89、head size 256 的 Prefill 分支使用 `BLOCK_M=32`、8 warps、2 stages、Prefill tile 64；其他 shape 保持原分支。

将 Humming 已有的 `binary_split` 激活能力接入 Qwen MLP：让 down projection 的输入处理同时完成 `SiLU(gate) × up`、动态 INT4 量化和 scale 计算，省去独立激活及中间张量读写。

最终 Prefill 参数为 batched tokens 1536、scheduled tokens 1408，stream interval 2。该阶段是组合优化，不把全部收益归因于 FP8 KV 单项。

## 3. 原生与最终配置

| 配置 | 原生 | 最终 |
|---|---|---|
| 线性层 | Marlin W4A16 | Humming W4A4 |
| MTP 草稿数 | 4 | 2，greedy |
| KV / Attention | BF16 / FlashAttention | FP8 / Triton SM89 |
| CUDA Graph 上限 | 32 | C14 为 42，即 `C × (2+1)` |
| Batched / Scheduled tokens | 2048 / 随批量 | 1536 / 1408 |
| Prefix caching | 关闭 | 关闭 |

## 4. 实现边界与工程化

HyperQwen 的 MTP、Marlin W4A8、FP8 speculative attention，以及 Humming 的 W4A4 kernel、融合激活算子均来自上游。我们的工作是选型与性能分析、参数组合、W4A4 编译兼容适配、SM89 attention 分支调优，以及融合参数透传和模型前向接入。

复现入口固定模型与代码版本，校验源码和合成数据指纹。每次测试创建独立 RUN_ID、tmux 会话和输出目录，检查服务就绪、记录实际退出码，失败时保留现场；服务只绑定本机端口。展示结果对应的环境、命令和原始记录见 [evidence](../evidence/README.md)，完整运行日志在复现时生成。

## 5. 成本与取舍

全部优化在单卡内完成，不增加 GPU 数量。W4A4 提高吞吐，但激活量化需要结合 IFBench 判断质量损失；FP8 KV 降低缓存开销，也需要配套验证精度。并发升高后，吞吐增加不一定继续改善 TTFT。

速度优先采用优化版 HyperQwen 与 W4A4 + MTP2 + FP8 KV + 融合，IFBench100 Prompt strict 为 73.00%；精度优先采用原生 HyperQwen 与 W4A8 + BF16 KV，Prompt strict 为 80.00%。后续继续分析长输入排队与输出质量失败样例。
