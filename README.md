# Qwen3.8-27B 单卡推理优化

赛道 A：AI Infra。作者：**王嘉乐、毛俊琪、吴松灿**。

在单张 RTX 4090（49140 MiB）上，针对 8192-token 输入、512-token 输出的并发推理，组合 W4A4、MTP2、FP8 KV、CUDA Graph 与 Prefill 参数优化，并接入 SM89 attention 调优和 SiLU/INT4 量化融合。

答辩展示结果：**C14 输出 371.708 token/s，TTFT P95 9.899 s，TPOT P95 38.348 ms/token**。

- [Benchmark 报告](docs/BENCHMARK.md)
- [技术与优化说明](docs/TECHNICAL.md)
- [展示结果与原始记录](evidence/README.md)
- [并发演示](docs/DEMO.md)
- [来源与署名](NOTICE.md)

## 1. 环境与固定版本

| 项目 | 配置 |
|---|---|
| GPU | 单张 RTX 4090，49140 MiB，SM89，450 W 功率上限 |
| 系统 | Ubuntu 22.04.5，NVIDIA driver 610.57.04 |
| CUDA / Python | CUDA toolkit 13.0 / Python 3.12.14 |
| 推理运行时 | vLLM 0.31.0 + HyperQwen 上游 patch stack |
| 核心依赖 | PyTorch 2.13.0（cu130）、Triton 3.7.1、Humming 0.1.16 |
| 量化 | INT4 权重、动态对称 group-128 INT4 激活、FP8 KV cache |
| 其他状态 | BF16 主计算 dtype，FP16 Mamba/SSM cache |

普通 24 GiB RTX 4090 不是本项目的同规格复现环境。目标机需预装 CUDA 13.0 toolkit、uv、git、curl、tar、patch、tmux、make 和 g++，并能访问 Python 包源及 Hugging Face。

| 来源 | 固定 revision |
|---|---|
| [syv-ai/HyperQwen](https://github.com/syv-ai/HyperQwen) | `95ee3b64b0df7014ee2626529524d6b99c3dcdd1` |
| [dbirks/Qwen3.8-27B-W4A16-AutoRound](https://huggingface.co/dbirks/Qwen3.8-27B-W4A16-AutoRound) | `1f05c441c4e64ae0549de44fa9ea5a6d43610314` |
| [syvai/qwen3.8-27b-3090-fast-variant](https://huggingface.co/syvai/qwen3.8-27b-3090-fast-variant) | `124c14e7e8c7d2f5402933b9af368e772a9fcf0c` |

基础权重为 27B 级 INT4 模型；准备脚本处理 lm_head、embedding 和 MTP，叠加固定 fast variant，再配置 group-128 INT4 激活。Embedding 保留上游 INT8 权重配置，不对其应用 INT4 激活。

## 2. 安装与模型准备

以下命令在复现 GPU 机器执行。安装目录默认是本仓库下的 `.runtime/`；也可以先设置绝对路径 `WORK_DIR`。建议预留至少 80 GiB 磁盘空间，模型准备过程中保留下载副本和准备副本。

先为复现账号配置仓库读取权限，以下使用已授权的 SSH 克隆。文件交付时也可使用 `git clone --branch main ucloud-competition.bundle ucloud-competition`，无需复制维护者私钥。

```bash
git clone git@github.com:Nehawjl/ucloud-competition.git
cd ucloud-competition

export CUDA_HOME=/usr/local/cuda-13.0
bash scripts/setup.sh
bash scripts/prepare_model.sh
bash scripts/setup_eval.sh
```

`setup.sh` 校验上游源码 SHA-256，安装固定版本并依次应用上游 patch stack 和本仓库补丁。`prepare_model.sh` 固定两个模型 revision，不覆盖已有模型目录，也不重新校准 INT4 权重。首次启动包含 kernel 编译和 CUDA Graph capture，需要等待。

## 3. 性能复现

每条命令启动独立 detached tmux 会话和全新服务，测试结束后停止该服务并返回实际退出码。请勿并行执行 GPU 测试。

```bash
# 先确认最终启动参数，不启动 GPU 服务
CONCURRENCY=14 bash scripts/serve.sh --print

# 官方四档，逐档测试
for c in 1 2 4 8; do
  CONCURRENCY="$c" bash scripts/run.sh perf || break
done

# 扩展并发
CONCURRENCY=14 bash scripts/run.sh perf
```

性能设置：闭环请求、每档 8 预热 + 100 正式请求、固定 seed 42、8192/512 token、temperature 0、ignore_eos。输入由比赛工具按 tiktoken/o200k_base 重建，脚本核对固定数据指纹，报告记录服务端实际长度。TTFT P95 ≤10 s 且 TPOT P95 ≤40 ms/token 时判为双 SLO 通过。

默认服务端口为 `18099`。需要更换时使用 `PORT=18098 CONCURRENCY=14 bash scripts/run.sh perf`。服务仅绑定 `127.0.0.1`，不开放公网访问。

## 4. IFBench100

```bash
CONCURRENCY=14 bash scripts/run.sh accuracy
```

使用比赛提供的 EvalScope 1.13.0 评测入口：IFBench 原生前 100 题、请求并发 2、`max_tokens=32768`，其余使用工具默认生成配置。精度测试不继承性能测试的 temperature 或 ignore_eos。

## 5. 输出与异常处理

新运行的输出保存在 `.runtime/runs/<RUN_ID>/`，包括启动参数、服务日志、环境/硬件信息、指标、评测报告，以及 `started_at.txt`、`ended_at.txt`、`status.txt`、`exit_code.txt`。这些运行产物默认被 Git 忽略。

- 同名 RUN_ID 不复用；端口已占用时拒绝启动，避免误测已有服务。
- 服务启动超时或评测失败会保留日志，并写入实际退出码后通知等待方。
- 终端断开后，tmux 内任务继续运行；重新连接先检查 `exit_code.txt`，若尚未生成，再用 `tmux wait-for <RUN_ID>-done` 等待。
- 若需要中断，向对应会话中的任务发送 Ctrl+C；不要终止其他实验会话。

## 6. 仓库内容

```text
README.md                 复现入口与模型获取方式
docs/BENCHMARK.md          答辩对应的性能与精度结果
docs/TECHNICAL.md          选型、优化过程与取舍
evidence/                 PPT 结果索引、原始明细与测试配置
config/versions.env        固定源码/模型版本及校验值
requirements.txt          推理环境核心依赖
patches/                  复现所需的实现补丁
scripts/                  安装、模型准备、服务与评测入口
third_party/official-eval/ 比赛方评测工具
tests/                    不依赖 GPU 的配置测试
demo/                     C1–C15 可选的并发演示与自由对话
```

本仓库提供展示结果对应的精简记录，不包含完整实验历史、模型权重、缓存或虚拟环境。完整上游源码由安装脚本按固定版本获取；本项目实现修改均在 `patches/` 中。

本地静态检查：`python3 -m unittest discover -s tests -v`。

## 7. 并发演示

页面运行在 controller，连接实验机推理服务，支持 C1–C15、流式任务卡片、请求计时、服务端监控和记录下载。启动方法见[并发演示说明](docs/DEMO.md)。
