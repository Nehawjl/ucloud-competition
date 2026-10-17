# 并发演示

Gradio 页面运行在 controller，只向实验机的推理服务发请求，不加载 GPU 模型。前端使用独立环境，不修改推理依赖。

## 启动

先按 [README](../README.md) 完成模型与环境准备。确认 GPU 没有其他模型服务，再在实验机的部署目录启动：

```bash
export CUDA_HOME=/usr/local/cuda-13.0
# 若安装时指定了 WORK_DIR，这里设置为同一个绝对路径
CONCURRENCY=14 PORT=18109 bash scripts/serve_demo.sh
```

服务在独立 tmux 会话运行，只监听 `127.0.0.1:18109`。命令会输出 RUN_ID、日志目录和停止命令；日志包含实际启动参数、环境版本、模型配置、GPU 遥测及起止状态。

在 controller 建立隧道，保持该终端开启：

```bash
ssh -N -L 127.0.0.1:18109:127.0.0.1:18109 ucloud-4090
```

在 controller 的本仓库另开终端：

```bash
bash scripts/demo.sh
```

浏览器访问 `http://127.0.0.1:7860`。首次启动自动安装 `demo/requirements.txt`，依赖 uv 和 Python 3.12。若页面和模型服务在同一台机器，无需 SSH 隧道。

更换接口或页面端口：

```bash
DEMO_BASE_URL=http://127.0.0.1:18109/v1 bash scripts/demo.sh --port 7861
```

接口密钥通过 `DEMO_API_KEY` 环境变量传入，不填在 URL 或页面里。页面默认仅本机可访问，不创建公开分享链接；非本机绑定必须同时设置 `DEMO_USER`、`DEMO_PASSWORD`。

## 展示方式

1. 选择 C1–C15，默认 C8；设置每路任务数和输出上限。
2. 点击“开始并发演示”。每个通道独立请求模型，完成后自动补入下一条任务；卡片显示真实流式输出。
3. 观察本轮吞吐、平均 TTFT/TPOT，以及服务端运行数、排队数、KV 占用和输出速率曲线。
4. 切到自由对话让评委提问，或切到 Benchmark 查看完整测试结果和原始记录。

C 是客户端请求并发，不会在页面内重启模型或修改服务端槽位。服务默认 14 槽；更高的请求并发可能排队。自由对话与并发演示共享队列，不会互相插入请求。

token 数取自服务返回的 usage，不用字符数或流式数据包数量代替。TTFT 从发送请求计到首个生成 token，包含思考输出；TPOT 为完成请求的 `(E2E−TTFT)/(输出 token 数−1)`。服务总输出速率来自 `/metrics` 的 token 计数差分。页面不把少量演示请求标成正式 Benchmark 的 P95。

每轮的报告和原始流式记录保存在 `.runtime/demo-records/<RUN_ID>/`，页面可下载报告。记录包括任务输入、模型输出和计时，默认不提交 Git。

本地功能检查（不请求 GPU）：

```bash
.runtime/demo-env/bin/python -m unittest discover -s tests -v
```
