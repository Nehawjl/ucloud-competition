from __future__ import annotations

import argparse
import html
import json
import os
from pathlib import Path
import threading
import uuid

os.environ.setdefault("GRADIO_ANALYTICS_ENABLED", "False")
for proxy_key in ("NO_PROXY", "no_proxy"):
    os.environ[proxy_key] = ",".join(filter(None, [os.environ.get(proxy_key, ""), "127.0.0.1", "localhost"]))

import gradio as gr

from .client import Config, Round, message_history

ROOT = Path(__file__).resolve().parents[1]
ACTIVE: dict[str, Round] = {}
ACTIVE_LOCK = threading.Lock()


def esc(value) -> str:
    return html.escape(str(value))


def number(value, digits=1, suffix="") -> str:
    return "—" if value is None else f"{value:,.{digits}f}{suffix}"


def kpis(run: Round | None = None) -> str:
    if run:
        data = run.summary()
        first = [r.timing()["ttft_s"] for r in run.requests if r.first is not None]
        tpot = [r.timing()["tpot_ms"] for r in run.requests if r.timing()["tpot_ms"] is not None]
        values = [("本轮输出吞吐", number(data["round_token_s"], 1), "token/s"),
                  ("平均首 Token", number(sum(first) / len(first) if first else None, 3), "s"),
                  ("平均 TPOT", number(sum(tpot) / len(tpot) if tpot else None, 2), "ms/token"),
                  ("完成请求", str(data["completed"]), f"/ {data['request_count']}")]
        status = run.status
    else:
        values = [("本轮输出吞吐", "—", "token/s"), ("平均首 Token", "—", "s"),
                  ("平均 TPOT", "—", "ms/token"), ("完成请求", "0", "/ —")]
        status = "等待开始"
    items = "".join(f'<div class="kpi"><span>{title}</span><div>{value}<small>{unit}</small></div></div>' for title, value, unit in values)
    return f'<div class="round-state"><i></i>{esc(status)}</div><div class="kpi-grid">{items}</div>'


def board(run: Round | None = None, concurrency=8) -> str:
    slots = run.slots if run else [None] * concurrency
    cards = []
    for slot, request in enumerate(slots):
        if not request:
            cards.append(f'<article class="task idle"><header><b>通道 {slot+1:02d}</b><span>等待</span></header><p class="empty">请求将在这里流式输出</p><footer>TTFT —<span>输出 — token</span></footer></article>')
            continue
        timing = request.timing()
        cls = "done" if request.state == "完成" else "error" if request.state == "失败" else "live"
        content = request.text or ("思考中…" if request.reasoning and request.state != "完成" else "等待首 Token…")
        if request.state == "完成" and not request.text:
            content = "本次输出为思考内容，可在记录中查看。"
        if request.error:
            content = request.error
        cards.append(f'<article class="task {cls}"><header><b>通道 {slot+1:02d}</b><span>{esc(request.state)}</span></header>'
                     f'<div class="task-title">#{request.index:02d} · {esc(request.title)}</div><pre>{esc(content)}</pre>'
                     f'<footer>TTFT {number(timing["ttft_s"], 2, " s")}<span>输出 {number(request.output_tokens, 0)} token</span></footer>'
                     f'<footer>输入 {number(request.prompt_tokens, 0)} token<span>TPOT {number(timing["tpot_ms"], 1, " ms")}</span></footer></article>')
    return '<div class="task-grid">' + "".join(cards) + '</div>'


def monitor(run: Round | None = None) -> str:
    server = run.server if run else {}
    items = [("运行请求", server.get("running"), ""), ("排队请求", server.get("waiting"), ""),
             ("KV Cache 占用", server.get("kv_percent"), "%")]
    values = "".join(f'<div class="server-row"><span>{title}</span><b>{number(value, 1 if unit else 0, unit)}</b></div>' for title, value, unit in items)
    points = run.rates[-50:] if run else []
    if len(points) > 1:
        start, end = points[0][0], points[-1][0]
        peak = max(10, max(p[1] for p in points))
        coords = " ".join(f'{8+(x-start)/max(end-start,.001)*300:.1f},{115-y/peak*95:.1f}' for x, y in points)
        chart = f'<svg viewBox="0 0 316 130" role="img" aria-label="服务端实测输出速率"><path d="M8 115 H308" stroke="#dfe7e2"/><polygon points="8,115 {coords} 308,115" fill="#d7f8e4"/><polyline points="{coords}" fill="none" stroke="#159464" stroke-width="2.5"/></svg>'
        rate = number(points[-1][1], 1)
    else:
        chart = '<div class="chart-empty">运行后显示实测曲线</div>'
        rate = "—"
    error = f'<p class="monitor-note">{esc(run.server_error)}</p>' if run and run.server_error else ''
    elapsed = number(run.summary()["elapsed_s"], 1, " s") if run else "—"
    c = run.c if run else "—"
    return f'<div class="monitor-panel"><div class="eyebrow">SERVER TELEMETRY</div><h3>服务端状态</h3>{values}<div class="rate-label">服务总输出速率</div><div class="rate-value">{rate}<small> token/s</small></div>{chart}<div class="server-row"><span>发起并发</span><b>C{c}</b></div><div class="server-row"><span>本轮耗时</span><b>{elapsed}</b></div>{error}</div>'


def benchmark() -> str:
    rows = []
    for c in (1, 2, 4, 8, 9, 10, 11, 12, 13, 14, 15):
        source = ROOT / "evidence/performance/final" / f"c{c}/summary.json"
        if not source.exists():
            continue
        data = json.loads(source.read_text())
        ttft, tpot = data["metrics"]["time_to_first_token"]["p95"] / 1000, data["metrics"]["inter_token_latency"]["p95"]
        status = "通过" if ttft <= 10 and tpot <= 40 else "未通过"
        href = f"https://github.com/Nehawjl/ucloud-competition/tree/main/evidence/performance/final/c{c}"
        rows.append(f'<tr><td>C{c}</td><td>{data["output_token_throughput"]:.3f}</td><td>{ttft:.3f}</td><td>{tpot:.3f}</td><td>{status}</td><td><a href="{href}" target="_blank" rel="noopener">原始记录 ↗</a></td></tr>')
    return '<div class="benchmark"><h2>已完成的 Benchmark</h2><p>单卡 RTX 4090 · 8192 输入 / 512 输出 · 每档 8 预热 + 100 正式请求</p><table><thead><tr><th>并发</th><th>输出 token/s</th><th>TTFT P95 / s</th><th>TPOT P95 / ms</th><th>双 SLO</th><th>记录</th></tr></thead><tbody>' + ''.join(rows) + '</tbody></table></div>'


def build(config: Config) -> gr.Blocks:
    def stop(session):
        with ACTIVE_LOCK:
            run = ACTIVE.get(session)
        if run:
            run.cancel.set()
        return "停止请求已发送。" if run else "当前没有运行中的任务。"

    async def run_batch(c, rounds, limit, prompt, thinking, session):
        run = Round(config, int(c), int(rounds), int(limit), prompt, thinking)
        with ACTIVE_LOCK:
            ACTIVE[session] = run
        try:
            yield kpis(run), board(run), monitor(run), None, ""
            async for snapshot in run.stream():
                report = str(snapshot.directory / "report.json") if snapshot.ended else None
                yield kpis(snapshot), board(snapshot), monitor(snapshot), report, ""
        finally:
            with ACTIVE_LOCK:
                ACTIVE.pop(session, None)

    async def chat(message, history, thinking, session):
        if not message.strip():
            yield history, "", kpis()
            return
        clean = message_history(history)
        run = Round(config, 1, 1, 2048, message, thinking, clean)
        with ACTIVE_LOCK:
            ACTIVE[session] = run
        updated = clean + [{"role": "user", "content": message}, {"role": "assistant", "content": ""}]
        try:
            async for snapshot in run.stream():
                request = snapshot.requests[0]
                updated[-1]["content"] = request.text or request.error or ("思考中…" if request.reasoning else snapshot.server_error or "等待首 Token…")
                yield updated, "", kpis(snapshot)
        finally:
            with ACTIVE_LOCK:
                ACTIVE.pop(session, None)

    with gr.Blocks(title="Qwen · 单卡并发推理", analytics_enabled=False, fill_width=True) as app:
        session = gr.State(value=lambda: uuid.uuid4().hex)
        gr.HTML(f'<div class="masthead"><div><div class="brand">QWEN <span>/ AI INFRA</span></div><h1>单卡并发推理</h1><p>同一张 GPU，同时处理多路请求。</p></div><div class="config-badge"><span>{esc(config.model)} · RTX 4090</span><b>{esc(config.label)}</b></div></div>')
        with gr.Tabs():
            with gr.Tab("并发演示"):
                with gr.Row(elem_classes="control-bar"):
                    c = gr.Dropdown(choices=[(f"C{i}", i) for i in range(1, 16)], value=8, label="并发数 C", min_width=130, scale=1)
                    rounds = gr.Slider(1, 4, value=2, step=1, label="每路任务数", min_width=160, scale=1)
                    limit = gr.Dropdown([128, 256, 512, 1024, 2048], value=512, label="输出 token 上限", min_width=150, scale=1)
                    thinking = gr.Checkbox(value=False, label="思考模式", min_width=120, scale=1)
                    start = gr.Button("开始并发演示", variant="primary", min_width=180, scale=1)
                    stop_button = gr.Button("停止", min_width=80, scale=0)
                with gr.Accordion("自定义任务（留空使用不同的预设问题）", open=False):
                    prompt = gr.Textbox(label="任务内容", lines=3, max_lines=6, placeholder="所有并发请求独立执行这项任务；每个请求都有自己的输出。")
                stats = gr.HTML(kpis())
                with gr.Row(equal_height=False):
                    with gr.Column(scale=4, min_width=480):
                        cards = gr.HTML(board())
                    with gr.Column(scale=1, min_width=260):
                        server = gr.HTML(monitor())
                        download = gr.File(label="下载本轮记录", interactive=False)
                notice = gr.Markdown("")
                gr.Markdown("实时演示不替代 100 请求 Benchmark。token 数来自服务端；KV 占用不等于整卡显存占用。", elem_classes="caption")
                start.click(run_batch, [c, rounds, limit, prompt, thinking, session], [stats, cards, server, download, notice],
                            concurrency_id="inference", concurrency_limit=1, trigger_mode="once", stream_every=.2, show_progress="minimal", api_name="run_demo")
                stop_button.click(stop, session, notice, queue=False, api_visibility="private")
            with gr.Tab("自由对话"):
                chat_stats = gr.HTML(kpis())
                chatbot = gr.Chatbot(height=420, label="Qwen3.8-27B", buttons=["copy"], allow_file_downloads=False)
                with gr.Row():
                    message = gr.Textbox(label="问题", placeholder="输入问题，边生成边查看请求指标。", scale=6)
                    send = gr.Button("发送", variant="primary", scale=1)
                chat_thinking = gr.Checkbox(False, label="思考模式")
                chat_notice = gr.Markdown("")
                chat_stop = gr.Button("停止生成")
                for event_index, event in enumerate((send.click, message.submit)):
                    event(chat, [message, chatbot, chat_thinking, session], [chatbot, message, chat_stats],
                          concurrency_id="inference", concurrency_limit=1, trigger_mode="once", stream_every=.2, show_progress="minimal",
                          api_name="chat" if event_index == 0 else "chat_enter")
                chat_stop.click(stop, session, chat_notice, queue=False, api_visibility="private")
                gr.Markdown("并发演示与自由对话共享同一推理队列，不会在演示过程中额外插入聊天请求。", elem_classes="caption")
            with gr.Tab("Benchmark"):
                gr.HTML(benchmark())
                gr.Markdown("[Benchmark 报告](https://github.com/Nehawjl/ucloud-competition/blob/main/docs/BENCHMARK.md) · [全部展示结果与来源](https://github.com/Nehawjl/ucloud-competition/blob/main/evidence/README.md)")
    return app.queue(max_size=8, default_concurrency_limit=1)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", default=os.environ.get("DEMO_BASE_URL", "http://127.0.0.1:18109/v1"))
    parser.add_argument("--model", default="qwen3.8-27b")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=7860)
    parser.add_argument("--output-dir", type=Path, default=ROOT / ".runtime/demo-records")
    args = parser.parse_args()
    user, password = os.environ.get("DEMO_USER"), os.environ.get("DEMO_PASSWORD")
    if bool(user) != bool(password):
        parser.error("DEMO_USER and DEMO_PASSWORD must be set together")
    if args.host not in ("127.0.0.1", "localhost", "::1") and not (user and password):
        parser.error("Non-loopback binding requires DEMO_USER and DEMO_PASSWORD")
    config = Config(args.base_url, args.model, args.output_dir, os.environ.get("DEMO_API_KEY", ""))
    config.output_dir.mkdir(parents=True, exist_ok=True, mode=0o700)
    app = build(config)
    app.launch(server_name=args.host, server_port=args.port, share=False, show_error=False,
               theme=gr.themes.Soft(primary_hue="emerald", neutral_hue="slate",
                                    font=["Inter", "Noto Sans CJK SC", "sans-serif"], font_mono=["ui-monospace", "monospace"]),
               css_paths=Path(__file__).with_name("style.css"), footer_links=[],
               allowed_paths=[str(config.output_dir)], auth=(user, password) if user and password else None)


if __name__ == "__main__":
    main()
