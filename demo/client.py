from __future__ import annotations

import asyncio
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
import json
import hashlib
import platform
import math
from pathlib import Path
import re
import threading
import time
from typing import AsyncIterator
from urllib.parse import urlsplit
import uuid

import httpx

CLIENT_SHA256 = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()

TASKS = [
    ("代码解释", "用一个简短的 Python 例子解释生产者与消费者，以及队列为什么有用。"),
    ("文档总结", "用五个要点介绍软件项目的可复现性，并给出一个最小实践清单。"),
    ("结构化抽取", "把这段信息整理成 JSON：图书《算法入门》，价格 68 元，库存 12 本，分类为计算机。只输出 JSON。"),
    ("排障分析", "一个本地 Web 服务出现请求排队，但 CPU 利用率不高。列出排查顺序及每步要观察的指标。"),
    ("SQL 编写", "表 orders 有 user_id、amount、created_at 三列。写 SQL 统计每位用户本月的订单金额，并解释查询。"),
    ("概念讲解", "面向刚接触计算机系统的同学，解释吞吐量与延迟的区别，并举一个排队的例子。"),
    ("测试设计", "为一个接受整数列表、返回最大值的函数设计六个测试用例，覆盖边界和异常输入。"),
    ("文字改写", "把‘本系统通过多种手段显著提升处理能力并提供较好的用户体验’改写为三种更具体、简洁的技术表述。不要编造数字。"),
]


def message_history(history: list[dict]) -> list[dict]:
    result = []
    for message in history[-20:]:
        if message.get("role") not in ("user", "assistant"):
            continue
        content = message.get("content", "")
        if isinstance(content, list):
            content = "\n".join(item.get("text", "") for item in content if isinstance(item, dict) and item.get("type") == "text")
        if isinstance(content, str):
            result.append({"role": message["role"], "content": content})
    return result


def utc() -> str:
    return datetime.now(timezone.utc).isoformat()


@dataclass
class Config:
    base_url: str
    model: str
    output_dir: Path
    api_key: str = field(default="", repr=False)
    label: str = "W4A4 · MTP2 · FP8 KV · 融合"

    def __post_init__(self) -> None:
        url = urlsplit(self.base_url)
        if url.scheme not in ("http", "https") or not url.hostname or url.username or url.password or url.query or url.fragment:
            raise ValueError("Invalid inference URL; put credentials in DEMO_API_KEY, not the URL")
        self.base_url = self.base_url.rstrip("/")
        self.output_dir = self.output_dir.resolve()

    @property
    def metrics_url(self) -> str:
        return self.base_url.removesuffix("/v1") + "/metrics"

    def safe(self, value: str) -> str:
        return value.replace(self.api_key, "<REDACTED>") if self.api_key else value


@dataclass
class Request:
    index: int
    slot: int
    title: str
    prompt: str
    state: str = "等待"
    started_at: str | None = None
    ended_at: str | None = None
    started: float | None = None
    first: float | None = None
    ended: float | None = None
    prompt_tokens: int | None = None
    output_tokens: int | None = None
    text: str = ""
    reasoning: str = ""
    finish_reason: str | None = None
    error: str | None = None

    def timing(self, now: float | None = None) -> dict:
        elapsed = ((self.ended or now or time.perf_counter()) - self.started) if self.started is not None else None
        ttft = self.first - self.started if self.first is not None and self.started is not None else None
        tpot = None
        if self.state == "完成" and self.output_tokens and self.output_tokens > 1 and ttft is not None and elapsed is not None:
            tpot = (elapsed - ttft) * 1000 / (self.output_tokens - 1)
        return {"elapsed_s": elapsed, "ttft_s": ttft, "tpot_ms": tpot,
                "decode_token_s": 1000 / tpot if tpot and tpot > 0 else None}

    def record(self) -> dict:
        result = asdict(self)
        for name in ("started", "first", "ended"):
            result.pop(name)
        return {**result, **self.timing()}


def parse_metrics(text: str) -> dict:
    data: dict[str, list[float]] = {}
    for line in text.splitlines():
        match = re.match(r'^([\w:]+)(?:\{.*\})?\s+([-+\deE.]+)(?:\s+\d+)?$', line)
        if match:
            value = float(match[2])
            if math.isfinite(value):
                data.setdefault(match[1], []).append(value)
    def total(*names):
        for name in names:
            if name in data:
                return sum(data[name])
        return None
    kv = data.get("vllm:kv_cache_usage_perc", data.get("vllm:gpu_cache_usage_perc", []))
    return {"running": total("vllm:num_requests_running"), "waiting": total("vllm:num_requests_waiting"),
            "kv_percent": max(kv) * 100 if kv else None,
            "generation_tokens": total("vllm:generation_tokens_total")}


class Round:
    def __init__(self, config: Config, concurrency: int, rounds: int = 2, max_tokens: int = 512,
                 custom_prompt: str = "", thinking: bool = False, history: list[dict] | None = None):
        if not 1 <= int(concurrency) <= 15 or not 1 <= int(rounds) <= 4 or not 64 <= int(max_tokens) <= 2048:
            raise ValueError("C must be 1–15, rounds 1–4, max_tokens 64–2048")
        if len(custom_prompt) > 32000:
            raise ValueError("Prompt is too long")
        self.config, self.c, self.rounds, self.max_tokens = config, int(concurrency), int(rounds), int(max_tokens)
        self.thinking, self.history = thinking, history or []
        self.run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + f"-demo-c{self.c}-" + uuid.uuid4().hex[:8]
        self.directory = config.output_dir / self.run_id
        self.directory.mkdir(parents=True, exist_ok=False, mode=0o700)
        self.requests = []
        for index in range(self.c * self.rounds):
            title, prompt = TASKS[index % len(TASKS)]
            self.requests.append(Request(index + 1, index % self.c, title if not custom_prompt else "自定义任务",
                                         custom_prompt.strip() or prompt))
        self.slots: list[Request | None] = [None] * self.c
        self.cancel = threading.Event()
        self.started_at, self.started = utc(), time.perf_counter()
        self.ended_at: str | None = None
        self.ended: float | None = None
        self.status = "连接服务"
        self.server: dict = {}
        self.server_error = ""
        self.rates: list[tuple[float, float]] = []
        self.previous_counter: tuple[float, float] | None = None
        self.monitor_stop = asyncio.Event()

    def summary(self) -> dict:
        elapsed = (self.ended or time.perf_counter()) - self.started
        started = [r.started for r in self.requests if r.started is not None]
        terminal = all(r.state in ("完成", "失败", "已停止", "未执行") for r in self.requests)
        measurement_end = max((r.ended for r in self.requests if r.ended is not None), default=time.perf_counter()) if terminal else time.perf_counter()
        measurement_duration = measurement_end - min(started) if started else 0
        done = [r for r in self.requests if r.state == "完成"]
        known_tokens = sum(r.output_tokens or 0 for r in self.requests)
        return {"run_id": self.run_id, "status": self.status, "started_at": self.started_at,
                "ended_at": self.ended_at, "concurrency": self.c, "request_count": len(self.requests),
                "completed": len(done), "failed": sum(r.state == "失败" for r in self.requests),
                "elapsed_s": elapsed, "output_tokens": known_tokens,
                "measurement_duration_s": measurement_duration,
                "round_token_s": known_tokens / measurement_duration if measurement_duration > 0 else 0,
                "model": self.config.model, "configuration": self.config.label,
                "base_url": self.config.base_url, "python": platform.python_version(),
                "httpx": httpx.__version__, "client_sha256": CLIENT_SHA256,
                "generation": {"max_tokens": self.max_tokens, "temperature": 0, "thinking": self.thinking},
                "server": self.server, "requests": [r.record() for r in self.requests]}

    async def _request(self, client: httpx.AsyncClient, request: Request) -> None:
        request.state, request.started_at, request.started = "等待首 Token", utc(), time.perf_counter()
        body = {"model": self.config.model, "messages": self.history + [{"role": "user", "content": request.prompt}],
                "max_tokens": self.max_tokens, "temperature": 0, "stream": True,
                "stream_options": {"include_usage": True, "continuous_usage_stats": True},
                "chat_template_kwargs": {"enable_thinking": self.thinking}}
        raw_path = self.directory / f"request-{request.index:03d}.jsonl"
        try:
            with raw_path.open("x", encoding="utf-8") as log:
                terminal = False
                async with client.stream("POST", self.config.base_url + "/chat/completions", json=body) as response:
                    if response.is_error:
                        detail = (await response.aread()).decode(errors="replace")[:1000]
                        raise RuntimeError(f"HTTP {response.status_code}: {self.config.safe(detail)}")
                    async for line in response.aiter_lines():
                        if not line.startswith("data:"):
                            continue
                        payload = line[5:].strip()
                        log.write(self.config.safe(json.dumps({"elapsed_s": time.perf_counter() - request.started,
                                                               "data": payload}, ensure_ascii=False)) + "\n")
                        if payload == "[DONE]":
                            terminal = True
                            break
                        chunk = json.loads(payload)
                        if chunk.get("error"):
                            raise RuntimeError(self.config.safe(str(chunk["error"])))
                        usage = chunk.get("usage") or {}
                        if usage.get("prompt_tokens") is not None:
                            request.prompt_tokens = int(usage["prompt_tokens"])
                        if usage.get("completion_tokens") is not None:
                            request.output_tokens = int(usage["completion_tokens"])
                        for choice in chunk.get("choices", []):
                            delta = choice.get("delta") or {}
                            request.text += delta.get("content") or ""
                            request.reasoning += delta.get("reasoning_content") or delta.get("reasoning") or ""
                            request.finish_reason = choice.get("finish_reason") or request.finish_reason
                        if request.first is None and (request.text or request.reasoning or request.output_tokens):
                            request.first = time.perf_counter()
                            request.state = "生成中"
                if not terminal or not request.finish_reason:
                    raise RuntimeError("流式连接提前结束，未收到完整完成标记")
                if request.output_tokens is None or request.prompt_tokens is None:
                    raise RuntimeError("服务未返回 token 统计，不能计算生成速度")
                request.state = "完成"
        except asyncio.CancelledError:
            request.state = "已停止"
            raise
        except Exception as error:
            request.state, request.error = "失败", self.config.safe(str(error))[:1000]
        finally:
            request.ended, request.ended_at = time.perf_counter(), utc()

    async def _monitor(self, client: httpx.AsyncClient) -> None:
        while not self.monitor_stop.is_set():
            try:
                response = await client.get(self.config.metrics_url, timeout=2)
                response.raise_for_status()
                self.server = parse_metrics(response.text)
                self.server_error = ""
                counter, now = self.server.get("generation_tokens"), time.perf_counter()
                if counter is not None and self.previous_counter:
                    old_t, old_count = self.previous_counter
                    if counter >= old_count and now > old_t:
                        self.rates.append((now - self.started, (counter - old_count) / (now - old_t)))
                if counter is not None:
                    self.previous_counter = now, counter
            except Exception:
                self.server_error = "服务端监控暂不可用"
            try:
                await asyncio.wait_for(self.monitor_stop.wait(), timeout=1)
            except TimeoutError:
                pass

    async def stream(self) -> AsyncIterator[Round]:
        headers = {"Authorization": "Bearer " + self.config.api_key} if self.config.api_key else {}
        workers: list[asyncio.Task] = []
        monitor = None
        queue: asyncio.Queue[Request] = asyncio.Queue()
        for request in self.requests:
            queue.put_nowait(request)
        try:
            async with httpx.AsyncClient(headers=headers, timeout=httpx.Timeout(600, connect=10),
                                         limits=httpx.Limits(max_connections=self.c + 4, max_keepalive_connections=self.c + 4),
                                         trust_env=False) as client:
                response = await client.get(self.config.base_url + "/models", timeout=10)
                response.raise_for_status()
                models = {m["id"] for m in response.json().get("data", [])}
                if self.config.model not in models:
                    raise RuntimeError("服务中未找到配置的模型：" + self.config.model)
                self.status = "运行中"
                monitor = asyncio.create_task(self._monitor(client))
                async def worker(slot: int) -> None:
                    while not self.cancel.is_set():
                        try:
                            request = queue.get_nowait()
                        except asyncio.QueueEmpty:
                            return
                        request.slot = slot
                        self.slots[slot] = request
                        await self._request(client, request)
                workers = [asyncio.create_task(worker(slot)) for slot in range(self.c)]
                while not all(task.done() for task in workers):
                    if self.cancel.is_set():
                        for task in workers:
                            task.cancel()
                        break
                    yield self
                    await asyncio.sleep(.15)
                await asyncio.gather(*workers, return_exceptions=True)
                self.status = "已停止" if self.cancel.is_set() else ("部分失败" if any(r.state == "失败" for r in self.requests) else "已完成")
                self.monitor_stop.set()
                if monitor:
                    monitor.cancel()
                    await asyncio.gather(monitor, return_exceptions=True)
                    monitor = None
        except asyncio.CancelledError:
            self.status = "已停止"
            raise
        except Exception as error:
            self.status = "连接失败"
            self.server_error = self.config.safe(str(error))[:500]
        finally:
            for task in workers:
                if not task.done():
                    task.cancel()
            if workers:
                await asyncio.gather(*workers, return_exceptions=True)
            self.monitor_stop.set()
            if monitor:
                monitor.cancel()
                await asyncio.gather(monitor, return_exceptions=True)
            for request in self.requests:
                if request.state == "等待":
                    request.state = "未执行"
            self.ended, self.ended_at = time.perf_counter(), utc()
            (self.directory / "report.json").write_text(self.config.safe(json.dumps(self.summary(), ensure_ascii=False, indent=2)) + "\n", encoding="utf-8")
        yield self
