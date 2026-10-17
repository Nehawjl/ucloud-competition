"""CPU-only stream and concurrency checks. No model requests or benchmark claims."""
import asyncio
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

try:
    import httpx
    from demo.client import Config, Round, parse_metrics, message_history
except ImportError:
    httpx = None


@unittest.skipIf(httpx is None, "Install demo/requirements.txt for demo tests")
class DemoTests(unittest.IsolatedAsyncioTestCase):
    async def test_real_parallelism_and_usage_not_chunks(self):
        active = peak = 0
        class Stream(httpx.AsyncByteStream):
            def __init__(self):
                nonlocal active, peak
                self.open = True
                active += 1
                peak = max(peak, active)

            async def __aiter__(self):
                try:
                    await asyncio.sleep(.025)
                    first = {"choices": [{"delta": {"content": "一个数据包包含多个 token"}, "finish_reason": None}],
                             "usage": {"prompt_tokens": 100, "completion_tokens": 3}}
                    yield ("data: " + json.dumps(first) + "\n\n").encode()
                    await asyncio.sleep(.025)
                    last = {"choices": [{"delta": {"content": "。"}, "finish_reason": "stop"}],
                            "usage": {"prompt_tokens": 100, "completion_tokens": 5}}
                    yield ("data: " + json.dumps(last) + "\n\ndata: [DONE]\n\n").encode()
                finally:
                    await self.aclose()

            async def aclose(self):
                nonlocal active
                if self.open:
                    self.open = False
                    active -= 1
        async def handler(request):
            if request.url.path.endswith("/models"):
                return httpx.Response(200, json={"data": [{"id": "test"}]})
            if request.url.path == "/metrics":
                return httpx.Response(200, text=f"vllm:num_requests_running {active}\nvllm:num_requests_waiting 0\nvllm:kv_cache_usage_perc 0.2\nvllm:generation_tokens_total 10\n")
            return httpx.Response(200, stream=Stream())
        real_client = httpx.AsyncClient
        with tempfile.TemporaryDirectory() as directory:
            config = Config("http://test/v1", "test", Path(directory))
            with patch("demo.client.httpx.AsyncClient", side_effect=lambda **kw: real_client(transport=httpx.MockTransport(handler), **kw)):
                run = Round(config, 4, rounds=2, max_tokens=64)
                async for _ in run.stream():
                    pass
            self.assertEqual(peak, 4)
            self.assertEqual(run.status, "已完成")
            self.assertEqual(run.summary()["completed"], 8)
            self.assertEqual(run.summary()["output_tokens"], 40)
            self.assertTrue(all(r.output_tokens == 5 and r.prompt_tokens == 100 for r in run.requests))
            self.assertTrue(all(r.timing()["tpot_ms"] > 0 for r in run.requests))
            self.assertTrue((run.directory / "report.json").exists())

    async def test_missing_usage_fails_without_estimating_tokens(self):
        async def handler(request):
            if request.url.path.endswith("/models"):
                return httpx.Response(200, json={"data": [{"id": "test"}]})
            if request.url.path == "/metrics":
                return httpx.Response(200, text="")
            return httpx.Response(200, text='data: {"choices":[{"delta":{"content":"abc"},"finish_reason":"stop"}]}\n\ndata: [DONE]\n\n')
        real_client = httpx.AsyncClient
        with tempfile.TemporaryDirectory() as directory:
            with patch("demo.client.httpx.AsyncClient", side_effect=lambda **kw: real_client(transport=httpx.MockTransport(handler), **kw)):
                run = Round(Config("http://test/v1", "test", Path(directory)), 1, 1, 64)
                async for _ in run.stream():
                    pass
            self.assertEqual(run.status, "部分失败")
            self.assertIsNone(run.requests[0].output_tokens)
            self.assertIsNone(run.requests[0].timing()["tpot_ms"])

    async def test_cancellation_and_secret_redaction(self):
        secret = "unit-test-credential-not-a-real-key"
        class Stream(httpx.AsyncByteStream):
            async def __aiter__(self):
                await asyncio.sleep(5)
                yield b"data: [DONE]\n\n"
        async def handler(request):
            if request.url.path.endswith("/models"):
                return httpx.Response(200, json={"data": [{"id": "test"}]})
            if request.url.path == "/metrics":
                return httpx.Response(200, text="")
            return httpx.Response(200, stream=Stream())
        real_client = httpx.AsyncClient
        with tempfile.TemporaryDirectory() as directory:
            with patch("demo.client.httpx.AsyncClient", side_effect=lambda **kw: real_client(transport=httpx.MockTransport(handler), **kw)):
                run = Round(Config("http://test/v1", "test", Path(directory), secret), 2, 2, 64)
                count = 0
                async for _ in run.stream():
                    count += 1
                    if count == 2:
                        run.cancel.set()
            self.assertEqual(run.status, "已停止")
            self.assertEqual(run.summary()["completed"], 0)
            self.assertNotIn(secret, (run.directory / "report.json").read_text())

    async def test_metrics_and_validation(self):
        self.assertEqual(message_history([{"role": "user", "content": [{"type": "text", "text": "上一条问题"}]}]),
                         [{"role": "user", "content": "上一条问题"}])
        metrics = parse_metrics('vllm:num_requests_running{engine="0"} 8\nvllm:num_requests_waiting 2\nvllm:kv_cache_usage_perc 0.282\nvllm:generation_tokens_total 512\n')
        self.assertEqual(metrics["running"], 8)
        self.assertEqual(metrics["waiting"], 2)
        self.assertAlmostEqual(metrics["kv_percent"], 28.2)
        with tempfile.TemporaryDirectory() as directory:
            config = Config("http://test/v1", "test", Path(directory))
            for c in (0, 16):
                with self.assertRaises(ValueError):
                    Round(config, c)
        with self.assertRaises(ValueError):
            Config("http://user:password@test/v1", "test", Path("/tmp"))


if __name__ == "__main__":
    unittest.main()
