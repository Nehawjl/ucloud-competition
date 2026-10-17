"""Gradio HTTP integration against a local stub, never against a GPU service."""
import importlib.util
import json
from pathlib import Path
import socket
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import tempfile
import threading
import time
import unittest


@unittest.skipUnless(importlib.util.find_spec("gradio"), "Install demo/requirements.txt for UI tests")
class DemoUITests(unittest.TestCase):
    def test_http_ui_concurrency_chat_history_and_choices(self):
        from demo.app import build
        from demo.client import Config
        from gradio_client import Client
        import httpx
        active = peak = total_tokens = 0
        messages_seen = []
        lock = threading.Lock()

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *args):
                pass

            def do_GET(self):
                if self.path == "/v1/models":
                    value = json.dumps({"data": [{"id": "demo-test-stub"}]}).encode()
                    content_type = "application/json"
                else:
                    value = f"vllm:num_requests_running {active}\nvllm:num_requests_waiting 0\nvllm:kv_cache_usage_perc 0.1\nvllm:generation_tokens_total {total_tokens}\n".encode()
                    content_type = "text/plain"
                self.send_response(200)
                self.send_header("Content-Type", content_type)
                self.send_header("Content-Length", str(len(value)))
                self.end_headers()
                self.wfile.write(value)

            def do_POST(self):
                nonlocal active, peak, total_tokens
                body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
                messages_seen.append(body["messages"])
                with lock:
                    active += 1
                    peak = max(peak, active)
                try:
                    self.send_response(200)
                    self.send_header("Content-Type", "text/event-stream")
                    self.end_headers()
                    time.sleep(.08)
                    for text, tokens, finish in [("功能测试", 3, None), ("完成", 5, "stop")]:
                        data = {"choices": [{"delta": {"content": text}, "finish_reason": finish}],
                                "usage": {"prompt_tokens": 100, "completion_tokens": tokens}}
                        self.wfile.write(("data: " + json.dumps(data) + "\n\n").encode())
                        self.wfile.flush()
                        time.sleep(.08)
                    self.wfile.write(b"data: [DONE]\n\n")
                    self.wfile.flush()
                    with lock:
                        total_tokens += 5
                finally:
                    with lock:
                        active -= 1

        class StubServer(ThreadingHTTPServer):
            request_queue_size = 64

        stub = StubServer(("127.0.0.1", 0), Handler)
        threading.Thread(target=stub.serve_forever, daemon=True).start()
        app = None
        try:
            with tempfile.TemporaryDirectory() as directory:
                config = Config(f"http://127.0.0.1:{stub.server_port}/v1", "demo-test-stub", Path(directory), label="接口功能测试（模拟数据）")
                app = build(config)
                with socket.socket() as sock:
                    sock.bind(("127.0.0.1", 0))
                    port = sock.getsockname()[1]
                _, url, _ = app.launch(server_name="127.0.0.1", server_port=port, prevent_thread_lock=True,
                                       share=False, quiet=True, footer_links=[], allowed_paths=[directory])
                with httpx.Client(trust_env=False) as transport:
                    components = transport.get(url + "config").json()["components"]
                choices = next(c["props"] for c in components if c["type"] == "dropdown")
                self.assertEqual(len(choices["choices"]), 15)
                self.assertEqual(choices["value"], 8)
                client = Client(url, verbose=False, httpx_kwargs={"trust_env": False})
                for concurrency in (1, 4, 15):
                    with lock:
                        peak = 0
                    result = client.predict(concurrency, 1, 128, "并发功能测试", False, api_name="/run_demo")
                    report = json.loads(Path(result[3]).read_text())
                    self.assertEqual(report["completed"], concurrency)
                    self.assertEqual(report["output_tokens"], concurrency * 5)
                    self.assertEqual(peak, concurrency)
                    self.assertIn("已完成", result[0])
                history, _, _ = client.predict("第一条", [], False, api_name="/chat")
                client.predict("第二条", history, False, api_name="/chat")
                self.assertEqual(len(messages_seen[-1]), 3)
                self.assertEqual(messages_seen[-1][0]["content"], "第一条")
        finally:
            if app:
                app.close()
            stub.shutdown()
            stub.server_close()


if __name__ == "__main__":
    unittest.main()
