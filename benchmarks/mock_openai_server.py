"""Minimal OpenAI-compatible chat-completions server (stdlib only).

Stands in for `vllm serve` in tests and in the benchmark harness
self-check. It does NOT run a model — numbers measured against it
describe the harness and app overhead, never model or GPU performance.

    python -m benchmarks.mock_openai_server --port 8001 --tokens 64 --token-delay-ms 5

Failure injection (for tests): --fail-status 500, --hang-s 10 (delay
before first byte), --drop-after N (close the stream after N chunks).
"""
from __future__ import annotations
import argparse
import json
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Optional


class MockConfig:
    def __init__(self, tokens: int = 32, token_delay_ms: float = 0.0, fail_status: int = 0,
                 hang_s: float = 0.0, drop_after: Optional[int] = None, model: str = "mock-model"):
        self.tokens = tokens
        self.token_delay_ms = token_delay_ms
        self.fail_status = fail_status
        self.hang_s = hang_s
        self.drop_after = drop_after
        self.model = model


def _make_handler(cfg: MockConfig):
    class Handler(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def log_message(self, *args):  # keep test/benchmark output quiet
            pass

        def _json(self, status: int, body: dict):
            raw = json.dumps(body).encode()
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(raw)))
            self.end_headers()
            self.wfile.write(raw)

        def do_GET(self):
            if self.path in ("/health", "/v1/health"):
                return self._json(200, {"status": "ok"})
            if self.path == "/v1/models":
                return self._json(200, {"object": "list", "data": [{"id": cfg.model, "object": "model"}]})
            self._json(404, {"error": "not found"})

        def do_POST(self):
            length = int(self.headers.get("Content-Length", 0))
            req = json.loads(self.rfile.read(length) or b"{}")
            if self.path != "/v1/chat/completions":
                return self._json(404, {"error": "not found"})
            if cfg.hang_s:
                time.sleep(cfg.hang_s)
            if cfg.fail_status:
                return self._json(cfg.fail_status, {"error": {"message": "injected failure"}})
            n = min(cfg.tokens, int(req.get("max_tokens") or cfg.tokens))
            words = [f"tok{i} " for i in range(n)]
            created = int(time.time())
            if not req.get("stream"):
                time.sleep(cfg.token_delay_ms * n / 1000)
                return self._json(200, {
                    "id": "cmpl-mock", "object": "chat.completion", "created": created,
                    "model": cfg.model,
                    "choices": [{"index": 0, "finish_reason": "stop",
                                 "message": {"role": "assistant", "content": "".join(words)}}],
                    "usage": {"prompt_tokens": 0, "completion_tokens": n, "total_tokens": n},
                })
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.send_header("Transfer-Encoding", "chunked")
            self.end_headers()

            def chunk(payload: str):
                data = payload.encode()
                self.wfile.write(f"{len(data):x}\r\n".encode() + data + b"\r\n")
                self.wfile.flush()

            for i, w in enumerate(words):
                if cfg.drop_after is not None and i >= cfg.drop_after:
                    self.close_connection = True
                    return  # abort without the terminating chunk -> client sees a broken stream
                time.sleep(cfg.token_delay_ms / 1000)
                chunk("data: " + json.dumps({
                    "id": "cmpl-mock", "object": "chat.completion.chunk", "created": created,
                    "model": cfg.model,
                    "choices": [{"index": 0, "delta": {"content": w}, "finish_reason": None}],
                }) + "\n\n")
            chunk("data: [DONE]\n\n")
            self.wfile.write(b"0\r\n\r\n")

    return Handler


def start_in_thread(cfg: MockConfig, port: int = 0):
    """Starts the server on a background thread; returns (server, base_url)."""
    server = ThreadingHTTPServer(("127.0.0.1", port), _make_handler(cfg))
    server.daemon_threads = True
    threading.Thread(target=server.serve_forever, daemon=True).start()
    return server, f"http://127.0.0.1:{server.server_address[1]}/v1"


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8001)
    ap.add_argument("--tokens", type=int, default=64)
    ap.add_argument("--token-delay-ms", type=float, default=0.0)
    ap.add_argument("--model", default="mock-model")
    a = ap.parse_args()
    srv = ThreadingHTTPServer((a.host, a.port), _make_handler(
        MockConfig(tokens=a.tokens, token_delay_ms=a.token_delay_ms, model=a.model)))
    print(f"mock OpenAI server on http://{a.host}:{a.port}/v1 (model={a.model})")
    srv.serve_forever()
