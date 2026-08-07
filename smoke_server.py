#!/usr/bin/env python3
"""
Minimal local HTTP server to exercise handler.py WITHOUT Docker or models.

  python smoke_server.py
  # then:
  curl -s http://127.0.0.1:8000/health
  curl -s -X POST http://127.0.0.1:8000/run -H "Content-Type: application/json" -d "{\"input\":{\"ping\":true}}"

Mirrors RunPod shape: POST body {"input": {...}} → handler output as JSON.
Real I2V still needs ComfyUI + weights; use ping first to prove the Python path.
"""
from __future__ import annotations

import json
import os
import sys
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

# Allow importing handler from same directory
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from handler import handler  # noqa: E402


HOST = os.getenv("SMOKE_HOST", "127.0.0.1")
PORT = int(os.getenv("SMOKE_PORT", "8000"))


class Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def _json(self, code: int, obj: dict):
        body = json.dumps(obj, ensure_ascii=False).encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, fmt, *args):
        sys.stderr.write("%s - %s\n" % (self.address_string(), fmt % args))

    def do_GET(self):
        if self.path in ("/", "/health", "/healthz"):
            self._json(
                200,
                {
                    "ok": True,
                    "service": "generate_video smoke",
                    "endpoints": {
                        "GET /health": "liveness",
                        "POST /run": 'body {"input": {...}} same as RunPod',
                        "POST /runsync": "alias of /run",
                    },
                    "hint": 'POST {"input":{"ping":true}} first — no models required',
                },
            )
            return
        self._json(404, {"error": "not found", "path": self.path})

    def do_POST(self):
        if self.path not in ("/run", "/runsync", "/"):
            self._json(404, {"error": "not found", "path": self.path})
            return
        length = int(self.headers.get("Content-Length") or 0)
        raw = self.rfile.read(length) if length else b"{}"
        try:
            payload = json.loads(raw.decode("utf-8") or "{}")
        except json.JSONDecodeError as e:
            self._json(400, {"error": f"invalid json: {e}"})
            return

        # Accept either {"input": {...}} (RunPod) or bare input fields
        if isinstance(payload, dict) and "input" in payload:
            job = {"input": payload["input"] or {}, "id": payload.get("id", "local-smoke")}
        else:
            job = {"input": payload if isinstance(payload, dict) else {}, "id": "local-smoke"}

        try:
            out = handler(job)
        except Exception as e:
            self._json(500, {"status": "FAILED", "error": str(e)})
            return

        # RunPod-like envelope
        if isinstance(out, dict) and out.get("error"):
            self._json(200, {"status": "FAILED", "output": out, "id": job["id"]})
        else:
            self._json(200, {"status": "COMPLETED", "output": out, "id": job["id"]})


def main():
    httpd = ThreadingHTTPServer((HOST, PORT), Handler)
    print(f"smoke_server listening on http://{HOST}:{PORT}", flush=True)
    print(f'  curl -s http://{HOST}:{PORT}/health', flush=True)
    print(
        f'  curl -s -X POST http://{HOST}:{PORT}/run -H "Content-Type: application/json" '
        f'-d "{{\\"input\\":{{\\"ping\\":true}}}}"',
        flush=True,
    )
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\nbye", flush=True)


if __name__ == "__main__":
    main()
