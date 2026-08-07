#!/usr/bin/env python3
"""One-shot smoke: call handler.ping without starting a server."""
import json
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from handler import handler

cases = [
    ("ping", {"input": {"ping": True}}),
    ("missing_prompt", {"input": {}}),
    ("flf2v_rejected", {"input": {"prompt": "x", "end_image_url": "http://x"}}),
]

failed = 0
for name, job in cases:
    out = handler(job)
    print(f"=== {name} ===")
    print(json.dumps(out, indent=2))
    if name == "ping" and not out.get("ok"):
        failed += 1
    if name == "missing_prompt" and "prompt" not in str(out.get("error", "")).lower():
        failed += 1
    if name == "flf2v_rejected" and "FLF2V" not in str(out.get("error", "")):
        failed += 1

sys.exit(1 if failed else 0)
