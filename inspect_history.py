#!/usr/bin/env python3
import json
import sys
import urllib.request
from pathlib import Path

pid = sys.argv[1] if len(sys.argv) > 1 else "9a035323-22ce-4192-94be-a135cda04618"
url = f"http://127.0.0.1:8188/history/{pid}"
h = json.loads(urllib.request.urlopen(url, timeout=30).read())
if not h:
    print("empty history")
    # try full history last
    h = json.loads(urllib.request.urlopen("http://127.0.0.1:8188/history", timeout=30).read())
    print("full history keys", list(h.keys())[-5:])
    sys.exit(0)
key = list(h.keys())[0]
print("prompt_id", key)
print("status", h[key].get("status"))
outs = h[key].get("outputs", {})
print("output_nodes", list(outs.keys()))
for nid, o in outs.items():
    print("node", nid, "keys", list(o.keys()))
    for k, v in o.items():
        s = str(v)
        print(f"  {k}: {s[:400]}")

# search output dirs
for d in ["/ComfyUI/output", "/workspace/ComfyUI/output", "/workspace/ComfyUI/temp"]:
    p = Path(d)
    if p.is_dir():
        files = sorted(p.rglob("*"), key=lambda x: x.stat().st_mtime if x.is_file() else 0, reverse=True)[:20]
        print("dir", d)
        for f in files:
            if f.is_file():
                print(" ", f.stat().st_size, f)
