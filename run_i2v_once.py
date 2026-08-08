#!/usr/bin/env python3
"""One-shot I2V via handler (run on pod with ComfyUI already up)."""
import base64
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)) or "/workspace/generate_video")
os.chdir(os.environ.get("WORKDIR", "/workspace/generate_video"))

os.environ.setdefault("MODELS_ROOT", "/workspace/models")
os.environ.setdefault("SKIP_MODEL_DOWNLOAD", "1")
os.environ.setdefault("WORKFLOW_FILE", "/new_Wan22_api.json")
os.environ.setdefault("SERVER_ADDRESS", "127.0.0.1")
os.environ.setdefault("COMFY_INPUT_DIR", "/ComfyUI/input")

from handler import handler  # noqa: E402

def main():
    print("SUBMIT", time.strftime("%Y-%m-%dT%H:%M:%S"), flush=True)
    job = {
        "id": "direct",
        "input": {
            "prompt": os.environ.get(
                "I2V_PROMPT",
                "the girl is now wearing a see through sheer fabric that exposes her massive bare chest",
            ),
            "image_url": os.environ.get(
                "I2V_IMAGE_URL",
                "https://simp6.cuckcapital.cr/images4/ignaciaa.vrl-jaavi.v-Javiera-ignacia-024f830d82aa65b4e0.webp",
            ),
            "cfg": 1.0,
            "width": 480,
            "height": 832,
            "length": int(os.environ.get("I2V_LENGTH", "49")),
            "steps": 4,
            "seed": 42,
            "context_overlap": 48,
        },
    }
    out = handler(job)
    if not isinstance(out, dict):
        print("BAD_OUT", type(out), flush=True)
        sys.exit(1)
    if out.get("error"):
        print("ERROR", out["error"][:4000], flush=True)
        with open("/tmp/i2v_result_meta.json", "w") as f:
            json.dump(out, f, indent=2)
        sys.exit(2)
    vid = out.get("video")
    if not vid:
        print("NO_VIDEO keys=", list(out.keys()), flush=True)
        with open("/tmp/i2v_result_meta.json", "w") as f:
            json.dump(out, f, indent=2)
        sys.exit(3)
    raw = base64.b64decode(vid)
    path = os.environ.get("I2V_OUT", "/workspace/i2v_out.mp4")
    with open(path, "wb") as f:
        f.write(raw)
    print("SAVED", path, "bytes=", len(raw), flush=True)
    with open("/tmp/i2v_result_meta.json", "w") as f:
        json.dump({"status": "COMPLETED", "bytes": len(raw), "path": path}, f, indent=2)


if __name__ == "__main__":
    main()
