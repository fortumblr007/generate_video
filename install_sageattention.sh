#!/usr/bin/env bash
# Install SageAttention for RunPod Blackwell (sm_120) + CUDA 12.8 + Python 3.12.
# Prebuilt wheel targets torch 2.10 but imports/runs on torch 2.11+cu128 (verified).
#
# IMPORTANT: ComfyUI MUST be started with the same Python that gets this package.
#   /workspace/venv-bw/bin/python -u main.py --listen 127.0.0.1 --port 8188
# System /usr/bin/python3 does NOT see this install.
set -euo pipefail

VENV_PY="${VENV_PY:-/workspace/venv-bw/bin/python}"
PIP="${VENV_PY%/*}/pip"
SAGE_WHL="${SAGE_WHL:-https://huggingface.co/yo9otatara/prebuilt_wheels/resolve/main/sageattention-2.2.0%2Bcu128torch2.10.0sm120-cp312-cp312-linux_x86_64.whl}"

echo "Python: $($VENV_PY -V)"
$VENV_PY - <<'PY'
import torch
print("torch", torch.__version__, "cuda", torch.version.cuda, "cap", torch.cuda.get_device_capability() if torch.cuda.is_available() else None)
PY

# Triton is already required by sage; keep existing if present.
$PIP install -U --no-deps "$SAGE_WHL"

$VENV_PY - <<'PY'
from sageattention import sageattn
import torch
q = torch.randn(1, 8, 64, 64, device="cuda", dtype=torch.float16)
k = torch.randn(1, 8, 64, 64, device="cuda", dtype=torch.float16)
v = torch.randn(1, 8, 64, 64, device="cuda", dtype=torch.float16)
out = sageattn(q, k, v, tensor_layout="HND", is_causal=False)
print("sageattention OK", getattr(out, "shape", type(out)))
PY

echo "SageAttention install complete."
echo "Restart ComfyUI with: $VENV_PY -u /workspace/ComfyUI/main.py --listen 127.0.0.1 --port 8188"
