#!/bin/bash
# Free space on a full RunPod container disk after failed/partial model downloads.
# Does NOT delete complete .safetensors under MODELS_ROOT (unless FORCE_PURGE_MODELS=1).
set -euo pipefail

echo "=== disk before ==="
df -h / /workspace 2>/dev/null || df -h /

echo "=== kill downloaders ==="
pkill -x wget 2>/dev/null || true
pkill -x aria2c 2>/dev/null || true
pkill -x curl 2>/dev/null || true

echo "=== remove incomplete download junk ==="
# aria2 / wget partials
find /workspace /tmp /root -xdev \
  \( -name '*.part' -o -name '*.aria2' -o -name '*.tmp' -o -name '*.crdownload' \) \
  -type f -print -delete 2>/dev/null || true

# pip caches
rm -rf /root/.cache/pip /workspace/venv-bw/pip-cache 2>/dev/null || true
rm -rf /tmp/aria2* /tmp/*.deb /tmp/*.tar.bz2 2>/dev/null || true

# HuggingFace hub cache (can be huge duplicates of models)
if [ "${KEEP_HF_CACHE:-0}" != "1" ]; then
  du -sh /root/.cache/huggingface 2>/dev/null || true
  rm -rf /root/.cache/huggingface 2>/dev/null || true
fi

if [ "${FORCE_PURGE_MODELS:-0}" = "1" ]; then
  echo "FORCE_PURGE_MODELS=1 — removing /workspace/models"
  rm -rf /workspace/models
fi

echo "=== largest dirs under /workspace (top 15) ==="
du -h -d 1 /workspace 2>/dev/null | sort -hr | head -15 || true

echo "=== disk after ==="
df -h / /workspace 2>/dev/null || df -h /

echo "Done."
echo "Next: attach a 100GB Network Volume and download TO the volume:"
echo "  MODELS_ROOT=/runpod-volume/models bash pod_aria2_dl.sh"
echo "Do NOT download ~45GB models onto a small container disk."
