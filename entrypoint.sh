#!/bin/bash

# Exit immediately if a command exits with a non-zero status.
set -e

# IMPORTANT (RunPod Serverless):
# Do NOT download multi‑GB models here. That blocks registration and leaves the
# worker stuck in "initializing". Models are ensured in handler.py on first job
# (or pre-seeded via Network Volume / BAKE_MODELS=1 image).
#
# Optional: DOWNLOAD_MODELS_ON_START=1 forces a pre-download before ComfyUI
# (only use with baked/volume setups or very long start timeouts).
if [ "${DOWNLOAD_MODELS_ON_START:-0}" = "1" ] && [ "${SKIP_MODEL_DOWNLOAD:-0}" != "1" ]; then
  echo "DOWNLOAD_MODELS_ON_START=1: pre-downloading weights..."
  /download_models.sh
fi

# Start ComfyUI in the background (sage-attn if available)
echo "Starting ComfyUI in the background..."
if python -c "import sageattention" 2>/dev/null; then
  python /ComfyUI/main.py --listen --use-sage-attention &
else
  echo "sageattention not found; starting ComfyUI without --use-sage-attention"
  python /ComfyUI/main.py --listen &
fi

# Wait for ComfyUI to be ready (code-only start should be fast)
echo "Waiting for ComfyUI to be ready..."
max_wait="${COMFYUI_START_TIMEOUT:-180}"
wait_count=0
while [ $wait_count -lt $max_wait ]; do
    if curl -s http://127.0.0.1:8188/ > /dev/null 2>&1; then
        echo "ComfyUI is ready!"
        break
    fi
    echo "Waiting for ComfyUI... ($wait_count/$max_wait)"
    sleep 2
    wait_count=$((wait_count + 2))
done

if [ $wait_count -ge $max_wait ]; then
    echo "Error: ComfyUI failed to start within $max_wait seconds"
    exit 1
fi

# Start the handler in the foreground (registers with RunPod immediately after)
echo "Starting the handler..."
exec python handler.py
