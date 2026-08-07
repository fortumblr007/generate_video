#!/bin/bash
# RunPod Serverless entrypoint (volume-first).
set -e

# ---------------------------------------------------------------------------
# Network Volume (RunPod mounts at /runpod-volume)
# Models live on the volume so image pushes stay small (runtime rarely changes;
# app image is only handler/workflow).
# ---------------------------------------------------------------------------
VOLUME_ROOT="${NETWORK_VOLUME_PATH:-/runpod-volume}"

if [ -d "$VOLUME_ROOT" ]; then
  export MODELS_ROOT="${MODELS_ROOT:-$VOLUME_ROOT/models}"
  echo "[volume] detected $VOLUME_ROOT → MODELS_ROOT=$MODELS_ROOT"
  mkdir -p \
    "$MODELS_ROOT/diffusion_models" \
    "$MODELS_ROOT/loras" \
    "$MODELS_ROOT/text_encoders" \
    "$MODELS_ROOT/vae" \
    "$MODELS_ROOT/clip_vision" \
    "$VOLUME_ROOT/loras"

  # Point ComfyUI default model dirs at the volume (works even if a node
  # ignores extra_model_paths.yaml).
  for d in diffusion_models loras text_encoders vae clip_vision; do
    target="$MODELS_ROOT/$d"
    link="/ComfyUI/models/$d"
    mkdir -p "$target"
    rm -rf "$link"
    ln -sfn "$target" "$link"
  done
else
  export MODELS_ROOT="${MODELS_ROOT:-/ComfyUI/models}"
  echo "[volume] no $VOLUME_ROOT; using MODELS_ROOT=$MODELS_ROOT"
  mkdir -p \
    "$MODELS_ROOT/diffusion_models" \
    "$MODELS_ROOT/loras" \
    "$MODELS_ROOT/text_encoders" \
    "$MODELS_ROOT/vae" \
    "$MODELS_ROOT/clip_vision"
fi

# Do NOT download multi-GB models here (blocks RunPod registration).
# Handler calls download_models.sh on first real job unless SKIP_MODEL_DOWNLOAD=1.
# Optional pre-download only if you accept long start times:
if [ "${DOWNLOAD_MODELS_ON_START:-0}" = "1" ] && [ "${SKIP_MODEL_DOWNLOAD:-0}" != "1" ]; then
  echo "DOWNLOAD_MODELS_ON_START=1: pre-downloading weights into $MODELS_ROOT ..."
  MODELS_ROOT="$MODELS_ROOT" /download_models.sh
fi

echo "Starting ComfyUI in the background..."
if python -c "import sageattention" 2>/dev/null; then
  python /ComfyUI/main.py --listen --use-sage-attention &
else
  echo "sageattention not found; starting ComfyUI without --use-sage-attention"
  python /ComfyUI/main.py --listen &
fi

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

echo "Starting the handler..."
exec python /handler.py
