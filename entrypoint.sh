#!/bin/bash

# Exit immediately if a command exits with a non-zero status.
set -e

# Ensure model weights exist (no-op if baked into image or mounted via Network Volume).
# Set SKIP_MODEL_DOWNLOAD=1 to skip (e.g. you mount models yourself).
if [ "${SKIP_MODEL_DOWNLOAD:-0}" != "1" ]; then
  echo "Checking / downloading model weights..."
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


# Wait for ComfyUI to be ready
echo "Waiting for ComfyUI to be ready..."
max_wait=120  # 최대 2분 대기
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

# Start the handler in the foreground
# 이 스크립트가 컨테이너의 메인 프로세스가 됩니다.
echo "Starting the handler..."
exec python handler.py