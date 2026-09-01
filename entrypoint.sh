#!/bin/bash

# Start ComfyUI in the background, then register the RunPod worker immediately.
# Hub's "prepare AI API" deadline expires if handler.py has not called
# runpod.serverless.start() yet; waiting for Comfy here causes that timeout.
set -e

echo "Starting ComfyUI in the background..."
python /ComfyUI/main.py --listen --use-sage-attention &

echo "Starting the handler..."
exec python handler.py