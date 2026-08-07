#!/bin/bash
# Native I2V setup on a RunPod GPU Pod (no Docker).
# Installs ComfyUI + nodes + downloads models, starts ComfyUI :8188 and smoke :8080.
set -e
export DEBIAN_FRONTEND=noninteractive
LOG=/tmp/i2v_setup.log
exec > >(tee -a "$LOG") 2>&1
echo "===== SETUP START $(date -Is) ====="

cd /workspace
if [ ! -d /workspace/venv-bw ]; then
  python3 -m venv /workspace/venv-bw
fi
# shellcheck disable=SC1091
source /workspace/venv-bw/bin/activate
pip install -q -U pip
if ! python -c "import torch; assert torch.cuda.is_available()" 2>/dev/null; then
  pip install -q torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128
fi
python -c "import torch; print('torch', torch.__version__, 'cuda', torch.cuda.is_available(), torch.cuda.get_device_name(0))"

if [ ! -d /workspace/ComfyUI ]; then
  git clone --depth 1 https://github.com/comfyanonymous/ComfyUI.git /workspace/ComfyUI
fi
cd /workspace/ComfyUI
pip install -q -r requirements.txt
pip install -q websocket-client requests

cd /workspace/ComfyUI/custom_nodes
clone() { [ -d "$1" ] || git clone --depth 1 "$2" "$1"; }
clone ComfyUI-Manager https://github.com/Comfy-Org/ComfyUI-Manager.git
clone ComfyUI-GGUF https://github.com/city96/ComfyUI-GGUF
clone ComfyUI-KJNodes https://github.com/kijai/ComfyUI-KJNodes
clone ComfyUI-VideoHelperSuite https://github.com/Kosinkadink/ComfyUI-VideoHelperSuite
clone ComfyUI-GGUF-FantasyTalking https://github.com/kael558/ComfyUI-GGUF-FantasyTalking
clone ComfyUI-wanBlockswap https://github.com/orssorbit/ComfyUI-wanBlockswap
clone ComfyUI-WanVideoWrapper https://github.com/kijai/ComfyUI-WanVideoWrapper
clone IntelligentVRAMNode https://github.com/eddyhhlure1Eddy/IntelligentVRAMNode
clone auto_wan2.2animate_freamtowindow_server https://github.com/eddyhhlure1Eddy/auto_wan2.2animate_freamtowindow_server
clone ComfyUI-AdaptiveWindowSize https://github.com/eddyhhlure1Eddy/ComfyUI-AdaptiveWindowSize
if [ -d ComfyUI-AdaptiveWindowSize/ComfyUI-AdaptiveWindowSize ]; then
  cp -n ComfyUI-AdaptiveWindowSize/ComfyUI-AdaptiveWindowSize/* ComfyUI-AdaptiveWindowSize/ 2>/dev/null || true
fi
for d in ComfyUI-Manager ComfyUI-GGUF ComfyUI-KJNodes ComfyUI-VideoHelperSuite ComfyUI-GGUF-FantasyTalking ComfyUI-WanVideoWrapper; do
  [ -f "$d/requirements.txt" ] && pip install -q -r "$d/requirements.txt" || true
done

export MODELS_ROOT=/workspace/models
mkdir -p "$MODELS_ROOT"/{diffusion_models,loras,text_encoders,vae,clip_vision}
ln -sfn /workspace/ComfyUI /ComfyUI
for d in diffusion_models loras text_encoders vae clip_vision; do
  rm -rf "/ComfyUI/models/$d"
  ln -sfn "$MODELS_ROOT/$d" "/ComfyUI/models/$d"
done

dl() {
  local url="$1" dest="$2"
  if [ -f "$dest" ] && [ -s "$dest" ]; then echo "skip $dest"; return 0; fi
  echo "GET $dest"
  wget -c -q --show-progress --progress=dot:giga -O "${dest}.part" "$url"
  mv -f "${dest}.part" "$dest"
}
dl "https://huggingface.co/FX-FeiHou/wan2.2-Remix/resolve/main/NSFW/Wan2.2_Remix_NSFW_i2v_14b_high_lighting_fp8_e4m3fn_v3.0.safetensors" \
  "$MODELS_ROOT/diffusion_models/Wan2.2_Remix_NSFW_i2v_14b_high_lighting_fp8_e4m3fn_v3.0.safetensors"
dl "https://huggingface.co/FX-FeiHou/wan2.2-Remix/resolve/main/NSFW/Wan2.2_Remix_NSFW_i2v_14b_low_lighting_fp8_e4m3fn_v3.0.safetensors" \
  "$MODELS_ROOT/diffusion_models/Wan2.2_Remix_NSFW_i2v_14b_low_lighting_fp8_e4m3fn_v3.0.safetensors"
dl "https://huggingface.co/lightx2v/Wan2.2-Lightning/resolve/main/Wan2.2-I2V-A14B-4steps-lora-rank64-Seko-V1/high_noise_model.safetensors" \
  "$MODELS_ROOT/loras/high_noise_model.safetensors"
dl "https://huggingface.co/lightx2v/Wan2.2-Lightning/resolve/main/Wan2.2-I2V-A14B-4steps-lora-rank64-Seko-V1/low_noise_model.safetensors" \
  "$MODELS_ROOT/loras/low_noise_model.safetensors"
dl "https://huggingface.co/NSFW-API/NSFW-Wan-UMT5-XXL/resolve/main/nsfw_wan_umt5-xxl_fp8_scaled.safetensors" \
  "$MODELS_ROOT/text_encoders/nsfw_wan_umt5-xxl_fp8_scaled.safetensors"
dl "https://huggingface.co/Comfy-Org/Wan_2.1_ComfyUI_repackaged/resolve/main/split_files/clip_vision/clip_vision_h.safetensors" \
  "$MODELS_ROOT/clip_vision/clip_vision_h.safetensors"
dl "https://huggingface.co/Kijai/WanVideo_comfy/resolve/main/Wan2_1_VAE_bf16.safetensors" \
  "$MODELS_ROOT/vae/Wan2_1_VAE_bf16.safetensors"
echo "MODELS_DONE"
du -sh "$MODELS_ROOT"/* || true

REPO=/workspace/generate_video_smoke
if [ ! -d "$REPO" ]; then
  git clone --depth 1 https://github.com/fortumblr007/generate_video.git "$REPO"
fi
cp "$REPO/new_Wan22_api.json" /new_Wan22_api.json
cp "$REPO/handler.py" /handler.py
cat > /ComfyUI/extra_model_paths.yaml <<'YAML'
comfyui:
    base_path: /ComfyUI/
    is_default: true
    diffusion_models: models/diffusion_models
    loras: models/loras
    text_encoders: models/text_encoders
    clip: models/text_encoders
    clip_vision: models/clip_vision
    vae: models/vae
YAML

if ! curl -s --max-time 2 http://127.0.0.1:8188/ >/dev/null; then
  echo "Starting ComfyUI..."
  cd /workspace/ComfyUI
  nohup python main.py --listen 127.0.0.1 --port 8188 > /tmp/comfyui.log 2>&1 &
  echo "COMFY_PID=$!"
  for _ in $(seq 1 90); do
    if curl -s --max-time 2 http://127.0.0.1:8188/ >/dev/null; then echo COMFY_READY; break; fi
    sleep 2
  done
else
  echo COMFY_ALREADY
fi
curl -s --max-time 3 -o /dev/null -w "comfy_http=%{http_code}\n" http://127.0.0.1:8188/ || true
tail -40 /tmp/comfyui.log || true

if command -v fuser >/dev/null; then fuser -k 8080/tcp 2>/dev/null || true; fi
sleep 1
cd "$REPO"
SMOKE_HOST=0.0.0.0 SMOKE_PORT=8080 \
MODELS_ROOT=/workspace/models SKIP_MODEL_DOWNLOAD=1 \
WORKFLOW_FILE=/new_Wan22_api.json SERVER_ADDRESS=127.0.0.1 \
  nohup /workspace/venv-bw/bin/python smoke_server.py > /tmp/smoke_8080.log 2>&1 &
echo "SMOKE_PID=$!"
sleep 2
curl -s http://127.0.0.1:8080/health || true
echo
echo "===== SETUP COMPLETE $(date -Is) ====="
