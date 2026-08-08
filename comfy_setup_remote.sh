set -e
export DEBIAN_FRONTEND=noninteractive
LOG=/tmp/comfy_i2v_setup.log
exec > >(tee -a $LOG) 2>&1
echo "===== COMFY SETUP $(date -Is) ====="
if [ ! -d /workspace/venv-bw ]; then python3 -m venv /workspace/venv-bw; fi
source /workspace/venv-bw/bin/activate
pip install -q -U pip
python -c "import torch; print(torch.__version__, torch.cuda.is_available())" 2>/dev/null || pip install -q torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128
python -c "import torch; print('torch', torch.__version__, torch.cuda.is_available(), torch.cuda.get_device_name(0)); x=torch.zeros(1,device='cuda'); print('cuda_ok')"
if [ ! -d /workspace/ComfyUI ]; then git clone --depth 1 https://github.com/comfyanonymous/ComfyUI.git /workspace/ComfyUI; fi
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
for d in ComfyUI-Manager ComfyUI-GGUF ComfyUI-KJNodes ComfyUI-VideoHelperSuite ComfyUI-GGUF-FantasyTalking ComfyUI-WanVideoWrapper; do
  [ -f "$d/requirements.txt" ] && pip install -q -r "$d/requirements.txt" || true
done
ln -sfn /workspace/ComfyUI /ComfyUI
export MODELS_ROOT=/workspace/models
for d in diffusion_models loras text_encoders vae clip_vision; do
  rm -rf /ComfyUI/models/$d
  ln -sfn $MODELS_ROOT/$d /ComfyUI/models/$d
done
ls /ComfyUI/models/diffusion_models/
ls /ComfyUI/models/loras/
cd /workspace/generate_video
cp new_Wan22_api.json /new_Wan22_api.json
cp handler.py /handler.py
printf '%s\n' 'comfyui:' '    base_path: /ComfyUI/' '    is_default: true' '    diffusion_models: models/diffusion_models' '    loras: models/loras' '    text_encoders: models/text_encoders' '    clip: models/text_encoders' '    clip_vision: models/clip_vision' '    vae: models/vae' > /ComfyUI/extra_model_paths.yaml
if ! curl -s --max-time 2 http://127.0.0.1:8188/ >/dev/null; then
  cd /workspace/ComfyUI
  nohup python main.py --listen 127.0.0.1 --port 8188 > /tmp/comfyui.log 2>&1 &
  echo COMFY_PID=$!
  for i in $(seq 1 120); do
    curl -s --max-time 2 http://127.0.0.1:8188/ >/dev/null && echo COMFY_READY && break
    sleep 2
  done
else
  echo COMFY_ALREADY
fi
curl -s -o /dev/null -w "comfy_http=%{http_code}\n" http://127.0.0.1:8188/ || true
tail -30 /tmp/comfyui.log || true
if command -v fuser >/dev/null; then fuser -k 8080/tcp 2>/dev/null || true; fi
sleep 1
cd /workspace/generate_video
export MODELS_ROOT=/workspace/models SKIP_MODEL_DOWNLOAD=1 WORKFLOW_FILE=/new_Wan22_api.json SERVER_ADDRESS=127.0.0.1
SMOKE_HOST=0.0.0.0 SMOKE_PORT=8080 nohup /workspace/venv-bw/bin/python smoke_server.py > /tmp/smoke_8080.log 2>&1 &
echo SMOKE_PID=$!
sleep 3
curl -s http://127.0.0.1:8080/health || true
echo
echo "===== SETUP COMPLETE $(date -Is) ====="