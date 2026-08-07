#!/bin/bash
set +e
echo "=== current models ==="
ls -lh /workspace/models/diffusion_models/ 2>/dev/null
du -sh /workspace/models/* 2>/dev/null

pkill -x wget 2>/dev/null
pkill -x apt-get 2>/dev/null

# Install aria2 via .deb (skip full apt update)
if ! command -v aria2c >/dev/null 2>&1; then
  cd /tmp
  curl -fsSL -o aria2.deb \
    "http://archive.ubuntu.com/ubuntu/pool/universe/a/aria2/aria2_1.37.0-1build1_amd64.deb" \
    || curl -fsSL -o aria2.deb \
    "http://archive.ubuntu.com/ubuntu/pool/universe/a/aria2/aria2_1.36.0-1_amd64.deb"
  dpkg -i aria2.deb 2>&1 | tail -15
  # pull deps if needed
  apt-get install -y -f -qq 2>&1 | tail -10
fi
command -v aria2c
aria2c --version 2>/dev/null | head -1

export MODELS_ROOT=/workspace/models
mkdir -p "$MODELS_ROOT"/{diffusion_models,loras,text_encoders,vae,clip_vision}
pkill -x aria2c 2>/dev/null
find "$MODELS_ROOT" -name "*.part" -delete 2>/dev/null
find "$MODELS_ROOT" -name "*.aria2" -delete 2>/dev/null

cat > /tmp/models_all.aria2 <<EOF
https://huggingface.co/FX-FeiHou/wan2.2-Remix/resolve/main/NSFW/Wan2.2_Remix_NSFW_i2v_14b_high_lighting_fp8_e4m3fn_v3.0.safetensors
  dir=${MODELS_ROOT}/diffusion_models
  out=Wan2.2_Remix_NSFW_i2v_14b_high_lighting_fp8_e4m3fn_v3.0.safetensors
https://huggingface.co/FX-FeiHou/wan2.2-Remix/resolve/main/NSFW/Wan2.2_Remix_NSFW_i2v_14b_low_lighting_fp8_e4m3fn_v3.0.safetensors
  dir=${MODELS_ROOT}/diffusion_models
  out=Wan2.2_Remix_NSFW_i2v_14b_low_lighting_fp8_e4m3fn_v3.0.safetensors
https://huggingface.co/lightx2v/Wan2.2-Lightning/resolve/main/Wan2.2-I2V-A14B-4steps-lora-rank64-Seko-V1/high_noise_model.safetensors
  dir=${MODELS_ROOT}/loras
  out=high_noise_model.safetensors
https://huggingface.co/lightx2v/Wan2.2-Lightning/resolve/main/Wan2.2-I2V-A14B-4steps-lora-rank64-Seko-V1/low_noise_model.safetensors
  dir=${MODELS_ROOT}/loras
  out=low_noise_model.safetensors
https://huggingface.co/NSFW-API/NSFW-Wan-UMT5-XXL/resolve/main/nsfw_wan_umt5-xxl_fp8_scaled.safetensors
  dir=${MODELS_ROOT}/text_encoders
  out=nsfw_wan_umt5-xxl_fp8_scaled.safetensors
https://huggingface.co/Comfy-Org/Wan_2.1_ComfyUI_repackaged/resolve/main/split_files/clip_vision/clip_vision_h.safetensors
  dir=${MODELS_ROOT}/clip_vision
  out=clip_vision_h.safetensors
https://huggingface.co/Kijai/WanVideo_comfy/resolve/main/Wan2_1_VAE_bf16.safetensors
  dir=${MODELS_ROOT}/vae
  out=Wan2_1_VAE_bf16.safetensors
EOF

python3 - <<'PY'
from pathlib import Path
raw = Path("/tmp/models_all.aria2").read_text().strip()
blocks = []
cur = []
for line in raw.splitlines():
    if line.startswith("http") and cur:
        blocks.append("\n".join(cur))
        cur = [line]
    else:
        cur.append(line)
if cur:
    blocks.append("\n".join(cur))
keep = []
for b in blocks:
    lines = b.strip().splitlines()
    meta = {}
    for ln in lines[1:]:
        if "=" in ln:
            k, v = ln.strip().split("=", 1)
            meta[k.strip()] = v.strip()
    dest = Path(meta["dir"]) / meta["out"]
    if dest.is_file() and dest.stat().st_size > 10_000_000:
        print(f"skip complete {dest} ({dest.stat().st_size})")
    else:
        keep.append(b.strip())
Path("/tmp/models_run.aria2").write_text("\n\n".join(keep) + ("\n" if keep else ""))
print(f"todo_files={len(keep)}")
PY

if [ ! -s /tmp/models_run.aria2 ]; then
  echo ALL_COMPLETE
  du -sh /workspace/models/*
  exit 0
fi

if command -v aria2c >/dev/null 2>&1; then
  echo "Starting aria2c -x16 -s16 -j4"
  nohup aria2c -i /tmp/models_run.aria2 -x 16 -s 16 -j 4 \
    --file-allocation=none --continue=true --max-tries=0 --retry-wait=2 \
    --timeout=60 --connect-timeout=30 --min-split-size=1M \
    --summary-interval=10 --console-log-level=notice \
    > /tmp/aria2_models.log 2>&1 &
  echo ARIA2_PID=$!
else
  echo "aria2 missing; parallel curl fallback"
  nohup bash -c '
    R=/workspace/models
    while IFS= read -r url; do
      [ -z "$url" ] && continue
      case "$url" in http*) ;; *) continue ;; esac
    done < /dev/null
    curl -L -C - --http1.1 -o "$R/diffusion_models/Wan2.2_Remix_NSFW_i2v_14b_high_lighting_fp8_e4m3fn_v3.0.safetensors" "https://huggingface.co/FX-FeiHou/wan2.2-Remix/resolve/main/NSFW/Wan2.2_Remix_NSFW_i2v_14b_high_lighting_fp8_e4m3fn_v3.0.safetensors" &
    curl -L -C - --http1.1 -o "$R/diffusion_models/Wan2.2_Remix_NSFW_i2v_14b_low_lighting_fp8_e4m3fn_v3.0.safetensors" "https://huggingface.co/FX-FeiHou/wan2.2-Remix/resolve/main/NSFW/Wan2.2_Remix_NSFW_i2v_14b_low_lighting_fp8_e4m3fn_v3.0.safetensors" &
    curl -L -C - --http1.1 -o "$R/loras/high_noise_model.safetensors" "https://huggingface.co/lightx2v/Wan2.2-Lightning/resolve/main/Wan2.2-I2V-A14B-4steps-lora-rank64-Seko-V1/high_noise_model.safetensors" &
    curl -L -C - --http1.1 -o "$R/loras/low_noise_model.safetensors" "https://huggingface.co/lightx2v/Wan2.2-Lightning/resolve/main/Wan2.2-I2V-A14B-4steps-lora-rank64-Seko-V1/low_noise_model.safetensors" &
    curl -L -C - --http1.1 -o "$R/text_encoders/nsfw_wan_umt5-xxl_fp8_scaled.safetensors" "https://huggingface.co/NSFW-API/NSFW-Wan-UMT5-XXL/resolve/main/nsfw_wan_umt5-xxl_fp8_scaled.safetensors" &
    curl -L -C - --http1.1 -o "$R/clip_vision/clip_vision_h.safetensors" "https://huggingface.co/Comfy-Org/Wan_2.1_ComfyUI_repackaged/resolve/main/split_files/clip_vision/clip_vision_h.safetensors" &
    curl -L -C - --http1.1 -o "$R/vae/Wan2_1_VAE_bf16.safetensors" "https://huggingface.co/Kijai/WanVideo_comfy/resolve/main/Wan2_1_VAE_bf16.safetensors" &
    wait
    echo CURL_DONE
  ' > /tmp/curl_parallel.log 2>&1 &
  echo CURL_PID=$!
fi

sleep 6
ps -C aria2c -o pid,etime,cmd 2>/dev/null | head -5
ps -C curl -o pid,etime 2>/dev/null | head -10
echo "--- aria2 log ---"
tail -20 /tmp/aria2_models.log 2>/dev/null
echo "--- sizes ---"
du -sh /workspace/models/* 2>/dev/null
echo LAUNCHED_OK
