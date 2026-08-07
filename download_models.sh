#!/bin/bash
# Download Remix NSFW I2V + Lightning + NSFW UMT5 assets if missing.
# Safe to re-run; skips files that already exist (Network Volume friendly).
#
# MODELS_ROOT defaults:
#   1) $MODELS_ROOT if set
#   2) /runpod-volume/models if that dir exists or parent volume is mounted
#   3) /ComfyUI/models otherwise
set -euo pipefail

if [ -z "${MODELS_ROOT:-}" ]; then
  if [ -d /runpod-volume ]; then
    MODELS_ROOT=/runpod-volume/models
  else
    MODELS_ROOT=/ComfyUI/models
  fi
fi

mkdir -p \
  "$MODELS_ROOT/diffusion_models" \
  "$MODELS_ROOT/loras" \
  "$MODELS_ROOT/text_encoders" \
  "$MODELS_ROOT/vae" \
  "$MODELS_ROOT/clip_vision"

download() {
  local url="$1"
  local dest="$2"
  if [ -f "$dest" ] && [ -s "$dest" ]; then
    echo "[models] skip (exists): $dest"
    return 0
  fi
  echo "[models] downloading: $dest"
  local tmp="${dest}.part"
  rm -f "$tmp"
  wget -q --show-progress --progress=dot:giga -O "$tmp" "$url"
  mv -f "$tmp" "$dest"
  echo "[models] done: $dest"
}

echo "[models] ensuring required checkpoints under $MODELS_ROOT ..."

download \
  "https://huggingface.co/FX-FeiHou/wan2.2-Remix/resolve/main/NSFW/Wan2.2_Remix_NSFW_i2v_14b_high_lighting_fp8_e4m3fn_v3.0.safetensors" \
  "$MODELS_ROOT/diffusion_models/Wan2.2_Remix_NSFW_i2v_14b_high_lighting_fp8_e4m3fn_v3.0.safetensors"

download \
  "https://huggingface.co/FX-FeiHou/wan2.2-Remix/resolve/main/NSFW/Wan2.2_Remix_NSFW_i2v_14b_low_lighting_fp8_e4m3fn_v3.0.safetensors" \
  "$MODELS_ROOT/diffusion_models/Wan2.2_Remix_NSFW_i2v_14b_low_lighting_fp8_e4m3fn_v3.0.safetensors"

download \
  "https://huggingface.co/lightx2v/Wan2.2-Lightning/resolve/main/Wan2.2-I2V-A14B-4steps-lora-rank64-Seko-V1/high_noise_model.safetensors" \
  "$MODELS_ROOT/loras/high_noise_model.safetensors"

download \
  "https://huggingface.co/lightx2v/Wan2.2-Lightning/resolve/main/Wan2.2-I2V-A14B-4steps-lora-rank64-Seko-V1/low_noise_model.safetensors" \
  "$MODELS_ROOT/loras/low_noise_model.safetensors"

download \
  "https://huggingface.co/NSFW-API/NSFW-Wan-UMT5-XXL/resolve/main/nsfw_wan_umt5-xxl_fp8_scaled.safetensors" \
  "$MODELS_ROOT/text_encoders/nsfw_wan_umt5-xxl_fp8_scaled.safetensors"

download \
  "https://huggingface.co/Comfy-Org/Wan_2.1_ComfyUI_repackaged/resolve/main/split_files/clip_vision/clip_vision_h.safetensors" \
  "$MODELS_ROOT/clip_vision/clip_vision_h.safetensors"

download \
  "https://huggingface.co/Kijai/WanVideo_comfy/resolve/main/Wan2_1_VAE_bf16.safetensors" \
  "$MODELS_ROOT/vae/Wan2_1_VAE_bf16.safetensors"

echo "[models] all required assets present under $MODELS_ROOT"
