#!/bin/bash
# Download required models ONE FILE AT A TIME with aria2 multi-connection.
# Default: -x8 -s8 -j1 (8 threads per file, never parallel files).
#
# IMPORTANT: Prefer a Network Volume — container disks fill easily (~45GB+ weights).
#
#   MODELS_ROOT=/runpod-volume/models bash pod_aria2_dl.sh
#   ARIA2_CONNECTIONS=8 bash pod_aria2_dl.sh   # default
#
# Requires: aria2c (installs .deb if missing), curl, python3
set +e

if [ -z "${MODELS_ROOT:-}" ]; then
  if [ -d /runpod-volume ]; then
    MODELS_ROOT=/runpod-volume/models
  else
    MODELS_ROOT=/workspace/models
  fi
fi

echo "MODELS_ROOT=$MODELS_ROOT"
echo "=== disk ==="
df -h "$(dirname "$MODELS_ROOT")" 2>/dev/null || df -h /
avail_kb=$(df -Pk "$(dirname "$MODELS_ROOT")" 2>/dev/null | awk 'NR==2{print $4}')
if [ -n "${avail_kb:-}" ] && [ "$avail_kb" -lt 50000000 ]; then
  echo "WARNING: less than ~50GB free. Model pack needs ~45GB+. Prefer Network Volume."
  echo "Run: bash pod_cleanup_disk.sh"
fi

echo "=== current models ==="
ls -lh "$MODELS_ROOT"/diffusion_models/ 2>/dev/null
du -sh "$MODELS_ROOT"/* 2>/dev/null

pkill -x wget 2>/dev/null
pkill -x apt-get 2>/dev/null

# Install aria2 via .deb (skip full apt update when possible)
if ! command -v aria2c >/dev/null 2>&1; then
  cd /tmp
  curl -fsSL -o aria2.deb \
    "http://archive.ubuntu.com/ubuntu/pool/universe/a/aria2/aria2_1.37.0-1build1_amd64.deb" \
    || curl -fsSL -o aria2.deb \
    "http://archive.ubuntu.com/ubuntu/pool/universe/a/aria2/aria2_1.36.0-1_amd64.deb"
  dpkg -i aria2.deb 2>&1 | tail -15
  apt-get install -y -f -qq 2>&1 | tail -10
fi
command -v aria2c
aria2c --version 2>/dev/null | head -1

mkdir -p "$MODELS_ROOT"/{diffusion_models,loras,text_encoders,vae,clip_vision}
pkill -x aria2c 2>/dev/null
find "$MODELS_ROOT" -name "*.part" -delete 2>/dev/null
find "$MODELS_ROOT" -name "*.aria2" -delete 2>/dev/null

# Build aria2 input from models_urls.txt if present, else embedded list
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
URLS_FILE="${SCRIPT_DIR}/models_urls.txt"

if [ -f "$URLS_FILE" ]; then
  : > /tmp/models_all.aria2
  while IFS=$'\t' read -r rel url || [ -n "${rel:-}" ]; do
    case "$rel" in
      ''|\#*) continue ;;
    esac
    # allow space-separated fallback
    if [ -z "${url:-}" ]; then
      rel_first="${rel%% *}"
      url="${rel#* }"
      rel="$rel_first"
    fi
    dir="$MODELS_ROOT/$(dirname "$rel")"
    out="$(basename "$rel")"
    {
      echo "$url"
      echo "  dir=$dir"
      echo "  out=$out"
      echo
    } >> /tmp/models_all.aria2
  done < "$URLS_FILE"
else
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
fi

python3 - <<'PY'
from pathlib import Path
raw = Path("/tmp/models_all.aria2").read_text().strip()
blocks, cur = [], []
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
  du -sh "$MODELS_ROOT"/*
  exit 0
fi

if ! command -v aria2c >/dev/null 2>&1; then
  echo "ERROR: aria2c not installed"
  exit 1
fi

# One file at a time (-j1); 8 connections per file (-x8 -s8).
# Sequential loop so disk usage and failures are easier to control.
ARIA2_X="${ARIA2_CONNECTIONS:-8}"
ARIA2_J=1
LOG=/tmp/aria2_models.log
: > "$LOG"
echo "Sequential download: one file at a time, aria2 -x${ARIA2_X} -s${ARIA2_X} -j${ARIA2_J}"
echo "MODELS_ROOT=$MODELS_ROOT"
echo "Log: $LOG"

# Parse models_run.aria2 into sequential blocks and download each fully before next
python3 - <<'PY' > /tmp/models_queue.txt
from pathlib import Path
raw = Path("/tmp/models_run.aria2").read_text().strip()
if not raw:
    raise SystemExit(0)
blocks, cur = [], []
for line in raw.splitlines():
    if line.startswith("http") and cur:
        blocks.append("\n".join(cur))
        cur = [line]
    else:
        cur.append(line)
if cur:
    blocks.append("\n".join(cur))
for i, b in enumerate(blocks):
    Path(f"/tmp/model_one_{i}.aria2").write_text(b.strip() + "\n")
    print(i)
PY

n=0
while [ -f "/tmp/model_one_${n}.aria2" ]; do
  url=$(head -1 "/tmp/model_one_${n}.aria2")
  out=$(grep 'out=' "/tmp/model_one_${n}.aria2" | head -1 | sed 's/.*out=//')
  dir=$(grep 'dir=' "/tmp/model_one_${n}.aria2" | head -1 | sed 's/.*dir=//')
  echo ""
  echo "===== [$n] START $(date -Is) → $dir/$out ====="
  df -h "$MODELS_ROOT" 2>/dev/null | tail -1 || df -h / | tail -1
  aria2c -i "/tmp/model_one_${n}.aria2" \
    -x "$ARIA2_X" -s "$ARIA2_X" -j 1 \
    --file-allocation=none --continue=true --max-tries=0 --retry-wait=3 \
    --timeout=120 --connect-timeout=30 --min-split-size=1M \
    --summary-interval=15 --console-log-level=notice \
    2>&1 | tee -a "$LOG"
  rc=${PIPESTATUS[0]}
  if [ "$rc" -ne 0 ]; then
    echo "===== [$n] FAILED rc=$rc $out =====" | tee -a "$LOG"
    echo "Stopping sequential download (disk or network). Fix and re-run; completed files are skipped."
    exit "$rc"
  fi
  echo "===== [$n] DONE $out size=$(stat -c%s "$dir/$out" 2>/dev/null || echo '?') =====" | tee -a "$LOG"
  du -sh "$MODELS_ROOT"/* 2>/dev/null | tee -a "$LOG"
  n=$((n + 1))
done

echo "ALL_COMPLETE"
du -sh "$MODELS_ROOT"/* 2>/dev/null
df -h "$MODELS_ROOT" 2>/dev/null | tail -1 || true
