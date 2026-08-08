# Wan2.2 Remix NSFW I2V (Lightning) — RunPod Serverless

Image-to-video worker using:

- **FX-FeiHou Wan2.2 Remix NSFW I2V v3.0** (HIGH + LOW fp8)
- **NSFW UMT5** text encoder (fp8 scaled) via native **CLIPLoader + WanVideoTextEmbedBridge**
- **LightX2V Lightning** 4-step I2V LoRAs
- **SageAttention** (`attention_mode: sageattn`)
- **ComfyUI + Kijai WanVideoWrapper**
- RunPod Serverless handler

**I2V only** — first/last-frame (FLF2V) is not supported.

Fork of [wlsdml1114/generate_video](https://github.com/wlsdml1114/generate_video) with model and recipe updates for Remix NSFW + Lightning.

### Working notes (pod bring-up + quality)

- **[NOTES.md](NOTES.md)** — full pod layout, T5/Sage constraints, handler/workflow changes, run log, open issues
- **[docs/QUALITY_BACKLOG.md](docs/QUALITY_BACKLOG.md)** — quality is currently bad; experiment checklist
- **Lightning defaults** below (cfg 1 / steps 4) are the intended recipe; UI jobs with **cfg 7 / steps 8** looked poor and need a fix pass

---

## Models baked in the image

| Role | File |
|------|------|
| HIGH DiT | `Wan2.2_Remix_NSFW_i2v_14b_high_lighting_fp8_e4m3fn_v3.0.safetensors` |
| LOW DiT | `Wan2.2_Remix_NSFW_i2v_14b_low_lighting_fp8_e4m3fn_v3.0.safetensors` |
| T5 | `nsfw_wan_umt5-xxl_fp8_scaled.safetensors` |
| Lightning HIGH/LOW | `high_noise_model.safetensors` / `low_noise_model.safetensors` |
| VAE | `Wan2_1_VAE_bf16.safetensors` |
| CLIP Vision | `clip_vision_h.safetensors` |

Sources:

- [FX-FeiHou/wan2.2-Remix](https://huggingface.co/FX-FeiHou/wan2.2-Remix)
- [NSFW-API/NSFW-Wan-UMT5-XXL](https://huggingface.co/NSFW-API/NSFW-Wan-UMT5-XXL)
- [lightx2v/Wan2.2-Lightning](https://huggingface.co/lightx2v/Wan2.2-Lightning) (I2V Seko-V1)

---

## Lightning defaults

| Param | Default |
|-------|---------|
| `steps` | **4** |
| High/Low split | **2 / 2** |
| `cfg` | **1.0** |
| Scheduler | euler |
| `length` | 81 frames |
| `width` / `height` | 480 / 832 (rounded to multiples of 16) |

---

## Architecture (volume-first — avoid big pushes)

| Layer | Image / storage | Size | How often |
|-------|-----------------|------|-----------|
| **Base** | `fortumblr007/video-gen-base:1.0` | ~CUDA+PyTorch | Almost never |
| **Runtime** | `fortumblr007/generate-video-runtime:1.0` | ~24 GB ComfyUI+nodes | Rarely (node upgrades) |
| **App** | `fortumblr007/generate-video-nsfw-i2v:app` | **MBs** (handler, workflow, scripts) | **Every code change** |
| **Models** | **Network Volume** `/runpod-volume/models` | ~45 GB | Seed once |

Day-to-day you only rebuild/push **Dockerfile.app**. Docker Hub already has the runtime layers; push is tiny.

### Build & push

```bash
# 1) Base — once
docker build -f base.Dockerfile -t fortumblr007/video-gen-base:1.0 .
docker push fortumblr007/video-gen-base:1.0

# 2) Runtime (ComfyUI + nodes, no weights) — rare
docker build -f Dockerfile.runtime -t fortumblr007/generate-video-runtime:1.0 .
docker push fortumblr007/generate-video-runtime:1.0

# 3) App (handler/workflow only) — every code change
docker build -f Dockerfile.app -t fortumblr007/generate-video-nsfw-i2v:app .
docker push fortumblr007/generate-video-nsfw-i2v:app
```

### Network Volume (100 GB) — seed once

1. RunPod → **Storage** → **Network Volume** → create **100 GB** in the **same region** as the endpoint.
2. Attach volume to a temporary **GPU or CPU Pod** (mount path `/runpod-volume`).
3. On the Pod:

```bash
git clone https://github.com/fortumblr007/generate_video.git
cd generate_video
bash seed_volume.sh
# writes ~45GB under /runpod-volume/models/...
```

4. Detach from Pod; attach the **same volume** to the **Serverless endpoint**.
5. Optional env: `SKIP_MODEL_DOWNLOAD=1` after the volume is fully seeded.

Layout:

```text
/runpod-volume/models/diffusion_models/   # Remix DiTs
/runpod-volume/models/loras/              # Lightning
/runpod-volume/models/text_encoders/      # NSFW UMT5
/runpod-volume/models/vae/
/runpod-volume/models/clip_vision/
/runpod-volume/loras/                     # optional extra user LoRAs
```

Entrypoint symlinks `/ComfyUI/models/*` → volume dirs and sets `MODELS_ROOT`.

### RunPod Serverless endpoint

| Setting | Value |
|---------|--------|
| Container image | `fortumblr007/generate-video-nsfw-i2v:app` |
| GPU | 24 GB+ VRAM |
| Container disk | **20–40 GB** is enough when models are on the volume |
| Network volume | **100 GB**, same datacenter, mounted (default `/runpod-volume`) |
| Env (optional) | `SKIP_MODEL_DOWNLOAD=1` after seed; `MODELS_ROOT=/runpod-volume/models` |
| Smoke test | `{"input":{"ping":true}}` → should report `volume_mounted: true` |

Do **not** bake weights into the image (~90 GB). Keep weights on the volume.

Published tags:

| Tag | Notes |
|-----|--------|
| `fortumblr007/video-gen-base:1.0` | CUDA 12.8 + PyTorch cu128 |
| `fortumblr007/generate-video-runtime:1.0` | ComfyUI + nodes (push rarely) |
| `fortumblr007/generate-video-nsfw-i2v:app` | Thin app (push often) |
| `…:1.0-slim` | Older all-in-one slim (superseded by runtime+app) |

---

## API input

One of: `image_path` | `image_url` | `image_base64`.

| Parameter | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| `prompt` | string | Yes | — | Motion / scene description |
| `negative_prompt` | string | No | long default | Negative prompt |
| `image_path` / `image_url` / `image_base64` | string | One of | example image | Input image |
| `seed` | int | No | 42 | Seed |
| `cfg` | float | No | **1.0** | CFG (Lightning) |
| `width` | int | No | 480 | Width |
| `height` | int | No | 832 | Height |
| `length` | int | No | 81 | Frames |
| `steps` | int | No | **4** | Denoising steps |
| `context_overlap` | int | No | 48 | Context window overlap |
| `lora_pairs` | array | No | `[]` | Extra LoRA pairs (max 4); Lightning is already on `lora_0` |

**Not supported:** `end_image_*` (FLF2V) — returns an error.

### Example request

```json
{
  "input": {
    "prompt": "a person walking toward the camera, natural motion, cinematic lighting",
    "negative_prompt": "blurry, low quality, distorted, static",
    "image_url": "https://example.com/start.jpg",
    "width": 480,
    "height": 832,
    "length": 81,
    "steps": 4,
    "seed": 42,
    "cfg": 1.0
  }
}
```

### Extra LoRAs (optional)

Upload `.safetensors` to Network Volume `/loras/`, then:

```json
"lora_pairs": [
  {
    "high": "my_style_high.safetensors",
    "low": "my_style_low.safetensors",
    "high_weight": 0.8,
    "low_weight": 0.8
  }
]
```

### Output

Success: `{ "video": "<base64 mp4>" }`  
Error: `{ "error": "..." }`

---

## Python client

```python
from generate_video_client import GenerateVideoClient

client = GenerateVideoClient(
    runpod_endpoint_id="your-endpoint-id",
    runpod_api_key="your-runpod-api-key"
)

result = client.create_video_from_image(
    image_path="./example_image.png",
    prompt="running man, grab the gun",
    negative_prompt="blurry, low quality, distorted",
    width=480,
    height=832,
    length=81,
    steps=4,
    seed=42,
    cfg=1.0,
)

if result.get("status") == "COMPLETED":
    client.save_video_result(result, "./output_video.mp4")
else:
    print(result.get("error") or result)
```

---

## Test on RunPod Serverless

1. Push image and create an endpoint with the settings above.
2. Submit a job:

```bash
curl -s -X POST "https://api.runpod.ai/v2/ENDPOINT_ID/run" \
  -H "Authorization: Bearer RUNPOD_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{
    "input": {
      "prompt": "a person walking slowly toward the camera, natural motion",
      "image_url": "https://YOUR_PUBLIC_IMAGE.jpg",
      "width": 480,
      "height": 832,
      "length": 81,
      "steps": 4,
      "seed": 42,
      "cfg": 1.0
    }
  }'
```

3. Poll: `GET https://api.runpod.ai/v2/ENDPOINT_ID/status/JOB_ID`
4. On `COMPLETED`, decode `output.video` from base64 to `.mp4`.

Cold start can take several minutes (image pull + ComfyUI + model load). Use worker logs if the job fails.

---

## Repo layout

| File | Role |
|------|------|
| `Dockerfile` | Image build + model downloads |
| `entrypoint.sh` | Start ComfyUI, then handler |
| `handler.py` | RunPod serverless handler (I2V only) |
| `new_Wan22_api.json` | ComfyUI API workflow |
| `generate_video_client.py` | Client helper |
| `extra_model_paths.yaml` | Network volume model paths |

---

## Compliance

This worker is **uncensored-capable** (Remix NSFW + NSFW T5). You are responsible for RunPod / registry policies, applicable law, and who can call your endpoint.

---

## Credits

- [Wan2.2](https://github.com/Wan-Video/Wan2.2)
- [ComfyUI](https://github.com/comfyanonymous/ComfyUI)
- [ComfyUI-WanVideoWrapper](https://github.com/kijai/ComfyUI-WanVideoWrapper)
- [lightx2v/Wan2.2-Lightning](https://huggingface.co/lightx2v/Wan2.2-Lightning)
- [FX-FeiHou/wan2.2-Remix](https://huggingface.co/FX-FeiHou/wan2.2-Remix)
- Upstream template: [wlsdml1114/generate_video](https://github.com/wlsdml1114/generate_video)
