# Wan2.2 Remix NSFW I2V (Lightning) — RunPod Serverless

Image-to-video worker using:

- **FX-FeiHou Wan2.2 Remix NSFW I2V v3.0** (HIGH + LOW fp8)
- **NSFW UMT5** text encoder (fp8 scaled)
- **LightX2V Lightning** 4-step I2V LoRAs
- **ComfyUI + Kijai WanVideoWrapper**
- RunPod Serverless handler

**I2V only** — first/last-frame (FLF2V) is not supported.

Fork of [wlsdml1114/generate_video](https://github.com/wlsdml1114/generate_video) with model and recipe updates for Remix NSFW + Lightning.

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

## Build & push

**Slim (recommended for Docker Hub):** ~24 GB image; weights download on first worker start (or mount a Network Volume).

```bash
# Base (once)
docker build -f base.Dockerfile -t fortumblr007/video-gen-base:1.0 .
docker push fortumblr007/video-gen-base:1.0

# App without baked weights (pushes cleanly from a laptop)
docker build --build-arg BAKE_MODELS=0 -t fortumblr007/generate-video-nsfw-i2v:1.0-slim .
docker push fortumblr007/generate-video-nsfw-i2v:1.0-slim
```

**Full bake** (all HF weights in the image, ~90 GB; needs lots of RAM/disk and a stable upload):

```bash
docker build --build-arg BAKE_MODELS=1 -t fortumblr007/generate-video-nsfw-i2v:1.0 .
docker push fortumblr007/generate-video-nsfw-i2v:1.0
```

Published tags (this fork):

| Tag | Notes |
|-----|--------|
| `fortumblr007/generate-video-nsfw-i2v:1.0-slim` | ComfyUI + nodes; models via `download_models.sh` at start |
| `fortumblr007/generate-video-nsfw-i2v:slim` | Same digest as `1.0-slim` |
| `fortumblr007/video-gen-base:1.0` | CUDA 12.8 + PyTorch cu128 runtime base |

### RunPod Serverless endpoint

| Setting | Value |
|---------|--------|
| Container image | `fortumblr007/generate-video-nsfw-i2v:1.0-slim` |
| GPU | 24 GB+ VRAM (Ada / Hopper / Blackwell all fine with this base) |
| Container disk | ≥ **80 GB** for slim (models download into the container), or ≥ **20 GB** if models live on a Network Volume |
| Max workers / idle | First cold start downloads ~45 GB of weights — raise **execution / idle** timeouts accordingly |
| Network volume | **Recommended:** pre-seed `/ComfyUI/models` (or mount over it) so workers skip HuggingFace downloads. Extra user LoRAs can still go under `/loras` |
| Env (optional) | `SKIP_MODEL_DOWNLOAD=1` if you fully manage weights yourself |

Base image: `fortumblr007/video-gen-base:1.0` (not the upstream Blackwell-only image).

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
