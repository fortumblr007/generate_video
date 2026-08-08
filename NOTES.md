# I2V pod notes (serverless-adv / generate_video)

Working notes from native RunPod GPU pod bring-up, SageAttention, NSFW scaled T5,
and first 720×912 quality runs. **Output quality is currently bad** — pipeline runs
end-to-end, but sampling/defaults need a dedicated quality pass later.

Last updated: 2026-08-08  
Commit that landed the core wiring: `7d1ca10` (plus follow-ups).

---

## Product intent

- **I2V only** (no FLF2V / end-frame).
- Stack: **Remix NSFW Wan 2.2 I2V 14B** + **Lightning LoRAs** + **NSFW UMT5** (`nsfw_wan_umt5-xxl_fp8_scaled.safetensors`).
- **Non-negotiables (user):**
  1. **SageAttention** (`attention_mode: sageattn`)
  2. **Original T5 file** `nsfw_wan_umt5-xxl_fp8_scaled.safetensors` (not Kijai non-scaled substitute)

---

## Architecture (current proven path)

### Pod layout

| Path | Role |
|------|------|
| `/workspace/ComfyUI` | ComfyUI + custom nodes |
| `/ComfyUI` | symlink → `/workspace/ComfyUI` |
| `/workspace/models/{diffusion_models,loras,text_encoders,vae,clip_vision}` | Weights (network/workspace disk) |
| `/ComfyUI/models/*` | Symlinks into `/workspace/models/*` |
| `/workspace/venv-bw` | **Python env for Comfy + Sage** (torch 2.11+cu128) |
| `/workspace/generate_video` | App (handler, workflow, runners) |
| `/new_Wan22_api.json` | Workflow copy used by handler default path |

### Must-use Python

```bash
# ComfyUI MUST use venv-bw or Sage is invisible
/workspace/venv-bw/bin/python -u /workspace/ComfyUI/main.py --listen 127.0.0.1 --port 8188
```

System `/usr/bin/python3.12` had torch 2.8 and **no** sageattention — that caused
`Selected attention mode not available` until Comfy was restarted under `venv-bw`.

### Models (expected)

From `models_urls.txt` / `download_models.sh` / `pod_aria2_dl.sh` (sequential aria2 `-x8 -s8 -j1`):

- Diffusion high/low: Remix NSFW I2V fp8 lighting v3.0
- Lightning LoRAs: `high_noise_model.safetensors`, `low_noise_model.safetensors`
- T5: **`nsfw_wan_umt5-xxl_fp8_scaled.safetensors`**
- VAE, clip_vision as in URL list

Optional extra that was tried then **rejected for production path**:
- `umt5-xxl-enc-fp8_e4m3fn.safetensors` (Kijai) — works with `LoadWanVideoT5TextEncoder`,
  but **not** the required original NSFW scaled file.

---

## Workflow (`new_Wan22_api.json`) — what changed

### T5 (scaled NSFW)

`LoadWanVideoT5TextEncoder` **hard-rejects** any state dict containing key `scaled_fp8`:

```text
Invalid T5 text encoder model, fp8 scaled is not supported by this node
```

Kijai’s guidance: use native Comfy text encode + **bridge**.

| Node id | Class | Purpose |
|---------|--------|---------|
| **136** | `CLIPLoader` | `clip_name=nsfw_wan_umt5-xxl_fp8_scaled.safetensors`, `type=wan` |
| **137** | `CLIPTextEncode` | Positive prompt |
| **138** | `CLIPTextEncode` | Negative prompt |
| **135** | `WanVideoTextEmbedBridge` | Native CONDITIONING → `WANVIDEOTEXTEMBEDS` |

Samplers **220 / 540** still consume text embeds from node **135**.

### Attention

- `attention_mode`: **`sageattn`** on both WanVideo model loaders (HIGH + LOW).
- Requires package `sageattention` importable in **the same interpreter as Comfy**.

### Other workflow defaults (still Lightning-oriented)

- 4-step Lightning style split nodes (`569` steps / `575` split).
- Handler sets `split_step = max(1, steps // 2)`.
- Lightning stays on `lora_0` (high/low noise models).

---

## Handler (`handler.py`) — what changed

1. **Stage image** into `COMFY_INPUT_DIR` (default `/ComfyUI/input`) as `{task_id}_input.jpg`
   and pass **relative** name to `LoadImage` (absolute paths fail).
2. **Surface Comfy execution errors** from history (`execution_error`) instead of vague
   “video not found”.
3. **HTTP `/prompt` errors**: log body on 400/500.
4. **Video outputs**: read `gifs` / `videos` / `video`; resolve relative paths under
   `/ComfyUI/output` and `/workspace/ComfyUI/output`.
5. **Prompt injection**:
   - Prefer CLIP nodes **137/138** when present.
   - Fallback legacy `WanVideoTextEncode` node **135** `positive_prompt` / `negative_prompt`.

Env knobs:

| Env | Default | Meaning |
|-----|---------|---------|
| `WORKFLOW_FILE` | `/new_Wan22_api.json` if present else repo relative | API workflow JSON |
| `SERVER_ADDRESS` | `127.0.0.1` | Comfy host |
| `COMFY_INPUT_DIR` | `/ComfyUI/input` | Image stage dir |
| `MODELS_ROOT` | (volume logic) | Weights root |
| `SKIP_MODEL_DOWNLOAD` | unset | Set `1` on pod when weights already present |

---

## SageAttention install

Script: `install_sageattention.sh`

- Prebuilt wheel (Blackwell **sm120**, CUDA 12.8, cp312):  
  `sageattention-2.2.0+cu128torch2.10.0sm120` from Hugging Face `yo9otatara/prebuilt_wheels`
- Verified on pod: **import + CUDA forward** with torch **2.11.0+cu128**
- Triton was already present in venv (`3.6.0`)
- After install: **restart Comfy under venv-bw**

Smoke:

```bash
/workspace/venv-bw/bin/python -c "from sageattention import sageattn; print('ok')"
```

---

## How to run I2V on the pod

### Prerequisites

```bash
# Comfy up
curl -s -o /dev/null -w "%{http_code}\n" http://127.0.0.1:8188/

# Models present
ls -lh /workspace/models/text_encoders/nsfw_wan_umt5-xxl_fp8_scaled.safetensors
ls /workspace/models/diffusion_models/ | head
ls /workspace/models/loras/
```

### One-shot runner

```bash
cd /workspace/generate_video
export MODELS_ROOT=/workspace/models
export SKIP_MODEL_DOWNLOAD=1
export WORKFLOW_FILE=/new_Wan22_api.json
export SERVER_ADDRESS=127.0.0.1
export COMFY_INPUT_DIR=/ComfyUI/input
export I2V_OUT=/workspace/i2v_out.mp4
export I2V_PROMPT='...'
export I2V_IMAGE_URL='...'
export I2V_LENGTH=49
/workspace/venv-bw/bin/python -u run_i2v_once.py
```

`run_i2v_once.py` defaults are **Lightning-friendly** (cfg 1, steps 4, 480×832).
For UI-style jobs, pass full dict via a small inline Python (see run log below) —
handler accepts `negative_prompt`, `cfg`, `width`, `height`, `length`, `steps`, `seed`,
`context_overlap`.

### Seed −1

Handler does `int(seed)` only — **no random for −1**. Callers must expand:

```python
seed = random.randint(0, 2**31 - 1) if seed_in < 0 else int(seed_in)
```

**TODO:** bake seed −1 → random into `handler.py`.

---

## Run log (quality currently bad)

### Smoke / wiring proof (OK pipeline, small res)

| Output | Res | Frames | Steps | CFG | Seed | Notes |
|--------|-----|--------|-------|-----|------|-------|
| `i2v_out.mp4` | 480×832 | 49 | 4 | 1 | 42 | First success after T5/sage fixes |
| `i2v_out_sage_t5.mp4` | 480×832 | 49 | 4 | 1 | 42 | Original T5 + sageattn path |

~15.7 GB peak sampling VRAM at 480×832 / 49f / 4 steps. Sampling ~5.5 s/step with sage.

### User job 720×912 (bad quality — keep for later debug)

| Output | Res | Frames | Steps | CFG | Seed | Wall time |
|--------|-----|--------|-------|-----|------|-----------|
| `i2v_out_720x912.mp4` | 720×912 | 97 | 8 | **7** | 2016777954 | ~7 min (Comfy ~406 s) |
| `i2v_out_720x912_r2.mp4` | 720×912 | 97 | 8 | **7** | 1071044493 | ~6 min (357 s) |

Inputs (720 run):

- Image: `https://preview.redd.it/amazing-bikini-fit-v0-jcfjg5oaduhh1.jpeg?auto=webp&s=7a3bac23df6dfe464c64d31a27de0a90b6146fc1`
- Prompt: `the woman slowly removes her clothes revealing her massive bare boobs.`
- Negative: long morphing/deform list from UI screenshot (`Screenshot 2026-08-08 012709.png`)
- context_overlap: 48

Observed during 720 run:

- Context window seq len **64125**, section size 25, FreeNoise on
- High-noise phase: **~51 s/step** at 720×912 / 97f
- Peak VRAM **~22 GB**, util 100%
- Prompt executed ~356–406 s

**Videos are gitignored** (local only). Re-run from pod paths under `/workspace/` if still present.

---

## Known failures (fixed or still open)

| Issue | Status | Fix / note |
|-------|--------|------------|
| Scaled T5 via `LoadWanVideoT5TextEncoder` | **Fixed (path change)** | CLIPLoader + bridge |
| Missing `sageattention` | **Fixed** | Install wheel + Comfy on `venv-bw` |
| Sage installed but Comfy on system Python | **Fixed** | Restart Comfy with venv python |
| LoadImage absolute path | **Fixed** | Stage to `COMFY_INPUT_DIR` |
| Vague “video not found” | **Fixed** | History `execution_error` surfacing |
| Port 8000 not open on pod | Context | Use 8080/8180 for smoke; Comfy **8188** local |
| Huge Docker model bake | Deferred | Network volume + thin app image |
| **Output quality bad at cfg 7 / steps 8** | **OPEN** | See backlog |
| Seed −1 not random in handler | **OPEN** | Expand in handler |
| FantasyTalking / AdaptiveWindowSize import noise | Non-fatal | Can ignore or remove nodes |

---

## Quality fix backlog (priority)

1. **Lightning vs CFG/steps mismatch**  
   Lightning LoRAs are typically **cfg ≈ 1, steps ≈ 4**. UI used **cfg 7, steps 8**.
   Try:
   - A: cfg 1, steps 4 (Lightning default)
   - B: cfg 7, steps 8 **without** Lightning LoRAs (if “full” schedule desired)
   - C: cfg 1–2, steps 6–8 with Lightning still on

2. **Text encode path**  
   Bridge + native CLIP may differ from Wan’s own T5 encode (token length, padding,
   offload). Compare embeds / A-B vs Kijai non-scaled + `WanVideoTextEncode` for
   quality only (product still wants scaled file).

3. **Context / FreeNoise at 97 frames**  
   Logs: context schedule 25 frames, overlap 12 (workflow), user `context_overlap=48`.
   Audit nodes `498` / windowing for long clips — may cause morphing / instability.

4. **Resolution**  
   720×912 is heavier; confirm aspect matches source image crop to reduce warp.

5. **CFG schedule node `570`**  
   Handler forces start/end CFG to job cfg. Verify interaction with dual-sampler
   high/low Lightning split.

6. **Negative prompt length**  
   Very long negatives can dilute or fight Lightning; test short vs full list.

7. **Seed handling**  
   Implement `seed < 0 → random` in handler; log resolved seed in response meta.

8. **Serverless packaging**  
   - Bake Sage into runtime/app image or install at cold start from wheel URL  
   - Seed Network Volume with models (no 24/90GB image pushes)  
   - Ensure worker starts Comfy with the venv that has Sage

---

## Scripts inventory

| File | Purpose |
|------|---------|
| `handler.py` | RunPod serverless job handler |
| `new_Wan22_api.json` | Comfy API workflow |
| `run_i2v_once.py` | Pod one-shot I2V |
| `install_sageattention.sh` | Sage wheel + CUDA smoke |
| `pod_i2v_setup.sh` | Native Comfy I2V setup on pod |
| `pod_aria2_dl.sh` | Sequential 8-conn model downloads |
| `pod_cleanup_disk.sh` | Disk recovery helpers |
| `download_models.sh` / `models_urls.txt` | Model URL list + fetch |
| `seed_volume.sh` | Network volume seed helper |
| `comfy_setup_remote.sh` | Remote Comfy + nodes + model symlinks |
| `inspect_history.py` | Dump Comfy `/history/{id}` + output files |
| `smoke_server.py` / `smoke_test.py` | Local HTTP smoke without full models |
| `Dockerfile*` / `entrypoint.sh` | Container split (runtime + app) |

---

## Pod SSH (session snapshot — may change)

- Host/port were RunPod TCP SSH (example: `root@… -p … -i ~/.ssh/id_ed25519`)
- Open ports mentioned: **8080 / 8180** (not 8000); Comfy listens **127.0.0.1:8188**
- GPU used: **NVIDIA RTX PRO 4500 Blackwell** (sm_120), ~32 GB

Always re-check SSH endpoint from RunPod UI when the pod is recreated.

---

## Git hygiene

- `.gitignore` excludes `*.mp4`, logs, `__pycache__`, smoke stderr/out, job id files.
- Do **not** commit multi‑GB weights or generated videos.
- Quality experiments: log seed / cfg / steps / frames in commit messages or this file.

---

## Quick “is the stack healthy?” checklist

```bash
# 1) Comfy on venv + HTTP
ps -eo pid,cmd | grep 'main.py'   # should show /workspace/venv-bw/bin/python
curl -s -o /dev/null -w "%{http_code}\n" http://127.0.0.1:8188/

# 2) Sage
/workspace/venv-bw/bin/python -c "from sageattention import sageattn; print('sage_ok')"

# 3) T5 file name in workflow
grep -n nsfw_wan_umt5 /new_Wan22_api.json /workspace/generate_video/new_Wan22_api.json

# 4) attention_mode
grep -n attention_mode /new_Wan22_api.json
```

---

## Bottom line

- **Pipeline proven:** scaled NSFW T5 + SageAttention + Lightning dual sampler → MP4.
- **Product quality not proven:** 720×912 jobs with UI cfg/steps look **super bad**;
  treat cfg/steps/Lightning interaction and long-context settings as the next fix focus.
- Keep using this notes file when changing sampling defaults so we don’t re-lose the
  pod/Python/Sage/T5 constraints.
