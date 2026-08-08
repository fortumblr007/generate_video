# Quality backlog — I2V looks bad

Status: **OPEN** (2026-08-08). Pipeline completes; visual quality is unacceptable
on the 720×912 UI-config runs.

## Bad runs to re-compare later

| File | Seed | cfg | steps | frames | res |
|------|------|-----|-------|--------|-----|
| `i2v_out_720x912.mp4` | 2016777954 | 7 | 8 | 97 | 720×912 |
| `i2v_out_720x912_r2.mp4` | 1071044493 | 7 | 8 | 97 | 720×912 |

“OK” baseline for plumbing (not quality target):

| File | Seed | cfg | steps | frames | res |
|------|------|-----|-------|--------|-----|
| `i2v_out_sage_t5.mp4` | 42 | 1 | 4 | 49 | 480×832 |

## Hypotheses (ordered)

1. **CFG 7 + Lightning LoRA** — Lightning is tuned for ~cfg 1 / 4 steps. High CFG may wreck the schedule.
2. **Steps 8 with Lightning** — dual HIGH/LOW split becomes 4+4; may not match Lightning training.
3. **CLIP bridge text path** — different from `WanVideoTextEncode`; possible prompt adherence / motion issues.
4. **Long context (97f)** + FreeNoise / context windows — morphing, flicker, identity drift.
5. **Resolution / aspect** vs source image — warp if forced 720×912 without matching crop.
6. **Negative prompt bloat** — fight positive motion prompt.

## Experiments to run (checklist)

- [ ] A1: same image/prompt, **cfg=1, steps=4**, length=97, 720×912
- [ ] A2: same, **cfg=1, steps=4**, length=49, 720×912
- [ ] A3: same, **cfg=1, steps=4**, length=49, 480×832 (compare to sage_t5 baseline)
- [ ] B1: cfg=7, steps=8 **with Lightning strength 0 / disable lora_0** if workflow allows
- [ ] C1: shorten negative to 1 line
- [ ] D1: length=81 or 65 (common Wan lengths)
- [ ] E1: A/B text path: scaled T5 via bridge vs Kijai non-scaled + WanVideoTextEncode (quality-only)
- [ ] F1: log per-step previews / first-last frame stills for drift diagnosis

## Do not regress

- Must keep **sageattn** and **nsfw_wan_umt5-xxl_fp8_scaled.safetensors** unless product changes.
- Comfy must stay on **venv-bw** after Sage install.
