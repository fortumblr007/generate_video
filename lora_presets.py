"""Public LoRA presets for single-image Wan 2.2 generation."""

import math


DEFAULT_HIGH_WEIGHT = 0.8
DEFAULT_LOW_WEIGHT = 0.7
MAX_PRESETS = 2

PRESETS = {
    "assume_the_position": {
        "high": "WAN22_assume_the_position_high.safetensors",
        "low": "WAN22_assume_the_position_low.safetensors",
    },
    "airblow": {
        "high": "Airb-high-80.safetensors",
        "low": "Airb-low-70.safetensors",
    },
    "clothes_on_off": {
        "high": "Wan_ClothesOnOff_Trend.safetensors",
    },
    "sudden_outfit_change": {
        "high": "SuddenOutfitChange_V03.safetensors",
    },
    "tittdrop": {
        "high": "t1ttydr0p_high_noise.safetensors",
        "low": "t1ttydr0p_low_noise.safetensors",
    },
}


def resolve_lora_presets(job_input, is_flf2v=False):
    """Validate requested presets and return pairs for the existing workflows."""
    if "lora_pairs" in job_input:
        raise ValueError("lora_pairs is no longer supported; use lora_presets")

    selected = job_input.get("lora_presets", [])
    if not isinstance(selected, list):
        raise ValueError("lora_presets must be an array")
    if is_flf2v and selected:
        raise ValueError("lora_presets are supported only for single-image requests")
    if len(selected) > MAX_PRESETS:
        raise ValueError("lora_presets supports at most two presets")

    resolved = []
    seen = set()
    for item in selected:
        if not isinstance(item, dict):
            raise ValueError("each lora_presets item must be an object")
        if set(item) - {"name", "high_weight", "low_weight"}:
            raise ValueError("lora_presets items allow only name, high_weight, and low_weight")
        name = item.get("name")
        if not isinstance(name, str) or name not in PRESETS:
            raise ValueError(f"unknown LoRA preset: {name!r}")
        if name in seen:
            raise ValueError(f"duplicate LoRA preset: {name}")
        seen.add(name)

        weights = {}
        for key, default in (("high_weight", DEFAULT_HIGH_WEIGHT), ("low_weight", DEFAULT_LOW_WEIGHT)):
            value = item.get(key, default)
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
                raise ValueError(f"{key} for {name} must be a finite number")
            weights[key] = float(value)
        resolved.append({**PRESETS[name], **weights})
    return resolved
