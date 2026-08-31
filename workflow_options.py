"""Per-request workflow options for the RunPod handler."""

import os
import random

RIFE_REQUIRED_DEFAULTS = {
    "dtype": "float32",
    "torch_compile": False,
    "batch_size": 1,
}

DEFAULT_SEED = -1
DEFAULT_STEPS = 4
DEFAULT_CFG = 1.0
DEFAULT_HIGH_LORA_STRENGTH = 0.4
DEFAULT_LOW_LORA_STRENGTH = 1.0
DEFAULT_SAGE_ATTENTION = "auto"
MAX_SEED = 2**32 - 1
DEFAULT_MIN_FREE_VRAM_MB = 4096
DEFAULT_MAX_VRAM_USED_RATIO = 0.85
DEFAULT_MIN_FREE_RAM_MB = 2048
DEFAULT_MAX_RAM_USED_RATIO = 0.90
MIB = 1024 * 1024

LIGHTX2V_NODES = (
    ("283", "high_lora_strength"),
    ("284", "low_lora_strength"),
)

SAMPLING_NODES = (
    ("835", "RandomNoise", "noise_seed"),
    ("834", "BetaSamplingScheduler", "steps"),
    ("829", "SplitSigmas", "step"),
    ("830", "ScheduledCFGGuidance", "cfg"),
)


def get_keep_models_loaded(job_input):
    """Return the validated model-retention option from a RunPod input object."""
    keep_models_loaded = job_input.get("keep_models_loaded", False)
    if not isinstance(keep_models_loaded, bool):
        raise ValueError("keep_models_loaded must be a JSON boolean")
    return keep_models_loaded


def configure_model_retention(prompt, keep_models_loaded):
    """Toggle forced end-of-job model unloading on every VRAM Debug node."""
    if not isinstance(keep_models_loaded, bool):
        raise ValueError("keep_models_loaded must be a boolean")

    configured_nodes = 0
    for node in prompt.values():
        if node.get("class_type") != "VRAM_Debug":
            continue

        node.setdefault("inputs", {})["unload_all_models"] = not keep_models_loaded
        configured_nodes += 1

    if configured_nodes == 0:
        raise ValueError("workflow does not contain a VRAM_Debug node")

    return configured_nodes


def _env_int(name, default):
    raw = os.getenv(name)
    if raw is None or str(raw).strip() == "":
        return default
    try:
        return int(raw)
    except (TypeError, ValueError):
        return default


def _env_ratio(name, default):
    raw = os.getenv(name)
    if raw is None or str(raw).strip() == "":
        return default
    try:
        value = float(raw)
    except (TypeError, ValueError):
        return default
    if value < 0 or value > 1:
        return default
    return value


def get_retention_thresholds():
    """Return memory-pressure cutoffs used to override keep_models_loaded."""
    min_free_vram_mb = max(0, _env_int("MODEL_KEEP_MIN_FREE_VRAM_MB", DEFAULT_MIN_FREE_VRAM_MB))
    min_free_ram_mb = max(0, _env_int("MODEL_KEEP_MIN_FREE_RAM_MB", DEFAULT_MIN_FREE_RAM_MB))
    return {
        "min_free_vram_bytes": min_free_vram_mb * MIB,
        "max_vram_used_ratio": _env_ratio(
            "MODEL_KEEP_MAX_VRAM_USED_RATIO", DEFAULT_MAX_VRAM_USED_RATIO
        ),
        "min_free_ram_bytes": min_free_ram_mb * MIB,
        "max_ram_used_ratio": _env_ratio(
            "MODEL_KEEP_MAX_RAM_USED_RATIO", DEFAULT_MAX_RAM_USED_RATIO
        ),
    }


def _non_negative_int(value):
    if isinstance(value, bool):
        return None
    if isinstance(value, str):
        value = value.strip()
        if value == "":
            return None
        try:
            value = int(value)
        except ValueError:
            try:
                value = float(value)
            except ValueError:
                return None
    if not isinstance(value, (int, float)):
        return None
    if value < 0:
        return None
    return int(value)


def parse_comfy_system_stats(payload):
    """Extract RAM/VRAM bytes from a ComfyUI /system_stats payload."""
    stats = {}
    if not isinstance(payload, dict):
        return stats

    system = payload.get("system") or {}
    ram_total = _non_negative_int(system.get("ram_total"))
    ram_free = _non_negative_int(system.get("ram_free"))
    if ram_total is not None:
        stats["ram_total"] = ram_total
    if ram_free is not None:
        stats["ram_free"] = ram_free

    devices = payload.get("devices") or []
    selected = None
    for device in devices:
        if not isinstance(device, dict):
            continue
        if str(device.get("type", "")).lower() == "cuda":
            selected = device
            break
        if selected is None:
            selected = device
    if isinstance(selected, dict):
        vram_total = _non_negative_int(selected.get("vram_total"))
        vram_free = _non_negative_int(selected.get("vram_free"))
        if vram_total is not None:
            stats["vram_total"] = vram_total
        if vram_free is not None:
            stats["vram_free"] = vram_free
    return stats


def parse_nvidia_smi_csv(text):
    """Parse `nvidia-smi --query-gpu=memory.total,memory.free` MiB CSV."""
    if not text:
        return {}
    first_line = text.strip().splitlines()[0]
    parts = [part.strip() for part in first_line.split(",")]
    if len(parts) < 2:
        return {}
    try:
        total_mib = _non_negative_int(float(parts[0]))
        free_mib = _non_negative_int(float(parts[1]))
    except (TypeError, ValueError):
        return {}
    stats = {}
    if total_mib is not None:
        stats["vram_total"] = total_mib * MIB
    if free_mib is not None:
        stats["vram_free"] = free_mib * MIB
    return stats


def parse_meminfo(text):
    """Parse /proc/meminfo into ram_total / ram_free bytes using MemAvailable."""
    values = {}
    for line in text.splitlines():
        if ":" not in line:
            continue
        key, rest = line.split(":", 1)
        tokens = rest.split()
        if not tokens:
            continue
        amount = _non_negative_int(tokens[0])
        if amount is None:
            continue
        values[key.strip()] = amount * 1024
    stats = {}
    if "MemTotal" in values:
        stats["ram_total"] = values["MemTotal"]
    if "MemAvailable" in values:
        stats["ram_free"] = values["MemAvailable"]
    elif "MemFree" in values:
        stats["ram_free"] = values["MemFree"]
    return stats


def parse_cgroup_memory(current_text, max_text):
    """Parse cgroup current/max bytes. Ignore unlimited (`max`) cgroups."""
    if max_text is None or current_text is None:
        return {}
    max_text = str(max_text).strip()
    if max_text in ("", "max"):
        return {}
    ram_total = _non_negative_int(max_text)
    ram_used = _non_negative_int(str(current_text).strip())
    if ram_total is None or ram_used is None:
        return {}
    if ram_total >= (1 << 62):
        return {}
    return {
        "ram_total": ram_total,
        "ram_free": max(ram_total - ram_used, 0),
    }


def format_memory_stats(stats):
    formatted = {}
    for key in ("vram_free", "vram_total", "ram_free", "ram_total"):
        value = stats.get(key) if stats else None
        if isinstance(value, int):
            formatted[key] = f"{value / MIB:.0f}MiB"
    return formatted


def _pressure_for_pool(free_bytes, total_bytes, min_free_bytes, max_used_ratio, label):
    reasons = []
    if not isinstance(free_bytes, int) or not isinstance(total_bytes, int) or total_bytes <= 0:
        return reasons
    used_ratio = 1.0 - (free_bytes / total_bytes)
    if free_bytes < min_free_bytes:
        reasons.append(
            f"{label} free {free_bytes / MIB:.0f}MiB < {min_free_bytes / MIB:.0f}MiB"
        )
    if used_ratio >= max_used_ratio:
        reasons.append(
            f"{label} used {used_ratio:.0%} >= {max_used_ratio:.0%}"
        )
    return reasons


def memory_pressure_reasons(stats, thresholds=None):
    """Return human-readable reasons when VRAM or RAM is too tight to keep models."""
    stats = stats or {}
    thresholds = thresholds or get_retention_thresholds()
    reasons = []
    reasons.extend(
        _pressure_for_pool(
            stats.get("vram_free"),
            stats.get("vram_total"),
            thresholds["min_free_vram_bytes"],
            thresholds["max_vram_used_ratio"],
            "VRAM",
        )
    )
    reasons.extend(
        _pressure_for_pool(
            stats.get("ram_free"),
            stats.get("ram_total"),
            thresholds["min_free_ram_bytes"],
            thresholds["max_ram_used_ratio"],
            "RAM",
        )
    )
    return reasons


def resolve_keep_models_loaded(requested, stats, thresholds=None):
    """Honor keep_models_loaded only when there is enough free VRAM/RAM."""
    if not requested:
        return False, memory_pressure_reasons(stats, thresholds)
    reasons = memory_pressure_reasons(stats, thresholds)
    if reasons:
        return False, reasons
    return True, []


def ensure_rife_required_inputs(prompt):
    """Fill RIFE VFI widgets required by current ComfyUI-Frame-Interpolation."""
    configured_nodes = 0
    for node in prompt.values():
        if node.get("class_type") != "RIFE VFI":
            continue

        inputs = node.setdefault("inputs", {})
        for key, default in RIFE_REQUIRED_DEFAULTS.items():
            inputs.setdefault(key, default)
        configured_nodes += 1

    return configured_nodes


def split_steps(steps):
    return steps // 2


def resolve_seed(value, rng=None):
    """Return a concrete RandomNoise seed. -1 or omitted randomizes."""
    if value is None or value == DEFAULT_SEED:
        rng = rng or random.Random()
        return rng.randint(0, MAX_SEED)
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError("seed must be an integer")
    if value < DEFAULT_SEED:
        raise ValueError("seed must be -1 (random) or >= 0")
    return value


def get_steps(job_input):
    steps = job_input.get("steps", DEFAULT_STEPS)
    if isinstance(steps, bool) or not isinstance(steps, int):
        raise ValueError("steps must be an integer")
    if steps < 2:
        raise ValueError("steps must be >= 2")
    return steps


def get_cfg(job_input):
    cfg = job_input.get("cfg", DEFAULT_CFG)
    if isinstance(cfg, bool) or not isinstance(cfg, (int, float)):
        raise ValueError("cfg must be a number")
    if cfg < 0:
        raise ValueError("cfg must be >= 0")
    return float(cfg)


def configure_sampling(prompt, seed, steps, cfg):
    """Write seed/steps/CFG into the dual-pass ksampler graph."""
    split = split_steps(steps)
    values = {
        "835": seed,
        "834": steps,
        "829": split,
        "830": cfg,
    }
    for node_id, class_type, field in SAMPLING_NODES:
        node = prompt.get(node_id)
        if not isinstance(node, dict) or node.get("class_type") != class_type:
            raise ValueError(f"workflow is missing {class_type} node {node_id}")
        node.setdefault("inputs", {})[field] = values[node_id]
    return split


def get_lora_strength(job_input, key, default):
    strength = job_input.get(key, default)
    if isinstance(strength, bool) or not isinstance(strength, (int, float)):
        raise ValueError(f"{key} must be a number")
    return float(strength)


def configure_lightx2v_strengths(prompt, high_lora_strength, low_lora_strength):
    """Write baked LightX2V LoRA strengths on the high/low experts."""
    values = {
        "283": high_lora_strength,
        "284": low_lora_strength,
    }
    for node_id, _key in LIGHTX2V_NODES:
        node = prompt.get(node_id)
        if not isinstance(node, dict) or node.get("class_type") != "LoraLoaderModelOnly":
            raise ValueError(f"workflow is missing LightX2V LoraLoaderModelOnly node {node_id}")
        node.setdefault("inputs", {})["strength_model"] = values[node_id]


def _is_link_to(value, node_id):
    return (
        isinstance(value, list)
        and len(value) >= 1
        and str(value[0]) == str(node_id)
    )


def configure_sage_attention(prompt, mode=DEFAULT_SAGE_ATTENTION):
    """Set PathchSageAttentionKJ to a GPU-safe mode. auto picks Ampere vs Ada kernels."""
    if not isinstance(mode, str) or not mode:
        raise ValueError("sage_attention mode must be a non-empty string")

    configured_nodes = 0
    for node in prompt.values():
        if not isinstance(node, dict) or node.get("class_type") != "PathchSageAttentionKJ":
            continue
        node.setdefault("inputs", {})["sage_attention"] = mode
        configured_nodes += 1

    if configured_nodes == 0:
        raise ValueError("workflow does not contain a PathchSageAttentionKJ node")

    return configured_nodes


def bypass_torch_compile(prompt):
    """Point compile consumers at the compile node's model input so inductor never runs."""
    compile_nodes = {
        node_id: node
        for node_id, node in prompt.items()
        if isinstance(node, dict)
        and node.get("class_type") == "TorchCompileModelWanVideoV2"
    }
    if not compile_nodes:
        raise ValueError("workflow does not contain a TorchCompileModelWanVideoV2 node")

    for compile_id, compile_node in compile_nodes.items():
        upstream = (compile_node.get("inputs") or {}).get("model")
        if not isinstance(upstream, list) or not upstream:
            raise ValueError(f"TorchCompile node {compile_id} has no model input")
        replacement = [upstream[0], upstream[1] if len(upstream) > 1 else 0]
        for node_id, node in prompt.items():
            if node_id in compile_nodes or not isinstance(node, dict):
                continue
            inputs = node.setdefault("inputs", {})
            for key, value in list(inputs.items()):
                if _is_link_to(value, compile_id):
                    inputs[key] = list(replacement)

    return len(compile_nodes)


