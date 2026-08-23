"""Per-request workflow options for the RunPod handler."""

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
MAX_SEED = 2**32 - 1

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


