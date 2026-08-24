import runpod
from runpod.serverless.utils import rp_upload
import os
import websocket
import base64
import json
import uuid
import logging
import urllib.request
import urllib.parse
import binascii  # for Base64 decode errors
import subprocess
import time
import shutil
from workflow_options import (
    DEFAULT_HIGH_LORA_STRENGTH,
    DEFAULT_LOW_LORA_STRENGTH,
    DEFAULT_SAGE_ATTENTION,
    bypass_torch_compile,
    configure_lightx2v_strengths,
    configure_sage_attention,
    configure_model_retention,
    configure_sampling,
    ensure_rife_required_inputs,
    get_cfg,
    get_keep_models_loaded,
    get_lora_strength,
    get_steps,
    resolve_seed,
)
# Logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


server_address = os.getenv('SERVER_ADDRESS', '127.0.0.1')
client_id = str(uuid.uuid4())
DEFAULT_COMFY_INPUT_DIR = "/ComfyUI/input"


def get_comfy_input_dir():
    return os.getenv("COMFY_INPUT_DIR", DEFAULT_COMFY_INPUT_DIR)


def stage_image_for_loadimage(source_path, dest_filename):
    """Copy an image into ComfyUI's input dir and return the LoadImage basename."""
    if not source_path:
        raise Exception("image source path is empty")
    if not os.path.isfile(source_path):
        raise Exception(f"Image file does not exist: {source_path}")

    dest_filename = os.path.basename(dest_filename)
    if not dest_filename:
        raise Exception("LoadImage destination filename is empty")

    input_dir = get_comfy_input_dir()
    os.makedirs(input_dir, exist_ok=True)
    dest_path = os.path.join(input_dir, dest_filename)
    if os.path.abspath(source_path) != os.path.abspath(dest_path):
        shutil.copy2(source_path, dest_path)

    logger.info(f"Staged image for LoadImage: {source_path} -> {dest_path}")
    return dest_filename


def unique_loadimage_name(task_id, source_path, role):
    ext = os.path.splitext(source_path)[1] or ".png"
    return f"{task_id}_{role}{ext}"


def to_nearest_multiple_of_16(value):
    """Round to the nearest multiple of 16, with a minimum of 16."""
    try:
        numeric_value = float(value)
    except Exception:
        raise Exception(f"width/height is not a number: {value}")
    adjusted = int(round(numeric_value / 16.0) * 16)
    if adjusted < 16:
        adjusted = 16
    return adjusted


def process_input(input_data, temp_dir, output_filename, input_type):
    """Turn path/url/base64 input into a local file path."""
    if input_type == "path":
        logger.info(f"Using path input: {input_data}")
        return input_data
    elif input_type == "url":
        logger.info(f"Downloading URL input: {input_data}")
        os.makedirs(temp_dir, exist_ok=True)
        file_path = os.path.abspath(os.path.join(temp_dir, output_filename))
        return download_file_from_url(input_data, file_path)
    elif input_type == "base64":
        logger.info("Decoding Base64 input")
        return save_base64_to_file(input_data, temp_dir, output_filename)
    else:
        raise Exception(f"Unsupported input type: {input_type}")

        
def download_file_from_url(url, output_path):
    """Download a file from a URL with wget."""
    try:
        result = subprocess.run([
            'wget', '-O', output_path, '--no-verbose', url
        ], capture_output=True, text=True)
        
        if result.returncode == 0:
            logger.info(f"Downloaded URL to file: {url} -> {output_path}")
            return output_path
        else:
            logger.error(f"wget download failed: {result.stderr}")
            raise Exception(f"URL download failed: {result.stderr}")
    except subprocess.TimeoutExpired:
        logger.error("Download timed out")
        raise Exception("Download timed out")
    except Exception as e:
        logger.error(f"Download error: {e}")
        raise Exception(f"Download error: {e}")


def save_base64_to_file(base64_data, temp_dir, output_filename):
    """Decode Base64 data and write it to a file."""
    try:
        decoded_data = base64.b64decode(base64_data)
        os.makedirs(temp_dir, exist_ok=True)
        file_path = os.path.abspath(os.path.join(temp_dir, output_filename))
        with open(file_path, 'wb') as f:
            f.write(decoded_data)
        
        logger.info(f"Wrote Base64 input to '{file_path}'")
        return file_path
    except (binascii.Error, ValueError) as e:
        logger.error(f"Base64 decode failed: {e}")
        raise Exception(f"Base64 decode failed: {e}")
    
def queue_prompt(prompt):
    url = f"http://{server_address}:8188/prompt"
    logger.info(f"Queueing prompt to: {url}")
    p = {"prompt": prompt, "client_id": client_id}
    data = json.dumps(p).encode('utf-8')
    req = urllib.request.Request(url, data=data)
    return json.loads(urllib.request.urlopen(req).read())

def get_image(filename, subfolder, folder_type):
    url = f"http://{server_address}:8188/view"
    logger.info(f"Getting image from: {url}")
    data = {"filename": filename, "subfolder": subfolder, "type": folder_type}
    url_values = urllib.parse.urlencode(data)
    with urllib.request.urlopen(f"{url}?{url_values}") as response:
        return response.read()

def get_history(prompt_id):
    url = f"http://{server_address}:8188/history/{prompt_id}"
    logger.info(f"Getting history from: {url}")
    with urllib.request.urlopen(url) as response:
        return json.loads(response.read())

def _comfy_execution_error_message(payload):
    if not isinstance(payload, dict):
        return None
    parts = [
        payload.get("exception_type"),
        payload.get("node_type") or payload.get("node_id"),
        payload.get("exception_message"),
    ]
    text = ": ".join(str(part) for part in parts if part)
    return text or None


def get_videos(ws, prompt):
    prompt_id = queue_prompt(prompt)['prompt_id']
    logger.info("Comfy prompt queued: prompt_id=%s", prompt_id)
    output_videos = {}
    execution_error = None
    while True:
        out = ws.recv()
        if not isinstance(out, str):
            continue
        message = json.loads(out)
        msg_type = message.get("type")
        data = message.get("data") or {}
        if data.get("prompt_id") not in (None, prompt_id):
            continue
        if msg_type == "execution_error":
            execution_error = _comfy_execution_error_message(data) or "ComfyUI execution_error"
            logger.error("Comfy execution_error: %s", execution_error)
            break
        if msg_type == "executing" and data.get("node") is None:
            logger.info("Comfy prompt finished executing: prompt_id=%s", prompt_id)
            break

    history = get_history(prompt_id).get(prompt_id) or {}
    status = (history.get("status") or {}).get("status_str")
    outputs = history.get("outputs") or {}
    output_keys = {
        node_id: sorted(node_output.keys())
        for node_id, node_output in outputs.items()
        if isinstance(node_output, dict)
    }
    logger.info(
        "Comfy history: prompt_id=%s status=%s output_nodes=%s",
        prompt_id,
        status,
        output_keys,
    )
    if execution_error is None:
        execution_error = _comfy_execution_error_message(history.get("status") or {})

    for node_id, node_output in outputs.items():
        videos_output = []
        if isinstance(node_output, dict) and "gifs" in node_output:
            for video in node_output["gifs"]:
                with open(video["fullpath"], "rb") as f:
                    videos_output.append(base64.b64encode(f.read()).decode("utf-8"))
                logger.info(
                    "Found Comfy video output: node=%s path=%s",
                    node_id,
                    video.get("fullpath"),
                )
        output_videos[node_id] = videos_output

    if execution_error and not any(output_videos.values()):
        raise Exception(execution_error)

    return output_videos

def load_workflow(workflow_path):
    """Load a workflow JSON file."""
    if not os.path.isabs(workflow_path):
        current_dir = os.path.dirname(os.path.abspath(__file__))
        workflow_path = os.path.join(current_dir, workflow_path)
    with open(workflow_path, 'r', encoding='utf-8') as file:
        return json.load(file)

def get_next_available_node_id(prompt, start_id=1000):
    """Return the next unused node ID starting from start_id."""
    node_id = start_id
    while str(node_id) in prompt:
        node_id += 1
    return str(node_id)

def count_user_loras(lora_pairs):
    """Count user LoRA pairs, excluding lightx2v_4steps_lora."""
    if not lora_pairs:
        return 0
    
    count = 0
    for lora_pair in lora_pairs:
        high = lora_pair.get("high", "")
        low = lora_pair.get("low", "")
        
        # Skip baked LightX2V 4-step LoRAs
        if high and "lightx2v_4steps_lora" not in high:
            count += 1
        elif low and "lightx2v_4steps_lora" not in low:
            count += 1
        elif high and low and "lightx2v_4steps_lora" not in high and "lightx2v_4steps_lora" not in low:
            count += 1
    
    return count

def filter_user_loras(lora_pairs):
    """Return user LoRA pairs with lightx2v_4steps_lora entries removed."""
    if not lora_pairs:
        return []
    
    filtered = []
    for lora_pair in lora_pairs:
        high = lora_pair.get("high", "")
        low = lora_pair.get("low", "")
        
        if high and "lightx2v_4steps_lora" in high:
            continue
        if low and "lightx2v_4steps_lora" in low:
            continue
        
        filtered.append(lora_pair)
    
    return filtered

def apply_loras_to_workflow(prompt, lora_pairs, is_flf2v, workflow_file):
    """
    Apply user LoRA names and strengths to the pre-wired LoRA nodes.

    Each workflow JSON already contains the LoRA chain; this only updates
    lora_name and strength_model on those nodes.
    """
    if not lora_pairs:
        return
    
    # User LoRA node IDs per workflow file (HIGH, then LOW)
    # HIGH: UNETLoader(230) -> lightx2v(283) -> user LoRAs -> TorchCompile(391)
    # LOW: UNETLoader(235) -> lightx2v(284) -> user LoRAs -> TorchCompile(390)
    lora_node_mapping = {
        "workflow/wan22_nolora.json": {
            "high": [],
            "low": []
        },
        "workflow/wan22_1lora.json": {
            "high": ["282"],  # first user LoRA after lightx2v(283)
            "low": ["336"]   # first user LoRA after lightx2v(284)
        },
        "workflow/wan22_2lora.json": {
            "high": ["282", "339"],  # lightx2v(283) -> 282 -> 339
            "low": ["336", "285"]    # lightx2v(284) -> 336 -> 285
        },
        "workflow/wan22_3lora.json": {
            "high": ["282", "339", "340"],  # lightx2v(283) -> 282 -> 339 -> 340
            "low": ["336", "285", "286"]    # lightx2v(284) -> 336 -> 285 -> 286
        },
        "workflow/wan22_4lora.json": {
            "high": ["282", "339", "340", "341"],  # lightx2v(283) -> 282 -> 339 -> 340 -> 341
            "low": ["336", "285", "286", "337"]    # lightx2v(284) -> 336 -> 285 -> 286 -> 337
        },
        "workflow/wan22_flf2v.json": {
            "high": [],
            "low": []
        }
    }
    
    workflow_key = None
    for key in lora_node_mapping.keys():
        if key in workflow_file:
            workflow_key = key
            break
    
    if workflow_key is None:
        logger.warning(f"No LoRA node mapping for workflow file {workflow_file}")
        return
    
    high_user_nodes = lora_node_mapping[workflow_key]["high"]
    low_user_nodes = lora_node_mapping[workflow_key]["low"]
    
    logger.info(f"Workflow: {workflow_key}")
    logger.info(f"HIGH user LoRA nodes: {high_user_nodes}")
    logger.info(f"LOW user LoRA nodes: {low_user_nodes}")
    
    if len(high_user_nodes) < len(lora_pairs) or len(low_user_nodes) < len(lora_pairs):
        logger.warning(
            f"Not enough user LoRA nodes in the workflow. "
            f"needed HIGH={len(lora_pairs)}, LOW={len(lora_pairs)}, "
            f"found HIGH={len(high_user_nodes)}, LOW={len(low_user_nodes)}"
        )
        return
    
    for i, lora_pair in enumerate(lora_pairs):
        if i < len(high_user_nodes) and lora_pair.get("high"):
            high_node_id = high_user_nodes[i]
            prompt[high_node_id]["inputs"]["lora_name"] = lora_pair["high"]
            prompt[high_node_id]["inputs"]["strength_model"] = lora_pair.get("high_weight", 1.0)
            logger.info(
                f"Applied HIGH LoRA {i+1}: {lora_pair['high']} "
                f"(strength: {lora_pair.get('high_weight', 1.0)}) -> node {high_node_id}"
            )
        
        if i < len(low_user_nodes) and lora_pair.get("low"):
            low_node_id = low_user_nodes[i]
            prompt[low_node_id]["inputs"]["lora_name"] = lora_pair["low"]
            prompt[low_node_id]["inputs"]["strength_model"] = lora_pair.get("low_weight", 1.0)
            logger.info(
                f"Applied LOW LoRA {i+1}: {lora_pair['low']} "
                f"(strength: {lora_pair.get('low_weight', 1.0)}) -> node {low_node_id}"
            )

def handler(job):
    job_input = job.get("input", {})
    keep_models_loaded = get_keep_models_loaded(job_input)

    logger.info(f"Received job input: {job_input}")
    task_id = f"task_{uuid.uuid4()}"

    # Resolve start image (image, image_path, image_url, or image_base64)
    image_path = None
    if "image" in job_input:
        image_data = job_input["image"]
        if isinstance(image_data, str):
            if image_data.startswith("http://") or image_data.startswith("https://"):
                image_path = process_input(image_data, task_id, "input_image.jpg", "url")
            elif os.path.exists(image_data) or image_data.startswith("/"):
                image_path = process_input(image_data, task_id, "input_image.jpg", "path")
            else:
                image_path = process_input(image_data, task_id, "input_image.jpg", "base64")
        else:
            raise Exception("image must be a string")
    elif "image_path" in job_input:
        image_path = process_input(job_input["image_path"], task_id, "input_image.jpg", "path")
    elif "image_url" in job_input:
        image_path = process_input(job_input["image_url"], task_id, "input_image.jpg", "url")
    elif "image_base64" in job_input:
        image_path = process_input(job_input["image_base64"], task_id, "input_image.jpg", "base64")
    else:
        image_path = "/example_image.png"
        logger.info("Using default image file: /example_image.png")

    # Resolve optional end image (enables FLF2V)
    end_image_path_local = None
    if "end_image" in job_input:
        end_image_data = job_input["end_image"]
        if isinstance(end_image_data, str):
            if end_image_data.startswith("http://") or end_image_data.startswith("https://"):
                end_image_path_local = process_input(end_image_data, task_id, "end_image.jpg", "url")
            elif os.path.exists(end_image_data) or end_image_data.startswith("/"):
                end_image_path_local = process_input(end_image_data, task_id, "end_image.jpg", "path")
            else:
                end_image_path_local = process_input(end_image_data, task_id, "end_image.jpg", "base64")
        else:
            raise Exception("end_image must be a string")
    elif "end_image_path" in job_input:
        end_image_path_local = process_input(job_input["end_image_path"], task_id, "end_image.jpg", "path")
    elif "end_image_url" in job_input:
        end_image_path_local = process_input(job_input["end_image_url"], task_id, "end_image.jpg", "url")
    elif "end_image_base64" in job_input:
        end_image_path_local = process_input(job_input["end_image_base64"], task_id, "end_image.jpg", "base64")
    
    is_flf2v = end_image_path_local is not None
    
    lora_pairs = job_input.get("lora_pairs", [])
    user_lora_pairs = filter_user_loras(lora_pairs)
    lora_count = count_user_loras(lora_pairs)
    
    logger.info(f"User LoRA count (excluding lightx2v): {lora_count}")
    
    if is_flf2v:
        workflow_file = "workflow/wan22_flf2v.json"
        logger.info(f"Using FLF2V workflow: {workflow_file}")
    else:
        if lora_count == 0:
            workflow_file = "workflow/wan22_nolora.json"
        elif lora_count == 1:
            workflow_file = "workflow/wan22_1lora.json"
        elif lora_count == 2:
            workflow_file = "workflow/wan22_2lora.json"
        elif lora_count == 3:
            workflow_file = "workflow/wan22_3lora.json"
        elif lora_count >= 4:
            workflow_file = "workflow/wan22_4lora.json"
            if lora_count > 4:
                logger.warning(
                    f"LoRA count is {lora_count}. Only the first 4 pairs are supported."
                )
                user_lora_pairs = user_lora_pairs[:4]
        else:
            workflow_file = "workflow/wan22_nolora.json"
        
        logger.info(f"Using single image workflow: {workflow_file} (LoRA count: {lora_count})")
    
    prompt = load_workflow(workflow_file)
    configured_nodes = configure_model_retention(prompt, keep_models_loaded)
    rife_nodes = ensure_rife_required_inputs(prompt)
    compile_nodes = bypass_torch_compile(prompt)
    sage_nodes = configure_sage_attention(prompt, DEFAULT_SAGE_ATTENTION)
    seed = resolve_seed(job_input.get("seed", -1))
    steps = get_steps(job_input)
    cfg = get_cfg(job_input)
    split = configure_sampling(prompt, seed, steps, cfg)
    high_lora_strength = get_lora_strength(
        job_input, "high_lora_strength", DEFAULT_HIGH_LORA_STRENGTH
    )
    low_lora_strength = get_lora_strength(
        job_input, "low_lora_strength", DEFAULT_LOW_LORA_STRENGTH
    )
    configure_lightx2v_strengths(prompt, high_lora_strength, low_lora_strength)
    logger.info(
        "Model retention configured: keep_models_loaded=%s, "
        "unload_all_models=%s, nodes=%s",
        keep_models_loaded,
        not keep_models_loaded,
        configured_nodes,
    )
    logger.info("RIFE required inputs configured: nodes=%s", rife_nodes)
    logger.info("Torch compile bypassed: nodes=%s", compile_nodes)
    logger.info("Sage attention configured: mode=%s nodes=%s", DEFAULT_SAGE_ATTENTION, sage_nodes)
    logger.info(
        "Sampling configured: seed=%s steps=%s split=%s cfg=%s",
        seed,
        steps,
        split,
        cfg,
    )
    logger.info(
        "LightX2V strengths configured: high=%s low=%s",
        high_lora_strength,
        low_lora_strength,
    )

    image_name = stage_image_for_loadimage(
        image_path,
        unique_loadimage_name(task_id, image_path, "input_image"),
    )
    end_image_name = None
    if end_image_path_local:
        end_image_name = stage_image_for_loadimage(
            end_image_path_local,
            unique_loadimage_name(task_id, end_image_path_local, "end_image"),
        )
    
    length = job_input.get("length", 81)
    
    original_width = job_input.get("width", 480)
    original_height = job_input.get("height", 720)
    adjusted_width = to_nearest_multiple_of_16(original_width)
    adjusted_height = to_nearest_multiple_of_16(original_height)
    if adjusted_width != original_width:
        logger.info(f"Width adjusted to nearest multiple of 16: {original_width} -> {adjusted_width}")
    if adjusted_height != original_height:
        logger.info(f"Height adjusted to nearest multiple of 16: {original_height} -> {adjusted_height}")

    # Shared widgets for FLF2V and single-image workflows
    prompt["260"]["inputs"]["image"] = image_name
    prompt["246"]["inputs"]["value"] = job_input.get("prompt", "")
    negative_prompt = job_input.get("negative_prompt", "bright tones, overexposed, static, blurred details, subtitles, style, works, paintings, images, static, overall gray, worst quality, low quality, JPEG compression residue, ugly, incomplete, extra fingers, poorly drawn hands, poorly drawn faces, deformed, disfigured, misshapen limbs, fused fingers, still picture, messy background, three legs, many people in the background, walking backwards")
    prompt["247"]["inputs"]["value"] = negative_prompt
    prompt["849"]["inputs"]["value"] = adjusted_width
    prompt["848"]["inputs"]["value"] = adjusted_height
    prompt["846"]["inputs"]["value"] = length
    
    if is_flf2v:
        prompt["483"]["inputs"]["image"] = end_image_name
    
    if user_lora_pairs:
        apply_loras_to_workflow(prompt, user_lora_pairs, is_flf2v, workflow_file)

    ws_url = f"ws://{server_address}:8188/ws?clientId={client_id}"
    logger.info(f"Connecting to WebSocket: {ws_url}")
    
    http_url = f"http://{server_address}:8188/"
    logger.info(f"Checking HTTP connection to: {http_url}")
    
    max_http_attempts = 180
    for http_attempt in range(max_http_attempts):
        try:
            import urllib.request
            response = urllib.request.urlopen(http_url, timeout=5)
            logger.info(f"HTTP connected (attempt {http_attempt+1})")
            break
        except Exception as e:
            logger.warning(f"HTTP connect failed (attempt {http_attempt+1}/{max_http_attempts}): {e}")
            if http_attempt == max_http_attempts - 1:
                raise Exception("Cannot connect to the ComfyUI server. Is it running?")
            time.sleep(1)
    
    ws = websocket.WebSocket()
    max_attempts = int(180/5)  # 3 minutes, retry every 5 seconds
    for attempt in range(max_attempts):
        import time
        try:
            ws.connect(ws_url)
            logger.info(f"WebSocket connected (attempt {attempt+1})")
            break
        except Exception as e:
            logger.warning(f"WebSocket connect failed (attempt {attempt+1}/{max_attempts}): {e}")
            if attempt == max_attempts - 1:
                raise Exception("WebSocket connection timed out (3 minutes)")
            time.sleep(5)
    videos = get_videos(ws, prompt)
    ws.close()

    for node_id in videos:
        if videos[node_id]:
            logger.info("Returning video from node %s (%s bytes base64)", node_id, len(videos[node_id][0]))
            return {"video": videos[node_id][0]}

    logger.error("No video in Comfy history outputs: %s", {k: len(v) for k, v in videos.items()})
    return {"error": "No video was found."}

runpod.serverless.start({"handler": handler})
