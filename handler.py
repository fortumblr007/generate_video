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
import mimetypes
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
    format_memory_stats,
    get_cfg,
    get_keep_models_loaded,
    get_lora_strength,
    get_steps,
    parse_cgroup_memory,
    parse_comfy_system_stats,
    parse_meminfo,
    parse_nvidia_smi_csv,
    resolve_keep_models_loaded,
    resolve_seed,
)
# Logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


server_address = os.getenv('SERVER_ADDRESS', '127.0.0.1')
client_id = str(uuid.uuid4())
DEFAULT_COMFY_INPUT_DIR = "/ComfyUI/input"
DEFAULT_COMFY_OUTPUT_DIR = "/ComfyUI/output"
DEFAULT_COMFY_TEMP_DIR = "/ComfyUI/temp"
CATBOX_API_URL = "https://catbox.moe/user/api.php"
CATBOX_UPLOAD_ATTEMPTS = 2
CATBOX_UPLOAD_TIMEOUT_SECONDS = 30
CATBOX_RETRY_DELAY_SECONDS = 2
VHS_VIDEO_COMBINE_NODE_ID = "277"


def get_comfy_input_dir():
    return os.getenv("COMFY_INPUT_DIR", DEFAULT_COMFY_INPUT_DIR)


def get_comfy_output_dir():
    return os.getenv("COMFY_OUTPUT_DIR", DEFAULT_COMFY_OUTPUT_DIR)


def get_comfy_temp_dir():
    return os.getenv("COMFY_TEMP_DIR", DEFAULT_COMFY_TEMP_DIR)


def _remove_path(path):
    if not path or not os.path.exists(path):
        return
    try:
        if os.path.isdir(path):
            shutil.rmtree(path)
        else:
            os.remove(path)
    except OSError as error:
        logger.warning("Failed to remove %s: %s", path, error)


def _clear_directory(path, prefix=None):
    if not os.path.isdir(path):
        return
    for name in os.listdir(path):
        if prefix and not name.startswith(prefix):
            continue
        _remove_path(os.path.join(path, name))


def reclaim_worker_disk():
    """Drop leftover Comfy outputs so a warm worker does not fill the container disk."""
    _clear_directory(get_comfy_output_dir())
    _clear_directory(get_comfy_temp_dir())
    _clear_directory(get_comfy_input_dir(), prefix="task_")


def cleanup_job_files(task_id, staged_names=None):
    _remove_path(os.path.abspath(task_id))
    input_dir = get_comfy_input_dir()
    for name in staged_names or []:
        _remove_path(os.path.join(input_dir, name))
    _clear_directory(get_comfy_output_dir(), prefix=task_id)
    _clear_directory(get_comfy_temp_dir(), prefix=task_id)


def set_job_video_prefix(prompt, task_id):
    node = prompt.get(VHS_VIDEO_COMBINE_NODE_ID)
    if not isinstance(node, dict):
        raise Exception("workflow is missing VHS_VideoCombine node 277")
    node.setdefault("inputs", {})["filename_prefix"] = task_id
    return VHS_VIDEO_COMBINE_NODE_ID


def build_handler_result(
    *,
    video=None,
    error=None,
    saved_input_url=None,
    saved_end_input_url=None,
    input_upload_warnings=None,
):
    result = {
        "saved_input_url": saved_input_url,
        "saved_end_input_url": saved_end_input_url,
        "input_upload_warnings": input_upload_warnings or [],
    }
    if video is not None:
        result["video"] = video
    if error is not None:
        result["error"] = error
    return result


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
        encoded_data = base64_data
        if isinstance(encoded_data, str) and encoded_data.startswith("data:"):
            header, separator, encoded_data = encoded_data.partition(",")
            if not separator or ";base64" not in header.lower():
                raise ValueError("invalid Base64 data URI")
        decoded_data = base64.b64decode(encoded_data, validate=True)
        os.makedirs(temp_dir, exist_ok=True)
        file_path = os.path.abspath(os.path.join(temp_dir, output_filename))
        with open(file_path, 'wb') as f:
            f.write(decoded_data)
        
        logger.info(f"Wrote Base64 input to '{file_path}'")
        return file_path
    except (binascii.Error, ValueError) as e:
        logger.error(f"Base64 decode failed: {e}")
        raise Exception(f"Base64 decode failed: {e}")


def _multipart_form_body(fields, file_field, file_path):
    """Build a multipart/form-data request body using only the standard library."""
    boundary = f"----ComfyBridge{uuid.uuid4().hex}"
    chunks = []
    for name, value in fields.items():
        chunks.extend([
            f"--{boundary}\r\n".encode("ascii"),
            f'Content-Disposition: form-data; name="{name}"\r\n\r\n'.encode("ascii"),
            str(value).encode("utf-8"),
            b"\r\n",
        ])

    filename = os.path.basename(file_path) or "input_image.jpg"
    content_type = mimetypes.guess_type(filename)[0] or "application/octet-stream"
    chunks.extend([
        f"--{boundary}\r\n".encode("ascii"),
        (
            f'Content-Disposition: form-data; name="{file_field}"; '
            f'filename="{filename}"\r\n'
        ).encode("utf-8"),
        f"Content-Type: {content_type}\r\n\r\n".encode("ascii"),
    ])
    with open(file_path, "rb") as input_file:
        chunks.append(input_file.read())
    chunks.extend([b"\r\n", f"--{boundary}--\r\n".encode("ascii")])
    return b"".join(chunks), boundary


def upload_file_to_catbox(file_path, userhash):
    """Upload a resolved input file to Catbox and return its durable URL."""
    body, boundary = _multipart_form_body(
        {"reqtype": "fileupload", "userhash": userhash},
        "fileToUpload",
        file_path,
    )
    last_error = None
    for attempt in range(CATBOX_UPLOAD_ATTEMPTS):
        request = urllib.request.Request(
            CATBOX_API_URL,
            data=body,
            headers={
                "Content-Type": f"multipart/form-data; boundary={boundary}",
                "User-Agent": "ComfyBridge-RunPod/1.0",
            },
            method="POST",
        )
        try:
            with urllib.request.urlopen(
                request, timeout=CATBOX_UPLOAD_TIMEOUT_SECONDS
            ) as response:
                saved_url = response.read().decode("utf-8").strip()
            parsed = urllib.parse.urlparse(saved_url)
            if parsed.scheme != "https" or parsed.hostname != "files.catbox.moe":
                raise Exception("Catbox returned an invalid upload URL")
            return saved_url
        except Exception as error:
            last_error = error
            logger.warning(
                "Catbox upload failed (attempt %s/%s): %s",
                attempt + 1,
                CATBOX_UPLOAD_ATTEMPTS,
                error,
            )
            if attempt + 1 < CATBOX_UPLOAD_ATTEMPTS:
                time.sleep(CATBOX_RETRY_DELAY_SECONDS)
    raise Exception(f"Catbox upload failed: {last_error}")


def archive_resolved_input(file_path, input_type, userhash, label):
    """Best-effort archive for URL/Base64 inputs."""
    if input_type not in ("url", "base64"):
        return None, None
    if not userhash:
        return None, f"{label}: catbox_userhash is missing"
    try:
        saved_url = upload_file_to_catbox(file_path, userhash)
        logger.info("Archived %s input to Catbox: %s", label, saved_url)
        return saved_url, None
    except Exception as error:
        return None, f"{label}: {error}"


def redacted_job_input(job_input):
    """Return log-safe job metadata without credentials or Base64 payloads."""
    safe = dict(job_input)
    if "catbox_userhash" in safe:
        safe["catbox_userhash"] = "[REDACTED]"
    for key in ("image_base64", "end_image_base64"):
        if key in safe:
            safe[key] = f"[BASE64 {len(str(safe[key]))} chars]"
    for key in ("image", "end_image"):
        value = safe.get(key)
        if isinstance(value, str) and not value.startswith(("http://", "https://", "/")):
            safe[key] = f"[POSSIBLE BASE64 {len(value)} chars]"
    return safe


def wait_for_http_server(http_url, max_attempts=300, retry_delay=1):
    for http_attempt in range(max_attempts):
        try:
            with urllib.request.urlopen(http_url, timeout=5):
                logger.info("HTTP connected (attempt %s)", http_attempt + 1)
                return
        except Exception as error:
            logger.warning(
                "HTTP connect failed (attempt %s/%s): %s",
                http_attempt + 1,
                max_attempts,
                error,
            )
            if http_attempt == max_attempts - 1:
                raise Exception("Cannot connect to the ComfyUI server. Is it running?")
            time.sleep(retry_delay)


def connect_websocket(ws_url, max_attempts=36, retry_delay=5):
    ws = websocket.WebSocket()
    for attempt in range(max_attempts):
        try:
            ws.connect(ws_url)
            logger.info("WebSocket connected (attempt %s)", attempt + 1)
            return ws
        except Exception as error:
            logger.warning(
                "WebSocket connect failed (attempt %s/%s): %s",
                attempt + 1,
                max_attempts,
                error,
            )
            if attempt == max_attempts - 1:
                raise Exception("WebSocket connection timed out (3 minutes)")
            time.sleep(retry_delay)


def _read_text_if_exists(path):
    if not path or not os.path.exists(path):
        return None
    with open(path, encoding="utf-8") as handle:
        return handle.read()


def read_host_memory_stats():
    stats = {}
    meminfo_path = "/proc/meminfo"
    try:
        meminfo = _read_text_if_exists(meminfo_path)
        if meminfo:
            stats.update(parse_meminfo(meminfo))
    except OSError as error:
        logger.warning("Failed to read %s: %s", meminfo_path, error)

    cgroup_candidates = (
        ("/sys/fs/cgroup/memory.current", "/sys/fs/cgroup/memory.max"),
        (
            "/sys/fs/cgroup/memory/memory.usage_in_bytes",
            "/sys/fs/cgroup/memory/memory.limit_in_bytes",
        ),
    )
    for current_path, max_path in cgroup_candidates:
        try:
            cgroup_stats = parse_cgroup_memory(
                _read_text_if_exists(current_path),
                _read_text_if_exists(max_path),
            )
        except OSError as error:
            logger.warning("Failed to read cgroup memory %s: %s", current_path, error)
            continue
        if cgroup_stats:
            stats.update(cgroup_stats)
            break
    return stats


def read_nvidia_smi_stats():
    try:
        result = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=memory.total,memory.free",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True,
            text=True,
            timeout=5,
        )
    except (OSError, subprocess.SubprocessError) as error:
        logger.warning("nvidia-smi failed: %s", error)
        return {}
    if result.returncode != 0:
        logger.warning("nvidia-smi exited %s: %s", result.returncode, result.stderr.strip())
        return {}
    return parse_nvidia_smi_csv(result.stdout)


def fetch_comfy_system_stats():
    url = f"http://{server_address}:8188/system_stats"
    with urllib.request.urlopen(url, timeout=5) as response:
        payload = json.loads(response.read())
    return parse_comfy_system_stats(payload)


def collect_memory_stats():
    stats = {}
    stats.update(read_host_memory_stats())
    stats.update(read_nvidia_smi_stats())
    try:
        comfy_stats = fetch_comfy_system_stats()
        stats.update(comfy_stats)
    except Exception as error:
        logger.warning("Comfy /system_stats failed: %s", error)
    return stats


def _free_wait_seconds():
    raw = os.getenv("COMFY_FREE_WAIT_SECONDS", "2")
    try:
        return max(float(raw), 0.0)
    except (TypeError, ValueError):
        return 2.0


def free_comfy_models():
    """Ask Comfy to unload, then wait until the idle worker actually does it.

    POST /free only sets queue flags. Comfy applies them after the next empty
    queue wait, so queueing the next prompt too fast can run with models still loaded.
    """
    before = {}
    try:
        before = fetch_comfy_system_stats()
    except Exception as error:
        logger.warning("Comfy /system_stats before /free failed: %s", error)

    url = f"http://{server_address}:8188/free"
    body = json.dumps({"unload_models": True, "free_memory": True}).encode("utf-8")
    request = urllib.request.Request(
        url,
        data=body,
        method="POST",
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(request, timeout=30) as response:
        response.read()
    logger.info("Requested Comfy model unload via /free")

    deadline = time.monotonic() + _free_wait_seconds()
    while time.monotonic() < deadline:
        time.sleep(0.25)
        try:
            after = fetch_comfy_system_stats()
        except Exception as error:
            logger.warning("Comfy /system_stats after /free failed: %s", error)
            continue
        before_free = before.get("vram_free")
        after_free = after.get("vram_free")
        if (
            isinstance(before_free, int)
            and isinstance(after_free, int)
            and after_free > before_free
        ):
            logger.info(
                "VRAM increased after /free: %s -> %s",
                format_memory_stats({"vram_free": before_free}),
                format_memory_stats({"vram_free": after_free}),
            )
            return


def apply_dynamic_model_retention(requested_keep):
    stats = collect_memory_stats()
    logger.info("Worker memory: %s", format_memory_stats(stats))
    keep_models_loaded, reasons = resolve_keep_models_loaded(requested_keep, stats)
    if reasons:
        logger.warning(
            "Memory pressure detected (%s); unloading models before generation",
            "; ".join(reasons),
        )
        try:
            free_comfy_models()
        except Exception as error:
            logger.warning("Comfy /free failed: %s", error)
    return keep_models_loaded, stats, reasons


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
                video_path = video.get("fullpath")
                with open(video_path, "rb") as f:
                    videos_output.append(base64.b64encode(f.read()).decode("utf-8"))
                logger.info(
                    "Found Comfy video output: node=%s path=%s",
                    node_id,
                    video_path,
                )
                _remove_path(video_path)
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
    saved_input_url = None
    saved_end_input_url = None
    input_upload_warnings = []
    task_id = f"task_{uuid.uuid4()}"
    staged_names = []
    ws = None
    try:
        reclaim_worker_disk()
        job_input = job.get("input", {})
        requested_keep_models_loaded = get_keep_models_loaded(job_input)

        logger.info("Received job input: %s", redacted_job_input(job_input))
        catbox_userhash = str(job_input.get("catbox_userhash") or "").strip()

        # Resolve start image (image, image_path, image_url, or image_base64)
        image_path = None
        image_input_type = "default"
        if "image" in job_input:
            image_data = job_input["image"]
            if isinstance(image_data, str):
                if image_data.startswith("http://") or image_data.startswith("https://"):
                    image_input_type = "url"
                    image_path = process_input(image_data, task_id, "input_image.jpg", "url")
                elif os.path.exists(image_data) or image_data.startswith("/"):
                    image_input_type = "path"
                    image_path = process_input(image_data, task_id, "input_image.jpg", "path")
                else:
                    image_input_type = "base64"
                    image_path = process_input(image_data, task_id, "input_image.jpg", "base64")
            else:
                raise Exception("image must be a string")
        elif "image_path" in job_input:
            image_input_type = "path"
            image_path = process_input(job_input["image_path"], task_id, "input_image.jpg", "path")
        elif "image_url" in job_input:
            image_input_type = "url"
            image_path = process_input(job_input["image_url"], task_id, "input_image.jpg", "url")
        elif "image_base64" in job_input:
            image_input_type = "base64"
            image_path = process_input(job_input["image_base64"], task_id, "input_image.jpg", "base64")
        else:
            image_path = "/example_image.png"
            logger.info("Using default image file: /example_image.png")

        saved_input_url, upload_warning = archive_resolved_input(
            image_path, image_input_type, catbox_userhash, "start image"
        )
        if upload_warning:
            input_upload_warnings.append(upload_warning)

        # Resolve optional end image (enables FLF2V)
        end_image_path_local = None
        end_image_input_type = None
        if "end_image" in job_input:
            end_image_data = job_input["end_image"]
            if isinstance(end_image_data, str):
                if end_image_data.startswith("http://") or end_image_data.startswith("https://"):
                    end_image_input_type = "url"
                    end_image_path_local = process_input(end_image_data, task_id, "end_image.jpg", "url")
                elif os.path.exists(end_image_data) or end_image_data.startswith("/"):
                    end_image_input_type = "path"
                    end_image_path_local = process_input(end_image_data, task_id, "end_image.jpg", "path")
                else:
                    end_image_input_type = "base64"
                    end_image_path_local = process_input(end_image_data, task_id, "end_image.jpg", "base64")
            else:
                raise Exception("end_image must be a string")
        elif "end_image_path" in job_input:
            end_image_input_type = "path"
            end_image_path_local = process_input(job_input["end_image_path"], task_id, "end_image.jpg", "path")
        elif "end_image_url" in job_input:
            end_image_input_type = "url"
            end_image_path_local = process_input(job_input["end_image_url"], task_id, "end_image.jpg", "url")
        elif "end_image_base64" in job_input:
            end_image_input_type = "base64"
            end_image_path_local = process_input(job_input["end_image_base64"], task_id, "end_image.jpg", "base64")

        if end_image_path_local:
            saved_end_input_url, upload_warning = archive_resolved_input(
                end_image_path_local,
                end_image_input_type,
                catbox_userhash,
                "end image",
            )
            if upload_warning:
                input_upload_warnings.append(upload_warning)

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
        staged_names.append(image_name)
        end_image_name = None
        if end_image_path_local:
            end_image_name = stage_image_for_loadimage(
                end_image_path_local,
                unique_loadimage_name(task_id, end_image_path_local, "end_image"),
            )
            staged_names.append(end_image_name)

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
        set_job_video_prefix(prompt, task_id)

        if is_flf2v:
            prompt["483"]["inputs"]["image"] = end_image_name

        if user_lora_pairs:
            apply_loras_to_workflow(prompt, user_lora_pairs, is_flf2v, workflow_file)

        ws_url = f"ws://{server_address}:8188/ws?clientId={client_id}"
        logger.info(f"Connecting to WebSocket: {ws_url}")

        http_url = f"http://{server_address}:8188/"
        logger.info(f"Checking HTTP connection to: {http_url}")

        wait_for_http_server(http_url)
        keep_models_loaded, memory_stats, pressure_reasons = apply_dynamic_model_retention(
            requested_keep_models_loaded
        )
        configured_nodes = configure_model_retention(prompt, keep_models_loaded)
        logger.info(
            "Model retention configured: requested=%s keep_models_loaded=%s "
            "unload_all_models=%s pressure=%s nodes=%s memory=%s",
            requested_keep_models_loaded,
            keep_models_loaded,
            not keep_models_loaded,
            pressure_reasons or "none",
            configured_nodes,
            format_memory_stats(memory_stats),
        )
        ws = connect_websocket(ws_url)
        videos = get_videos(ws, prompt)

        for node_id in videos:
            if videos[node_id]:
                logger.info("Returning video from node %s (%s bytes base64)", node_id, len(videos[node_id][0]))
                return build_handler_result(
                    video=videos[node_id][0],
                    saved_input_url=saved_input_url,
                    saved_end_input_url=saved_end_input_url,
                    input_upload_warnings=input_upload_warnings,
                )

        logger.error("No video in Comfy history outputs: %s", {k: len(v) for k, v in videos.items()})
        return build_handler_result(
            error="No video was found.",
            saved_input_url=saved_input_url,
            saved_end_input_url=saved_end_input_url,
            input_upload_warnings=input_upload_warnings,
        )
    except Exception as error:
        logger.exception("Job failed: %s", error)
        return build_handler_result(
            error=str(error),
            saved_input_url=saved_input_url,
            saved_end_input_url=saved_end_input_url,
            input_upload_warnings=input_upload_warnings,
        )
    finally:
        if ws is not None:
            try:
                ws.close()
            except Exception as close_error:
                logger.warning("Failed to close WebSocket: %s", close_error)
        cleanup_job_files(task_id, staged_names)

runpod.serverless.start({"handler": handler})
