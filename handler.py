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
import binascii
import subprocess
import time

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

server_address = os.getenv('SERVER_ADDRESS', '127.0.0.1')
client_id = str(uuid.uuid4())

# I2V-only workflow (Remix NSFW + Lightning). FLF2V is not supported.
WORKFLOW_FILE = "/new_Wan22_api.json"

DEFAULT_NEGATIVE = (
    "bright tones, overexposed, static, blurred details, subtitles, style, works, "
    "paintings, images, static, overall gray, worst quality, low quality, "
    "JPEG compression residue, ugly, incomplete, extra fingers, poorly drawn hands, "
    "poorly drawn faces, deformed, disfigured, misshapen limbs, fused fingers, "
    "still picture, messy background, three legs, many people in the background, walking backwards"
)


def to_nearest_multiple_of_16(value):
    """Round to nearest multiple of 16 (min 16)."""
    try:
        numeric_value = float(value)
    except Exception:
        raise Exception(f"width/height is not numeric: {value}")
    adjusted = int(round(numeric_value / 16.0) * 16)
    if adjusted < 16:
        adjusted = 16
    return adjusted


def process_input(input_data, temp_dir, output_filename, input_type):
    if input_type == "path":
        logger.info(f"Path input: {input_data}")
        return input_data
    elif input_type == "url":
        logger.info(f"URL input: {input_data}")
        os.makedirs(temp_dir, exist_ok=True)
        file_path = os.path.abspath(os.path.join(temp_dir, output_filename))
        return download_file_from_url(input_data, file_path)
    elif input_type == "base64":
        logger.info("Base64 input")
        return save_base64_to_file(input_data, temp_dir, output_filename)
    else:
        raise Exception(f"Unsupported input type: {input_type}")


def download_file_from_url(url, output_path):
    try:
        result = subprocess.run(
            ['wget', '-O', output_path, '--no-verbose', url],
            capture_output=True,
            text=True,
        )
        if result.returncode == 0:
            logger.info(f"Downloaded: {url} -> {output_path}")
            return output_path
        logger.error(f"wget failed: {result.stderr}")
        raise Exception(f"URL download failed: {result.stderr}")
    except Exception as e:
        logger.error(f"Download error: {e}")
        raise Exception(f"Download error: {e}")


def save_base64_to_file(base64_data, temp_dir, output_filename):
    try:
        decoded_data = base64.b64decode(base64_data)
        os.makedirs(temp_dir, exist_ok=True)
        file_path = os.path.abspath(os.path.join(temp_dir, output_filename))
        with open(file_path, 'wb') as f:
            f.write(decoded_data)
        logger.info(f"Saved base64 to {file_path}")
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


def get_history(prompt_id):
    url = f"http://{server_address}:8188/history/{prompt_id}"
    logger.info(f"Getting history from: {url}")
    with urllib.request.urlopen(url) as response:
        return json.loads(response.read())


def get_videos(ws, prompt):
    prompt_id = queue_prompt(prompt)['prompt_id']
    output_videos = {}
    while True:
        out = ws.recv()
        if isinstance(out, str):
            message = json.loads(out)
            if message['type'] == 'executing':
                data = message['data']
                if data['node'] is None and data['prompt_id'] == prompt_id:
                    break
        else:
            continue

    history = get_history(prompt_id)[prompt_id]
    for node_id in history['outputs']:
        node_output = history['outputs'][node_id]
        videos_output = []
        if 'gifs' in node_output:
            for video in node_output['gifs']:
                with open(video['fullpath'], 'rb') as f:
                    video_data = base64.b64encode(f.read()).decode('utf-8')
                videos_output.append(video_data)
        output_videos[node_id] = videos_output

    return output_videos


def load_workflow(workflow_path):
    with open(workflow_path, 'r') as file:
        return json.load(file)


def handler(job):
    job_input = job.get("input", {})
    logger.info(f"Received job input keys: {list(job_input.keys())}")
    task_id = f"task_{uuid.uuid4()}"

    # Reject FLF2V inputs explicitly (I2V-only product)
    if any(k in job_input for k in ("end_image_path", "end_image_url", "end_image_base64")):
        return {
            "error": "FLF2V is not supported. This endpoint is image-to-video only. "
                     "Do not send end_image_path / end_image_url / end_image_base64."
        }

    # Image: path | url | base64
    if "image_path" in job_input:
        image_path = process_input(job_input["image_path"], task_id, "input_image.jpg", "path")
    elif "image_url" in job_input:
        image_path = process_input(job_input["image_url"], task_id, "input_image.jpg", "url")
    elif "image_base64" in job_input:
        image_path = process_input(job_input["image_base64"], task_id, "input_image.jpg", "base64")
    else:
        image_path = "/example_image.png"
        logger.info("Using default image: /example_image.png")

    # Optional user LoRAs (max 4 pairs). Lightning stays on lora_0.
    lora_pairs = job_input.get("lora_pairs", []) or []
    if len(lora_pairs) > 4:
        logger.warning(f"Got {len(lora_pairs)} LoRA pairs; only first 4 are used.")
        lora_pairs = lora_pairs[:4]
    lora_count = len(lora_pairs)

    # Lightning defaults
    length = int(job_input.get("length", 81))
    steps = int(job_input.get("steps", 4))
    cfg = float(job_input.get("cfg", 1.0))
    seed = int(job_input.get("seed", 42))
    width = job_input.get("width", 480)
    height = job_input.get("height", 832)
    context_overlap = int(job_input.get("context_overlap", 48))
    prompt_text = job_input.get("prompt")
    if not prompt_text:
        return {"error": "prompt is required"}

    negative_prompt = job_input.get("negative_prompt", DEFAULT_NEGATIVE)

    # Split: half steps for HIGH then LOW (Lightning 4 → 2/2)
    split_step = max(1, int(steps // 2))

    logger.info(
        f"I2V Remix NSFW + Lightning | steps={steps} split={split_step} "
        f"cfg={cfg} {width}x{height} frames={length} loras={lora_count}"
    )

    prompt = load_workflow(WORKFLOW_FILE)

    prompt["244"]["inputs"]["image"] = image_path
    prompt["541"]["inputs"]["num_frames"] = length
    prompt["135"]["inputs"]["positive_prompt"] = prompt_text
    prompt["135"]["inputs"]["negative_prompt"] = negative_prompt
    prompt["220"]["inputs"]["seed"] = seed
    prompt["540"]["inputs"]["seed"] = seed
    prompt["540"]["inputs"]["cfg"] = cfg

    # CFG schedule for high-noise sampler (node 220 uses 570)
    if "570" in prompt:
        prompt["570"]["inputs"]["cfg_scale_start"] = cfg
        prompt["570"]["inputs"]["cfg_scale_end"] = cfg

    adjusted_width = to_nearest_multiple_of_16(width)
    adjusted_height = to_nearest_multiple_of_16(height)
    if adjusted_width != width:
        logger.info(f"Width adjusted: {width} -> {adjusted_width}")
    if adjusted_height != height:
        logger.info(f"Height adjusted: {height} -> {adjusted_height}")
    prompt["235"]["inputs"]["value"] = adjusted_width
    prompt["236"]["inputs"]["value"] = adjusted_height
    prompt["498"]["inputs"]["context_overlap"] = context_overlap
    prompt["498"]["inputs"]["context_frames"] = length

    # Steps + split (nodes 569 / 575)
    if "569" in prompt:
        prompt["569"]["inputs"]["value"] = steps
        logger.info(f"Steps set to: {steps}")
    if "575" in prompt:
        prompt["575"]["inputs"]["value"] = split_step
        logger.info(f"Split step set to: {split_step}")

    # User LoRAs on lora_1..lora_4 (lora_0 = baked Lightning)
    if lora_count > 0:
        for i, lora_pair in enumerate(lora_pairs):
            if i >= 4:
                break
            lora_high = lora_pair.get("high")
            lora_low = lora_pair.get("low")
            lora_high_weight = lora_pair.get("high_weight", 1.0)
            lora_low_weight = lora_pair.get("low_weight", 1.0)
            slot = i + 1  # lora_1..

            if lora_high:
                prompt["279"]["inputs"][f"lora_{slot}"] = lora_high
                prompt["279"]["inputs"][f"strength_{slot}"] = lora_high_weight
                logger.info(f"LoRA {slot} HIGH: {lora_high} @ {lora_high_weight}")
            if lora_low:
                prompt["553"]["inputs"][f"lora_{slot}"] = lora_low
                prompt["553"]["inputs"][f"strength_{slot}"] = lora_low_weight
                logger.info(f"LoRA {slot} LOW: {lora_low} @ {lora_low_weight}")

    # Wait for ComfyUI HTTP
    http_url = f"http://{server_address}:8188/"
    max_http_attempts = 180
    for http_attempt in range(max_http_attempts):
        try:
            urllib.request.urlopen(http_url, timeout=5)
            logger.info(f"HTTP OK (attempt {http_attempt + 1})")
            break
        except Exception as e:
            logger.warning(f"HTTP wait {http_attempt + 1}/{max_http_attempts}: {e}")
            if http_attempt == max_http_attempts - 1:
                raise Exception("ComfyUI server not reachable")
            time.sleep(1)

    ws_url = f"ws://{server_address}:8188/ws?clientId={client_id}"
    logger.info(f"Connecting WebSocket: {ws_url}")
    ws = websocket.WebSocket()
    max_attempts = int(180 / 5)
    for attempt in range(max_attempts):
        try:
            ws.connect(ws_url)
            logger.info(f"WebSocket OK (attempt {attempt + 1})")
            break
        except Exception as e:
            logger.warning(f"WebSocket wait {attempt + 1}/{max_attempts}: {e}")
            if attempt == max_attempts - 1:
                raise Exception("WebSocket connect timeout (3 min)")
            time.sleep(5)

    videos = get_videos(ws, prompt)
    ws.close()

    for node_id in videos:
        if videos[node_id]:
            return {"video": videos[node_id][0]}

    return {"error": "Video not found."}


runpod.serverless.start({"handler": handler})
