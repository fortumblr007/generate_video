# Wan2.2 Generate Video API Client

This project provides a Python client for generating videos from images using **Wan2.2** through a RunPod Serverless endpoint. The worker uses ComfyUI's native dual-pass ksampler (high-noise then low-noise), baked LightX2V 4-step LoRAs, and five selectable baked LoRA presets for single-image requests. A request can use up to two of them.

[![Runpod](https://api.runpod.io/badge/fortumblr007/generate_video)](https://console.runpod.io/hub/listing/fortumblr007/generate_video)

**Wan2.2** is an advanced AI model that converts static images into dynamic videos with natural motion and realistic animations. It's built on top of ComfyUI and provides high-quality video generation capabilities.

## 🎨 Engui Studio Integration

[![EnguiStudio](https://raw.githubusercontent.com/wlsdml1114/Engui_Studio/main/assets/banner.png)](https://github.com/wlsdml1114/Engui_Studio)

This Wan2.2 client is primarily designed for **Engui Studio**, a comprehensive AI model management platform. While it can be used via API, Engui Studio provides enhanced features and broader model support.

## ✨ Key Features

*   **Wan2.2 Model**: Powered by the advanced Wan2.2 AI model for high-quality video generation.
*   **Image-to-Video Generation**: Converts static images into dynamic videos with natural motion.
*   **Base64 Encoding Support**: Handles image encoding/decoding automatically.
*   **LoRA Configuration**: Baked LightX2V lightning LoRAs plus two selectable high/low LoRA presets.
*   **Batch Processing**: Process multiple images in a single operation.
*   **Error Handling**: Comprehensive error handling and logging.
*   **Async Job Management**: Automatic job submission and status monitoring.
*   **ComfyUI Integration**: Built on ComfyUI for flexible workflow management.

## 🚀 RunPod Serverless Template

This template includes all the necessary components to run **Wan2.2** as a RunPod Serverless Worker.

*   **Dockerfile**: Configures the environment and installs all dependencies required for Wan2.2 model execution.
*   **handler.py**: Implements the handler function that processes requests for RunPod Serverless.
*   **entrypoint.sh**: Performs initialization tasks when the worker starts.
*   **workflow/wan22_*.json**: Dual-pass Wan 2.2 I2V graphs (zero, one, or two selected presets, plus FLF2V).
*   Requires a **32GB+** GPU (5090 / 5000 Ada / A6000 / A40 / L40 class). 24GB cards (4090) will OOM when both 14B experts load.

## 📖 Python Client Usage

### Basic Usage

```python
from generate_video_client import GenerateVideoClient

# Initialize client
client = GenerateVideoClient(
    runpod_endpoint_id="your-endpoint-id",
    runpod_api_key="your-runpod-api-key"
)

# Generate video from image
result = client.create_video_from_image(
    image_path="./example_image.png",
    prompt="running man, grab the gun",
    negative_prompt="blurry, low quality, distorted",
    width=480,
    height=832,
    length=81,
    steps=4,
    seed=-1,
    cfg=1.0,
    high_lora_strength=0.4,
    low_lora_strength=1.0
)

# Save result if successful
if result.get('status') == 'COMPLETED':
    client.save_video_result(result, "./output_video.mp4")
else:
    print(f"Error: {result.get('error')}")
```

### Using LoRA

```python
# Select either or both baked presets. Omitted weights use 0.8 high and 0.7 low.
lora_presets = [{"name": "assume_the_position"}]

# Generate video with LoRA
result = client.create_video_from_image(
    image_path="./example_image.png",
    prompt="running man, grab the gun",
    negative_prompt="blurry, low quality, distorted",
    width=480,
    height=832,
    length=81,
    steps=4,
    seed=-1,
    cfg=1.0,
    high_lora_strength=0.4,
    low_lora_strength=1.0,
    lora_presets=lora_presets
)
```

### Batch Processing

```python
# Process multiple images
batch_result = client.batch_process_images(
    image_folder_path="./input_images",
    output_folder_path="./output_videos",
    prompt="running man, grab the gun",
    negative_prompt="blurry, low quality, distorted",
    width=480,
    height=832,
    length=81,
    steps=4,
    seed=-1,
    cfg=1.0,
    high_lora_strength=0.4,
    low_lora_strength=1.0
)

print(f"Batch processing completed: {batch_result['successful']}/{batch_result['total_files']} successful")
```

## 🔧 API Reference

### Input

The `input` object must contain the following fields. Images can be input using **path, URL or Base64** - one method for each.

#### Image Input (use only one)
| Parameter | Type | Required | Default | Description |
| --- | --- | --- | --- | --- |
| `image` | `string` | No | - | Auto-detected path, URL, or Base64 image |
| `image_path` | `string` | No | - | Local path to the input image |
| `image_url` | `string` | No | - | URL of the input image |
| `image_base64` | `string` | No | - | Base64 encoded string of the input image |
| `end_image`, `end_image_url`, `end_image_base64` | `string` | No | - | Optional final frame; enables FLF2V and accepts the same URL/Base64 forms |
| `catbox_userhash` | `string` | No | - | Catbox account hash used to archive URL/Base64 start and end images |

When `catbox_userhash` is supplied, the worker uploads the exact resolved image files to Catbox before generation. Upload failures do not stop generation and are reported in `input_upload_warnings`. Catbox uploads are public; keep the user hash out of logs and source control.

#### LoRA Configuration
| Parameter | Type | Required | Default | Description |
| --- | --- | --- | --- | --- |
| `lora_presets` | `array` | No | `[]` | Up to two of `assume_the_position`, `airblow`, `clothes_on_off`, `sudden_outfit_change`, and `tittdrop` |

All preset files are baked into the image. `clothes_on_off` and `sudden_outfit_change` have a high-noise file only; their low-noise node is left at strength 0. No network volume is needed for these LoRAs. The old filename-based `lora_pairs` field is rejected.

#### LoRA Preset Structure
| Parameter | Type | Required | Default | Description |
| --- | --- | --- | --- | --- |
| `name` | `string` | Yes | - | `assume_the_position`, `airblow`, `clothes_on_off`, `sudden_outfit_change`, or `tittdrop` |
| `high_weight` | `float` | No | `0.8` | High-noise LoRA weight |
| `low_weight` | `float` | No | `0.7` | Low-noise LoRA weight |

Duplicate or unknown presets and non-finite weights are rejected. First/last-frame requests do not support nonempty `lora_presets`.

#### Video Generation Parameters
| Parameter | Type | Required | Default | Description |
| --- | --- | --- | --- | --- |
| `prompt` | `string` | Yes | - | Description text for the video to be generated |
| `negative_prompt` | `string` | No | - | Negative prompt to exclude unwanted elements from the video |
| `seed` | `integer` | No | `-1` | RandomNoise seed. `-1` (default) makes the handler pick a random seed |
| `cfg` | `float` | No | `1.0` | CFG scale for the high-noise pass |
| `high_lora_strength` | `float` | No | `0.4` | Strength of the baked high-noise LightX2V 4-step LoRA |
| `low_lora_strength` | `float` | No | `1.0` | Strength of the baked low-noise LightX2V 4-step LoRA |
| `width` | `integer` | No | `480` | Width of the output video in pixels |
| `height` | `integer` | No | `832` | Height of the output video in pixels |
| `length` | `integer` | No | `81` | Length of the generated video |
| `steps` | `integer` | No | `4` | Total denoising steps, always split in half across high/low noise |
| `keep_models_loaded` | `boolean` | No | `false` | When `true`, skip the workflow's forced end-of-job model unload so a warm worker can reuse models when VRAM/RAM headroom allows |

`seed` of `-1` (or omitting `seed`) makes the handler draw a random seed and write it into the high-noise `RandomNoise` node. The low-noise sampler does not take a seed.

`high_lora_strength` and `low_lora_strength` control the baked LightX2V lightning LoRAs on the high-noise and low-noise experts. Both must be positive finite numbers, so LightX2V stays enabled. Set `cfg` to `1.0` with LightX2V; CFG above 1 fights that distill (slower, more artifacts). The worker does not enforce the CFG recommendation.

`keep_models_loaded` must be a JSON boolean, not a string. The worker still unloads models when free VRAM is below 4 GiB or about 85% used (or host/cgroup RAM is similarly tight), including a Comfy `/free` call before generation so a previous keep-loaded job cannot OOM the next one. Tune with `MODEL_KEEP_MIN_FREE_VRAM_MB`, `MODEL_KEEP_MAX_VRAM_USED_RATIO`, `MODEL_KEEP_MIN_FREE_RAM_MB`, and `MODEL_KEEP_MAX_RAM_USED_RATIO`. ComfyUI may also selectively evict models during a job when it needs VRAM.

**Request Examples:**

#### 1. Basic Generation (LightX2V only)
```json
{
  "input": {
    "prompt": "running man, grab the gun",
    "negative_prompt": "blurry, low quality, distorted",
    "image_base64": "data:image/jpeg;base64,/9j/4AAQSkZJRgABAQAAAQABAAD...",
    "seed": -1,
    "cfg": 1.0,
    "high_lora_strength": 0.4,
    "low_lora_strength": 1.0,
    "width": 480,
    "height": 832,
    "length": 81,
    "steps": 4,
    "keep_models_loaded": true
  }
}
```

#### 2. With One LoRA Preset
```json
{
  "input": {
    "prompt": "running man, grab the gun",
    "negative_prompt": "blurry, low quality, distorted",
    "image_base64": "data:image/jpeg;base64,/9j/4AAQSkZJRgABAQAAAQABAAD...",
    "seed": -1,
    "cfg": 1.0,
    "high_lora_strength": 0.4,
    "low_lora_strength": 1.0,
    "width": 480,
    "height": 832,
    "lora_presets": [{"name": "assume_the_position"}]
  }
}
```

#### 3. Both LoRA Presets
```json
{
  "input": {
    "prompt": "running man, grab the gun",
    "negative_prompt": "blurry, low quality, distorted",
    "image_path": "/my_volume/image.jpg",
    "seed": -1,
    "cfg": 1.0,
    "high_lora_strength": 0.4,
    "low_lora_strength": 1.0,
    "width": 480,
    "height": 832,
    "lora_presets": [
      {"name": "assume_the_position"},
      {"name": "airblow", "high_weight": 0.6, "low_weight": 0.5}
    ]
  }
}
```

#### 4. URL Image Input
```json
{
  "input": {
    "prompt": "running man, grab the gun",
    "negative_prompt": "blurry, low quality, distorted",
    "image_url": "https://example.com/image.jpg",
    "seed": -1,
    "cfg": 1.0,
    "high_lora_strength": 0.4,
    "low_lora_strength": 1.0,
    "width": 480,
    "height": 832
  }
}
```

### Output

#### Success

If the job is successful, it returns a JSON object with the generated video Base64 encoded.

| Parameter | Type | Description |
| --- | --- | --- |
| `video` | `string` | Base64 encoded video file data. |
| `saved_input_url` | `string` or `null` | Durable Catbox URL for the resolved start image. |
| `saved_end_input_url` | `string` or `null` | Durable Catbox URL for the optional resolved end image. |
| `input_upload_warnings` | `array` | Non-fatal Catbox archival warnings. |

**Success Response Example:**

```json
{
  "video": "data:video/mp4;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNkYPhfDwAChwGA60e6kgAAAABJRU5ErkJggg==",
  "saved_input_url": "https://files.catbox.moe/example.jpg",
  "saved_end_input_url": null,
  "input_upload_warnings": []
}
```

#### Error

If the job fails, it returns a JSON object containing an error message. Archival fields are still included when the start/end images were already uploaded.

| Parameter | Type | Description |
| --- | --- | --- |
| `error` | `string` | Description of the error that occurred. |
| `saved_input_url` | `string` or `null` | Durable Catbox URL for the resolved start image, if archival already succeeded. |
| `saved_end_input_url` | `string` or `null` | Durable Catbox URL for the optional resolved end image, if archival already succeeded. |
| `input_upload_warnings` | `array` | Non-fatal Catbox archival warnings. |

**Error Response Example:**

```json
{
  "error": "No video was found.",
  "saved_input_url": "https://files.catbox.moe/example.jpg",
  "saved_end_input_url": null,
  "input_upload_warnings": []
}
```

## 🛠️ Direct API Usage

1.  Create a Serverless Endpoint on RunPod based on this repository.
2.  Once the build is complete and the endpoint is active, submit jobs via HTTP POST requests according to the API Reference above.

### 📁 Using Network Volumes

Instead of directly transmitting Base64 encoded images, you can use RunPod's Network Volumes to handle large image files. The preset LoRAs are already in the worker image.

1.  **Create and Connect Network Volume**: Create a Network Volume (e.g., S3-based volume) from the RunPod dashboard and connect it to your Serverless Endpoint settings.
2.  **Upload Files**: Upload input images to the Network Volume.
3.  **Specify Paths**: Set `image_path` to the full path to an image (e.g., `"/my_volume/images/portrait.jpg"`). Select LoRAs with `lora_presets`.

## 🔧 Client Methods

### GenerateVideoClient Class

#### `__init__(runpod_endpoint_id, runpod_api_key)`
Initialize the client with RunPod endpoint ID and API key.

#### `create_video_from_image(...)`
Generate video from a single image.

**Parameters:**
- `image_path` (str): Path to the input image
- `prompt` (str): Text prompt for video generation
- `negative_prompt` (str): Negative prompt to exclude unwanted elements (default: None)
- `width` (int): Output video width (default: 480)
- `height` (int): Output video height (default: 832)
- `length` (int): Number of frames (default: 81)
- `steps` (int): Total denoising steps, split in half (default: 4)
- `seed` (int): RandomNoise seed. `-1` randomizes (default: -1)
- `cfg` (float): High-noise CFG scale (default: 1.0)
- `high_lora_strength` (float): Baked high-noise LightX2V LoRA strength (default: 0.4)
- `low_lora_strength` (float): Baked low-noise LightX2V LoRA strength (default: 1.0)
- `lora_presets` (list): Selected baked presets (default: none; each selected preset defaults to high weight 0.8 and low weight 0.7)
- `keep_models_loaded` (bool): Skip forced end-of-job model unloading (default: False)

#### `batch_process_images(image_folder_path, output_folder_path, valid_extensions, ...)`
Process multiple images in a folder.

**Parameters:**
- `image_folder_path` (str): Path to folder containing images
- `output_folder_path` (str): Path to save output videos
- `valid_extensions` (tuple): Valid image extensions (default: ('.jpg', '.jpeg', '.png', '.bmp', '.tiff'))
- Other parameters same as `create_video_from_image`

#### `save_video_result(result, output_path)`
Save video result to file.

**Parameters:**
- `result` (dict): Job result dictionary
- `output_path` (str): Path to save the video file

## 🔧 Wan2.2 Workflow Configuration

This worker uses native ComfyUI ksampler graphs under `workflow/`:

*   **wan22_nolora.json** through **wan22_2lora.json**: single-image I2V, selected by preset count
*   **wan22_flf2v.json**: first-and-last-frame I2V when `end_image*` is set

Each graph runs a high-noise expert then a low-noise expert (steps split in half), baked LightX2V LoRAs, Sage Attention, and RIFE frame interpolation. Torch compile is disabled at runtime.

## 🙏 About Wan2.2

**Wan2.2** is a state-of-the-art AI model for image-to-video generation that produces high-quality videos with natural motion and realistic animations. This project provides a Python client and RunPod serverless template for easy deployment and usage of the Wan2.2 model.

### Key Features of Wan2.2:
- **High-Quality Output**: Generates videos with excellent visual quality and smooth motion
- **Natural Animation**: Creates realistic and natural-looking movements from static images
- **LoRA Support**: Supports LoRA (Low-Rank Adaptation) for fine-tuned video generation
- **ComfyUI Integration**: Built on ComfyUI for flexible workflow management
- **Customizable Parameters**: Full control over video generation parameters

## 🙏 Original Project

This project is based on the following original repository. All rights to the model and core logic belong to the original authors.

*   **Wan2.2:** [https://github.com/Wan-Video/Wan2.2](https://github.com/Wan-Video/Wan2.2)
*   **ComfyUI:** [https://github.com/comfyanonymous/ComfyUI](https://github.com/comfyanonymous/ComfyUI)
*   **ComfyUI-KJNodes:** [https://github.com/kijai/ComfyUI-KJNodes](https://github.com/kijai/ComfyUI-KJNodes)

## 📄 License

The original Wan2.2 project follows its respective license. This template also adheres to that license.
