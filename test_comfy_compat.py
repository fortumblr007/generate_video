import json
import os
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import patch


WORKFLOW_DIR = Path(__file__).parent / "workflow"
DOCKERFILE = Path(__file__).parent / "Dockerfile"

mock_runpod = types.ModuleType("runpod")
mock_runpod.serverless = types.ModuleType("serverless")
mock_runpod.serverless.start = lambda x: None
mock_runpod.serverless.utils = types.ModuleType("utils")
mock_runpod.serverless.utils.rp_upload = lambda x: None
sys.modules.setdefault("runpod", mock_runpod)
sys.modules.setdefault("runpod.serverless", mock_runpod.serverless)
sys.modules.setdefault("runpod.serverless.utils", mock_runpod.serverless.utils)

from handler import stage_image_for_loadimage, unique_loadimage_name
from workflow_options import RIFE_REQUIRED_DEFAULTS, ensure_rife_required_inputs


class StageImageForLoadImageTests(unittest.TestCase):
    def test_copies_into_input_dir_and_returns_basename(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            source = Path(temp_dir) / "source.png"
            source.write_bytes(b"png-bytes")
            input_dir = Path(temp_dir) / "input"

            with patch.dict(os.environ, {"COMFY_INPUT_DIR": str(input_dir)}):
                name = stage_image_for_loadimage(str(source), "task_input_image.png")

            self.assertEqual("task_input_image.png", name)
            self.assertFalse(os.path.isabs(name))
            staged = input_dir / name
            self.assertTrue(staged.is_file())
            self.assertEqual(b"png-bytes", staged.read_bytes())

    def test_stages_absolute_default_style_source(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            source = Path(temp_dir) / "example_image.png"
            source.write_bytes(b"default-image")
            input_dir = Path(temp_dir) / "input"

            with patch.dict(os.environ, {"COMFY_INPUT_DIR": str(input_dir)}):
                name = stage_image_for_loadimage(
                    str(source),
                    unique_loadimage_name("task_1", str(source), "input_image"),
                )

            self.assertEqual("task_1_input_image.png", name)
            self.assertEqual(b"default-image", (input_dir / name).read_bytes())

    def test_missing_source_raises(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            with patch.dict(os.environ, {"COMFY_INPUT_DIR": temp_dir}):
                with self.assertRaisesRegex(Exception, "does not exist"):
                    stage_image_for_loadimage(
                        os.path.join(temp_dir, "missing.png"),
                        "missing.png",
                    )


class RifeRequiredInputsTests(unittest.TestCase):
    def test_fills_missing_keys_and_preserves_existing_values(self):
        prompt = {
            "482": {
                "class_type": "RIFE VFI",
                "inputs": {
                    "ckpt_name": "rife49.pth",
                    "dtype": "float16",
                },
            },
            "1": {
                "class_type": "LoadImage",
                "inputs": {"image": "x.png"},
            },
        }

        configured = ensure_rife_required_inputs(prompt)

        self.assertEqual(1, configured)
        self.assertEqual("float16", prompt["482"]["inputs"]["dtype"])
        self.assertIs(False, prompt["482"]["inputs"]["torch_compile"])
        self.assertEqual(1, prompt["482"]["inputs"]["batch_size"])
        self.assertNotIn("dtype", prompt["1"]["inputs"])

    def test_all_baked_workflows_include_rife_required_keys(self):
        workflow_paths = sorted(WORKFLOW_DIR.glob("*.json"))
        self.assertEqual(6, len(workflow_paths))

        for workflow_path in workflow_paths:
            with self.subTest(workflow=workflow_path.name):
                with workflow_path.open(encoding="utf-8") as workflow_file:
                    prompt = json.load(workflow_file)

                rife_nodes = [
                    node for node in prompt.values()
                    if node.get("class_type") == "RIFE VFI"
                ]
                self.assertTrue(rife_nodes)
                for node in rife_nodes:
                    for key in RIFE_REQUIRED_DEFAULTS:
                        self.assertIn(key, node["inputs"])

                self.assertGreaterEqual(ensure_rife_required_inputs(prompt), 1)


class DockerfileCompatTests(unittest.TestCase):
    def test_pins_frame_interpolation_and_stages_default_image(self):
        dockerfile = DOCKERFILE.read_text(encoding="utf-8")
        self.assertIn(
            "git checkout 26545cc2dd95bc3d27f056016300673bdeee78f5",
            dockerfile,
        )
        self.assertIn(
            "mkdir -p /ComfyUI/input && cp /example_image.png /ComfyUI/input/example_image.png",
            dockerfile,
        )


if __name__ == "__main__":
    unittest.main()
