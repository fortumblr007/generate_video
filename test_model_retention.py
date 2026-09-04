import copy
import json
import os
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

from generate_video_client import GenerateVideoClient
from workflow_options import (
    MIB,
    configure_model_retention,
    get_keep_models_loaded,
    get_retention_thresholds,
    memory_pressure_reasons,
    parse_cgroup_memory,
    parse_comfy_system_stats,
    parse_meminfo,
    parse_nvidia_smi_csv,
    resolve_keep_models_loaded,
)

mock_runpod = types.ModuleType("runpod")
mock_runpod.serverless = types.ModuleType("serverless")
mock_runpod.serverless.start = lambda config: None
mock_runpod.serverless.utils = types.ModuleType("utils")
mock_runpod.serverless.utils.rp_upload = lambda value: None
sys.modules.setdefault("runpod", mock_runpod)
sys.modules.setdefault("runpod.serverless", mock_runpod.serverless)
sys.modules.setdefault("runpod.serverless.utils", mock_runpod.serverless.utils)

import handler


WORKFLOW_DIR = Path(__file__).parent / "workflow"
DOCKERFILE = Path(__file__).parent / "Dockerfile"


class ModelRetentionTests(unittest.TestCase):
    def test_input_defaults_to_existing_unload_behavior(self):
        self.assertFalse(get_keep_models_loaded({}))

    def test_input_accepts_json_booleans(self):
        self.assertTrue(get_keep_models_loaded({"keep_models_loaded": True}))
        self.assertFalse(get_keep_models_loaded({"keep_models_loaded": False}))

    def test_input_rejects_strings_and_numbers(self):
        for invalid_value in ("true", "false", 0, 1, None):
            with self.subTest(invalid_value=invalid_value):
                with self.assertRaisesRegex(ValueError, "JSON boolean"):
                    get_keep_models_loaded({"keep_models_loaded": invalid_value})

    def test_all_baked_workflows_support_per_request_retention(self):
        workflow_paths = sorted(WORKFLOW_DIR.glob("*.json"))
        self.assertEqual(6, len(workflow_paths))

        for workflow_path in workflow_paths:
            with self.subTest(workflow=workflow_path.name):
                with workflow_path.open(encoding="utf-8") as workflow_file:
                    original_prompt = json.load(workflow_file)

                keep_prompt = copy.deepcopy(original_prompt)
                configured = configure_model_retention(keep_prompt, True)
                self.assertGreaterEqual(configured, 1)
                keep_nodes = [
                    node for node in keep_prompt.values()
                    if node.get("class_type") == "VRAM_Debug"
                ]
                self.assertTrue(keep_nodes)
                self.assertTrue(all(
                    node["inputs"]["unload_all_models"] is False
                    for node in keep_nodes
                ))
                self.assertTrue(all(
                    node["inputs"]["empty_cache"] is True
                    and node["inputs"]["gc_collect"] is True
                    for node in keep_nodes
                ))

                unload_prompt = copy.deepcopy(original_prompt)
                configure_model_retention(unload_prompt, False)
                unload_nodes = [
                    node for node in unload_prompt.values()
                    if node.get("class_type") == "VRAM_Debug"
                ]
                self.assertTrue(all(
                    node["inputs"]["unload_all_models"] is True
                    for node in unload_nodes
                ))

    def test_missing_vram_debug_node_fails_loudly(self):
        with self.assertRaisesRegex(ValueError, "VRAM_Debug"):
            configure_model_retention({}, True)

    def test_python_client_sends_model_retention_option(self):
        client = GenerateVideoClient("test-endpoint", "test-key")
        client.submit_job = Mock(return_value="test-job")
        client.wait_for_completion = Mock(return_value={"status": "COMPLETED"})

        result = client.create_video_from_image(
            image=b"test-image",
            prompt="test prompt",
            keep_models_loaded=True,
        )

        self.assertEqual({"status": "COMPLETED"}, result)
        submitted_input = client.submit_job.call_args.args[0]
        self.assertTrue(submitted_input["keep_models_loaded"])

    def test_python_client_rejects_string_boolean(self):
        client = GenerateVideoClient("test-endpoint", "test-key")
        with self.assertRaisesRegex(TypeError, "boolean"):
            client.create_video_from_image(
                image=b"test-image",
                keep_models_loaded="true",
            )


class MemoryPressureTests(unittest.TestCase):
    def test_plenty_of_memory_is_not_pressure(self):
        stats = {
            "vram_total": 32 * MIB * 1024,
            "vram_free": 20 * MIB * 1024,
            "ram_total": 64 * MIB * 1024,
            "ram_free": 32 * MIB * 1024,
        }
        self.assertEqual([], memory_pressure_reasons(stats))

    def test_low_free_vram_is_pressure(self):
        stats = {
            "vram_total": 32 * MIB * 1024,
            "vram_free": 1024 * MIB,
            "ram_total": 64 * MIB * 1024,
            "ram_free": 32 * MIB * 1024,
        }
        reasons = memory_pressure_reasons(stats)
        self.assertTrue(any("VRAM free" in reason for reason in reasons))

    def test_high_vram_used_ratio_is_pressure(self):
        stats = {
            "vram_total": 40 * MIB * 1024,
            "vram_free": 4 * MIB * 1024,
            "ram_total": 64 * MIB * 1024,
            "ram_free": 32 * MIB * 1024,
        }
        reasons = memory_pressure_reasons(stats)
        self.assertTrue(any("VRAM used" in reason for reason in reasons))

    def test_low_free_ram_is_pressure(self):
        stats = {
            "vram_total": 32 * MIB * 1024,
            "vram_free": 20 * MIB * 1024,
            "ram_total": 16 * MIB * 1024,
            "ram_free": 512 * MIB,
        }
        reasons = memory_pressure_reasons(stats)
        self.assertTrue(any("RAM free" in reason for reason in reasons))

    def test_missing_stats_do_not_force_unload(self):
        self.assertEqual([], memory_pressure_reasons({}))
        keep, reasons = resolve_keep_models_loaded(True, {})
        self.assertTrue(keep)
        self.assertEqual([], reasons)

    def test_keep_request_is_overridden_on_pressure(self):
        stats = {"vram_total": 24 * MIB * 1024, "vram_free": 512 * MIB}
        keep, reasons = resolve_keep_models_loaded(True, stats)
        self.assertFalse(keep)
        self.assertTrue(reasons)

    def test_explicit_unload_stays_unload_even_with_headroom(self):
        stats = {
            "vram_total": 32 * MIB * 1024,
            "vram_free": 20 * MIB * 1024,
            "ram_total": 64 * MIB * 1024,
            "ram_free": 32 * MIB * 1024,
        }
        keep, reasons = resolve_keep_models_loaded(False, stats)
        self.assertFalse(keep)
        self.assertEqual([], reasons)

    def test_parse_comfy_system_stats_prefers_cuda_device(self):
        payload = {
            "system": {"ram_total": 64 * MIB, "ram_free": 8 * MIB},
            "devices": [
                {"type": "cpu", "vram_total": 1, "vram_free": 1},
                {"type": "cuda", "vram_total": 32 * MIB, "vram_free": 4 * MIB},
            ],
        }
        self.assertEqual(
            {
                "ram_total": 64 * MIB,
                "ram_free": 8 * MIB,
                "vram_total": 32 * MIB,
                "vram_free": 4 * MIB,
            },
            parse_comfy_system_stats(payload),
        )

    def test_parse_nvidia_smi_and_meminfo(self):
        self.assertEqual(
            {"vram_total": 32768 * MIB, "vram_free": 4096 * MIB},
            parse_nvidia_smi_csv("32768, 4096\n"),
        )
        self.assertEqual(
            {"ram_total": 32768000 * 1024, "ram_free": 8192000 * 1024},
            parse_meminfo("MemTotal: 32768000 kB\nMemAvailable: 8192000 kB\n"),
        )

    def test_parse_cgroup_ignores_unlimited_and_reads_limits(self):
        self.assertEqual({}, parse_cgroup_memory("100", "max"))
        self.assertEqual({}, parse_cgroup_memory("100", str(1 << 62)))
        self.assertEqual(
            {"ram_total": 8 * MIB, "ram_free": 3 * MIB},
            parse_cgroup_memory(str(5 * MIB), str(8 * MIB)),
        )

    def test_collect_memory_stats_keeps_cgroup_ram_over_comfy_host_ram(self):
        cgroup_ram = {
            "ram_total": 64 * 1024 * MIB,
            "ram_free": 8 * 1024 * MIB,
        }
        comfy_host_stats = {
            "ram_total": 1024 * 1024 * MIB,
            "ram_free": 900 * 1024 * MIB,
            "vram_total": 32 * 1024 * MIB,
            "vram_free": 16 * 1024 * MIB,
        }
        with (
            patch.object(handler, "read_host_memory_stats", return_value=cgroup_ram),
            patch.object(handler, "read_nvidia_smi_stats", return_value={}),
            patch.object(handler, "fetch_comfy_system_stats", return_value=comfy_host_stats),
        ):
            stats = handler.collect_memory_stats()

        self.assertEqual(cgroup_ram["ram_total"], stats["ram_total"])
        self.assertEqual(cgroup_ram["ram_free"], stats["ram_free"])
        self.assertEqual(comfy_host_stats["vram_total"], stats["vram_total"])

    def test_thresholds_read_env(self):
        with patch.dict(os.environ, {
            "MODEL_KEEP_MIN_FREE_VRAM_MB": "4096",
            "MODEL_KEEP_MAX_VRAM_USED_RATIO": "0.5",
            "MODEL_KEEP_MIN_FREE_RAM_MB": "1024",
            "MODEL_KEEP_MAX_RAM_USED_RATIO": "0.8",
        }):
            thresholds = get_retention_thresholds()
        self.assertEqual(4096 * MIB, thresholds["min_free_vram_bytes"])
        self.assertEqual(0.5, thresholds["max_vram_used_ratio"])
        self.assertEqual(1024 * MIB, thresholds["min_free_ram_bytes"])
        self.assertEqual(0.8, thresholds["max_ram_used_ratio"])

    def test_handler_unloads_and_frees_on_tight_vram(self):
        stats = {
            "vram_total": 32 * MIB * 1024,
            "vram_free": 512 * MIB,
            "ram_total": 64 * MIB * 1024,
            "ram_free": 32 * MIB * 1024,
        }
        with (
            patch.object(handler, "collect_memory_stats", return_value=stats),
            patch.object(handler, "free_comfy_models") as free_models,
        ):
            keep, returned_stats, reasons = handler.apply_dynamic_model_retention(True)

        self.assertFalse(keep)
        self.assertEqual(stats, returned_stats)
        self.assertTrue(reasons)
        free_models.assert_called_once()

    def test_handler_keeps_models_when_there_is_headroom(self):
        stats = {
            "vram_total": 32 * MIB * 1024,
            "vram_free": 20 * MIB * 1024,
            "ram_total": 64 * MIB * 1024,
            "ram_free": 32 * MIB * 1024,
        }
        with (
            patch.object(handler, "collect_memory_stats", return_value=stats),
            patch.object(handler, "free_comfy_models") as free_models,
        ):
            keep, _, reasons = handler.apply_dynamic_model_retention(True)

        self.assertTrue(keep)
        self.assertEqual([], reasons)
        free_models.assert_not_called()

    def test_handler_frees_leftover_models_even_when_keep_was_false(self):
        stats = {
            "vram_total": 32 * MIB * 1024,
            "vram_free": 512 * MIB,
            "ram_total": 64 * MIB * 1024,
            "ram_free": 32 * MIB * 1024,
        }
        with (
            patch.object(handler, "collect_memory_stats", return_value=stats),
            patch.object(handler, "free_comfy_models") as free_models,
        ):
            keep, _, reasons = handler.apply_dynamic_model_retention(False)

        self.assertFalse(keep)
        self.assertTrue(reasons)
        free_models.assert_called_once()


class FakeHttpResponse:
    def __init__(self, body=b""):
        self.body = body

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, traceback):
        return False

    def read(self):
        return self.body


def _request_url(request):
    if isinstance(request, str):
        return request
    return request.full_url


class HandlerRuntimeTests(unittest.TestCase):
    def _run_job(self, vram_free, keep_models_loaded=True):
        captured = {}
        with tempfile.TemporaryDirectory() as temp_dir:
            image = Path(temp_dir) / "start.png"
            image.write_bytes(b"png-bytes")
            output_dir = Path(temp_dir) / "output"
            input_dir = Path(temp_dir) / "input"
            comfy_temp = Path(temp_dir) / "temp"
            output_dir.mkdir()
            input_dir.mkdir()
            comfy_temp.mkdir()
            video_path = Path(temp_dir) / "result.mp4"
            video_path.write_bytes(b"mp4-bytes")

            comfy_stats = {
                "system": {"ram_total": 64 * 1024 ** 3, "ram_free": 32 * 1024 ** 3},
                "devices": [{
                    "type": "cuda",
                    "index": 0,
                    "name": "cuda:0",
                    "vram_total": 32 * 1024 ** 3,
                    "vram_free": vram_free,
                }],
            }

            def fake_urlopen(request, timeout=None):
                url = _request_url(request)
                if url.endswith("/system_stats"):
                    return FakeHttpResponse(json.dumps(comfy_stats).encode("utf-8"))
                if url.endswith("/free"):
                    captured["free"] = json.loads(request.data.decode("utf-8"))
                    return FakeHttpResponse(b"")
                if url.endswith("/prompt"):
                    captured["prompt"] = json.loads(request.data.decode("utf-8"))
                    return FakeHttpResponse(json.dumps({"prompt_id": "p1"}).encode("utf-8"))
                if "/history/p1" in url:
                    return FakeHttpResponse(json.dumps({
                        "p1": {
                            "status": {"status_str": "success"},
                            "outputs": {
                                "277": {"gifs": [{"fullpath": str(video_path)}]},
                            },
                        }
                    }).encode("utf-8"))
                return FakeHttpResponse(b"ok")

            ws = Mock()
            ws.recv.return_value = json.dumps({
                "type": "executing",
                "data": {"node": None, "prompt_id": "p1"},
            })

            with (
                patch.dict(os.environ, {
                    "COMFY_OUTPUT_DIR": str(output_dir),
                    "COMFY_INPUT_DIR": str(input_dir),
                    "COMFY_TEMP_DIR": str(comfy_temp),
                    "COMFY_FREE_WAIT_SECONDS": "0",
                }),
                patch.object(handler.urllib.request, "urlopen", side_effect=fake_urlopen),
                patch.object(handler, "read_nvidia_smi_stats", return_value={}),
                patch.object(handler, "read_host_memory_stats", return_value={
                    "ram_total": 64 * 1024 ** 3,
                    "ram_free": 32 * 1024 ** 3,
                }),
                patch.object(handler, "connect_websocket", return_value=ws),
            ):
                result = handler.handler({
                    "input": {
                        "image_path": str(image),
                        "prompt": "hello",
                        "keep_models_loaded": keep_models_loaded,
                    }
                })

            vram_nodes = []
            queued = captured.get("prompt", {}).get("prompt", {})
            for node in queued.values():
                if node.get("class_type") == "VRAM_Debug":
                    vram_nodes.append(node)
            captured["vram_nodes"] = vram_nodes
            captured["result"] = result
            return captured

    def test_full_handler_unloads_when_vram_is_below_4gb(self):
        captured = self._run_job(vram_free=512 * 1024 ** 2, keep_models_loaded=True)
        result = captured["result"]
        self.assertIn("video", result)
        self.assertNotIn("error", result)
        self.assertEqual(
            {"unload_models": True, "free_memory": True},
            captured["free"],
        )
        self.assertTrue(captured["vram_nodes"])
        self.assertTrue(all(
            node["inputs"]["unload_all_models"] is True
            for node in captured["vram_nodes"]
        ))
        queued = captured["prompt"]["prompt"]
        self.assertTrue(queued["277"]["inputs"]["filename_prefix"].startswith("task_"))

    def test_full_handler_keeps_models_when_vram_has_headroom(self):
        captured = self._run_job(vram_free=20 * 1024 ** 3, keep_models_loaded=True)
        result = captured["result"]
        self.assertIn("video", result)
        self.assertNotIn("error", result)
        self.assertNotIn("free", captured)
        self.assertTrue(captured["vram_nodes"])
        self.assertTrue(all(
            node["inputs"]["unload_all_models"] is False
            for node in captured["vram_nodes"]
        ))

    def test_free_comfy_models_waits_until_vram_increases(self):
        stats = [
            {"vram_total": 32 * 1024 ** 3, "vram_free": 512 * 1024 ** 2},
            {"vram_total": 32 * 1024 ** 3, "vram_free": 512 * 1024 ** 2},
            {"vram_total": 32 * 1024 ** 3, "vram_free": 12 * 1024 ** 3},
        ]
        with (
            patch.dict(os.environ, {"COMFY_FREE_WAIT_SECONDS": "5"}),
            patch.object(handler, "fetch_comfy_system_stats", side_effect=stats),
            patch.object(
                handler.urllib.request,
                "urlopen",
                return_value=FakeHttpResponse(b""),
            ),
            patch.object(handler.time, "sleep") as sleep,
        ):
            handler.free_comfy_models()
        self.assertEqual(2, sleep.call_count)


class BakeConfigurationTests(unittest.TestCase):
    def test_dockerfile_exposes_bake_time_image_and_model_arguments(self):
        dockerfile = DOCKERFILE.read_text(encoding="utf-8")
        self.assertIn("ARG BASE_IMAGE=", dockerfile)
        self.assertIn("FROM ${BASE_IMAGE} AS runtime", dockerfile)
        self.assertIn("ARG WAN_HIGH_NOISE_MODEL_URI=hf://", dockerfile)
        self.assertIn("ARG WAN_LOW_NOISE_MODEL_URI=hf://", dockerfile)
        self.assertIn('pip install --no-cache-dir -U "huggingface_hub[hf_transfer]"', dockerfile)
        self.assertIn("hf --help > /dev/null", dockerfile)
        self.assertIn('hf download "${WAN_HIGH_NOISE_MODEL_URI}"', dockerfile)
        self.assertIn('hf download "${WAN_LOW_NOISE_MODEL_URI}"', dockerfile)
        self.assertIn("--local-dir /tmp/wan-high --force-download", dockerfile)
        self.assertIn("--local-dir /tmp/wan-low --force-download", dockerfile)
        self.assertNotIn("wget ", dockerfile)
        self.assertEqual(dockerfile.count("hf download "), 7)

    def test_baked_models_are_refreshed_and_checksum_verified(self):
        dockerfile = DOCKERFILE.read_text(encoding="utf-8")
        self.assertIn("ARG MODEL_REFRESH=", dockerfile)
        self.assertIn("ARG WAN_HIGH_NOISE_MODEL_SHA256=", dockerfile)
        self.assertIn("ARG WAN_LOW_NOISE_MODEL_SHA256=", dockerfile)
        self.assertIn("${WAN_HIGH_NOISE_MODEL_SHA256}", dockerfile)
        self.assertIn("${WAN_LOW_NOISE_MODEL_SHA256}", dockerfile)
        self.assertEqual(2, dockerfile.count("sha256sum -c -"))

    def test_baked_model_destinations_match_every_workflow(self):
        expected_models = {
            "230": "wan2.2_i2v_high_noise_14B_fp8_scaled.safetensors",
            "235": "wan2.2_i2v_low_noise_14B_fp8_scaled.safetensors",
        }

        for workflow_path in sorted(WORKFLOW_DIR.glob("*.json")):
            with self.subTest(workflow=workflow_path.name):
                with workflow_path.open(encoding="utf-8") as workflow_file:
                    prompt = json.load(workflow_file)
                for node_id, model_name in expected_models.items():
                    self.assertEqual(
                        model_name,
                        prompt[node_id]["inputs"]["unet_name"],
                    )


if __name__ == "__main__":
    unittest.main()
