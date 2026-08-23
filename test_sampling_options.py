import copy
import json
import random
import unittest
from pathlib import Path
from unittest.mock import Mock

from generate_video_client import GenerateVideoClient
from workflow_options import (
    DEFAULT_HIGH_LORA_STRENGTH,
    DEFAULT_LOW_LORA_STRENGTH,
    MAX_SEED,
    SAMPLING_NODES,
    bypass_torch_compile,
    configure_lightx2v_strengths,
    configure_sampling,
    get_cfg,
    get_lora_strength,
    get_steps,
    resolve_seed,
    split_steps,
)


WORKFLOW_DIR = Path(__file__).parent / "workflow"


class SplitStepsTests(unittest.TestCase):
    def test_halves_even_and_odd_counts(self):
        self.assertEqual(2, split_steps(4))
        self.assertEqual(4, split_steps(8))
        self.assertEqual(3, split_steps(7))


class ResolveSeedTests(unittest.TestCase):
    def test_minus_one_and_none_randomize(self):
        rng = random.Random(0)
        first = resolve_seed(-1, rng=rng)
        rng = random.Random(0)
        second = resolve_seed(None, rng=rng)
        self.assertEqual(first, second)
        self.assertIsInstance(first, int)
        self.assertGreaterEqual(first, 0)
        self.assertLessEqual(first, MAX_SEED)

    def test_pinned_seed_is_unchanged(self):
        self.assertEqual(42, resolve_seed(42))
        self.assertEqual(0, resolve_seed(0))

    def test_rejects_invalid_seeds(self):
        with self.assertRaisesRegex(ValueError, "integer"):
            resolve_seed("42")
        with self.assertRaisesRegex(ValueError, "integer"):
            resolve_seed(True)
        with self.assertRaisesRegex(ValueError, "-1"):
            resolve_seed(-2)


class SamplingInputTests(unittest.TestCase):
    def test_omitted_job_input_uses_defaults(self):
        self.assertEqual(4, get_steps({}))
        self.assertEqual(1.0, get_cfg({}))
        self.assertEqual(2, split_steps(get_steps({})))

    def test_rejects_invalid_steps_and_cfg(self):
        with self.assertRaisesRegex(ValueError, "integer"):
            get_steps({"steps": "10"})
        with self.assertRaisesRegex(ValueError, ">= 2"):
            get_steps({"steps": 1})
        with self.assertRaisesRegex(ValueError, "integer"):
            get_steps({"steps": True})
        with self.assertRaisesRegex(ValueError, ">= 0"):
            get_cfg({"cfg": -0.1})
        with self.assertRaisesRegex(ValueError, "number"):
            get_cfg({"cfg": True})


class ConfigureSamplingTests(unittest.TestCase):
    def test_writes_widgets_on_nolora_workflow(self):
        workflow_path = WORKFLOW_DIR / "wan22_nolora.json"
        with workflow_path.open(encoding="utf-8") as workflow_file:
            prompt = json.load(workflow_file)

        split = configure_sampling(prompt, seed=7, steps=8, cfg=1.5)

        self.assertEqual(4, split)
        self.assertEqual(7, prompt["835"]["inputs"]["noise_seed"])
        self.assertEqual(8, prompt["834"]["inputs"]["steps"])
        self.assertEqual(4, prompt["829"]["inputs"]["step"])
        self.assertEqual(1.5, prompt["830"]["inputs"]["cfg"])

    def test_all_baked_workflows_have_sampling_nodes(self):
        workflow_paths = sorted(WORKFLOW_DIR.glob("*.json"))
        self.assertEqual(6, len(workflow_paths))

        for workflow_path in workflow_paths:
            with self.subTest(workflow=workflow_path.name):
                with workflow_path.open(encoding="utf-8") as workflow_file:
                    prompt = json.load(workflow_file)
                for node_id, class_type, field in SAMPLING_NODES:
                    node = prompt[node_id]
                    self.assertEqual(class_type, node["class_type"])
                    self.assertIn(field, node["inputs"])
                configure_sampling(copy.deepcopy(prompt), 1, 4, 1.0)

    def test_missing_node_fails_loudly(self):
        with self.assertRaisesRegex(ValueError, "RandomNoise"):
            configure_sampling({}, 1, 4, 1.0)


class ClientSamplingTests(unittest.TestCase):
    def test_python_client_sends_sampling_options(self):
        client = GenerateVideoClient("test-endpoint", "test-key")
        client.submit_job = Mock(return_value="test-job")
        client.wait_for_completion = Mock(return_value={"status": "COMPLETED"})

        result = client.create_video_from_image(
            image=b"test-image",
            prompt="test prompt",
            seed=7,
            steps=8,
            cfg=1.5,
        )

        self.assertEqual({"status": "COMPLETED"}, result)
        submitted_input = client.submit_job.call_args.args[0]
        self.assertEqual(7, submitted_input["seed"])
        self.assertEqual(8, submitted_input["steps"])
        self.assertEqual(1.5, submitted_input["cfg"])
        self.assertEqual(0.4, submitted_input["high_lora_strength"])
        self.assertEqual(1.0, submitted_input["low_lora_strength"])

    def test_python_client_sends_lightx2v_strengths(self):
        client = GenerateVideoClient("test-endpoint", "test-key")
        client.submit_job = Mock(return_value="test-job")
        client.wait_for_completion = Mock(return_value={"status": "COMPLETED"})

        client.create_video_from_image(
            image=b"test-image",
            high_lora_strength=0.2,
            low_lora_strength=0.8,
            cfg=3.0,
        )

        submitted_input = client.submit_job.call_args.args[0]
        self.assertEqual(0.2, submitted_input["high_lora_strength"])
        self.assertEqual(0.8, submitted_input["low_lora_strength"])
        self.assertEqual(3.0, submitted_input["cfg"])


class LightX2VStrengthTests(unittest.TestCase):
    def test_omitted_input_uses_defaults(self):
        self.assertEqual(
            0.4,
            get_lora_strength({}, "high_lora_strength", DEFAULT_HIGH_LORA_STRENGTH),
        )
        self.assertEqual(
            1.0,
            get_lora_strength({}, "low_lora_strength", DEFAULT_LOW_LORA_STRENGTH),
        )

    def test_rejects_non_numbers(self):
        with self.assertRaisesRegex(ValueError, "high_lora_strength"):
            get_lora_strength({"high_lora_strength": "0.4"}, "high_lora_strength", 0.4)
        with self.assertRaisesRegex(ValueError, "low_lora_strength"):
            get_lora_strength({"low_lora_strength": True}, "low_lora_strength", 1.0)

    def test_allows_zero_and_values_outside_zero_one(self):
        self.assertEqual(
            0.0,
            get_lora_strength({"high_lora_strength": 0}, "high_lora_strength", 0.4),
        )
        self.assertEqual(
            1.5,
            get_lora_strength({"low_lora_strength": 1.5}, "low_lora_strength", 1.0),
        )

    def test_writes_widgets_and_does_not_change_cfg(self):
        workflow_path = WORKFLOW_DIR / "wan22_nolora.json"
        with workflow_path.open(encoding="utf-8") as workflow_file:
            prompt = json.load(workflow_file)

        original_cfg = prompt["830"]["inputs"]["cfg"]
        configure_lightx2v_strengths(prompt, 0.2, 0.8)

        self.assertEqual(0.2, prompt["283"]["inputs"]["strength_model"])
        self.assertEqual(0.8, prompt["284"]["inputs"]["strength_model"])
        self.assertEqual(original_cfg, prompt["830"]["inputs"]["cfg"])
        self.assertEqual(3.0, get_cfg({"cfg": 3.0}))

    def test_all_baked_workflows_have_lightx2v_nodes(self):
        workflow_paths = sorted(WORKFLOW_DIR.glob("*.json"))
        self.assertEqual(6, len(workflow_paths))

        for workflow_path in workflow_paths:
            with self.subTest(workflow=workflow_path.name):
                with workflow_path.open(encoding="utf-8") as workflow_file:
                    prompt = json.load(workflow_file)
                self.assertEqual("LoraLoaderModelOnly", prompt["283"]["class_type"])
                self.assertEqual("LoraLoaderModelOnly", prompt["284"]["class_type"])
                configure_lightx2v_strengths(copy.deepcopy(prompt), 0.4, 1.0)


class BypassTorchCompileTests(unittest.TestCase):
    def _consumer_links(self, prompt, node_id):
        links = []
        for other_id, node in prompt.items():
            if other_id == node_id or not isinstance(node, dict):
                continue
            if node.get("class_type") == "TorchCompileModelWanVideoV2":
                continue
            for key, value in (node.get("inputs") or {}).items():
                if (
                    isinstance(value, list)
                    and value
                    and str(value[0]) == str(node_id)
                ):
                    links.append((other_id, key, value))
        return links

    def test_rewires_sage_to_compile_upstream_on_nolora(self):
        workflow_path = WORKFLOW_DIR / "wan22_nolora.json"
        with workflow_path.open(encoding="utf-8") as workflow_file:
            prompt = json.load(workflow_file)

        high_src = list(prompt["391"]["inputs"]["model"])
        low_src = list(prompt["390"]["inputs"]["model"])
        self.assertEqual(["283", 0], high_src)
        self.assertEqual(["284", 0], low_src)

        bypassed = bypass_torch_compile(prompt)

        self.assertEqual(2, bypassed)
        self.assertEqual(high_src, prompt["392"]["inputs"]["model"])
        self.assertEqual(low_src, prompt["393"]["inputs"]["model"])
        self.assertEqual([], self._consumer_links(prompt, "390"))
        self.assertEqual([], self._consumer_links(prompt, "391"))

    def test_follows_last_lora_on_1lora_workflow(self):
        workflow_path = WORKFLOW_DIR / "wan22_1lora.json"
        with workflow_path.open(encoding="utf-8") as workflow_file:
            prompt = json.load(workflow_file)

        high_src = list(prompt["391"]["inputs"]["model"])
        low_src = list(prompt["390"]["inputs"]["model"])
        self.assertEqual(["282", 0], high_src)
        self.assertEqual(["336", 0], low_src)

        bypass_torch_compile(prompt)

        self.assertEqual(high_src, prompt["392"]["inputs"]["model"])
        self.assertEqual(low_src, prompt["393"]["inputs"]["model"])

    def test_all_baked_workflows_bypass_compile_consumers(self):
        workflow_paths = sorted(WORKFLOW_DIR.glob("*.json"))
        self.assertEqual(6, len(workflow_paths))

        for workflow_path in workflow_paths:
            with self.subTest(workflow=workflow_path.name):
                with workflow_path.open(encoding="utf-8") as workflow_file:
                    prompt = json.load(workflow_file)
                compile_ids = [
                    node_id
                    for node_id, node in prompt.items()
                    if node.get("class_type") == "TorchCompileModelWanVideoV2"
                ]
                self.assertEqual(2, len(compile_ids))
                bypass_torch_compile(prompt)
                for compile_id in compile_ids:
                    self.assertEqual([], self._consumer_links(prompt, compile_id))

    def test_missing_compile_node_fails_loudly(self):
        with self.assertRaisesRegex(ValueError, "TorchCompileModelWanVideoV2"):
            bypass_torch_compile({})


if __name__ == "__main__":
    unittest.main()

