import json
import sys
import types
import unittest
from pathlib import Path
from unittest.mock import Mock

from generate_video_client import GenerateVideoClient
from lora_presets import PRESETS, resolve_lora_presets


mock_runpod = types.ModuleType("runpod")
mock_runpod.serverless = types.ModuleType("serverless")
mock_runpod.serverless.start = lambda config: None
mock_runpod.serverless.utils = types.ModuleType("utils")
mock_runpod.serverless.utils.rp_upload = lambda value: None
sys.modules.setdefault("runpod", mock_runpod)
sys.modules.setdefault("runpod.serverless", mock_runpod.serverless)
sys.modules.setdefault("runpod.serverless.utils", mock_runpod.serverless.utils)

from handler import apply_loras_to_workflow


WORKFLOW_DIR = Path(__file__).parent / "workflow"


class PresetTests(unittest.TestCase):
    def test_defaults_and_overrides(self):
        pairs = resolve_lora_presets({"lora_presets": [
            {"name": "assume_the_position"},
            {"name": "airblow", "high_weight": 0.5, "low_weight": 0.6},
        ]})
        self.assertEqual({**PRESETS["assume_the_position"], "high_weight": 0.8, "low_weight": 0.7}, pairs[0])
        self.assertEqual({**PRESETS["airblow"], "high_weight": 0.5, "low_weight": 0.6}, pairs[1])
        self.assertEqual([], resolve_lora_presets({}))

    def test_invalid_selections(self):
        bad_inputs = [
            {"lora_pairs": []},
            {"lora_presets": "airblow"},
            {"lora_presets": [{"name": "unknown"}]},
            {"lora_presets": [{"name": "airblow"}, {"name": "airblow"}]},
            {"lora_presets": [{"name": "airblow", "high_weight": True}]},
            {"lora_presets": [{"name": "airblow", "low_weight": float("nan")}]},
        ]
        for job_input in bad_inputs:
            with self.subTest(job_input=job_input), self.assertRaises(ValueError):
                resolve_lora_presets(job_input)
        with self.assertRaisesRegex(ValueError, "single-image"):
            resolve_lora_presets({"lora_presets": [{"name": "airblow"}]}, is_flf2v=True)

    def test_workflow_chain_for_each_selection(self):
        node_ids = {1: (["282"], ["336"]), 2: (["282", "339"], ["336", "285"])}
        for names in (("assume_the_position",), ("airblow",), ("airblow", "assume_the_position")):
            with self.subTest(names=names):
                pairs = resolve_lora_presets({"lora_presets": [{"name": name} for name in names]})
                workflow_path = WORKFLOW_DIR / f"wan22_{len(names)}lora.json"
                prompt = json.loads(workflow_path.read_text(encoding="utf-8"))
                apply_loras_to_workflow(prompt, pairs, f"workflow/{workflow_path.name}")
                high_nodes, low_nodes = node_ids[len(names)]
                for pair, high_node, low_node in zip(pairs, high_nodes, low_nodes):
                    self.assertEqual(pair["high"], prompt[high_node]["inputs"]["lora_name"])
                    self.assertEqual(pair["low"], prompt[low_node]["inputs"]["lora_name"])
                    self.assertEqual(pair["high_weight"], prompt[high_node]["inputs"]["strength_model"])
                    self.assertEqual(pair["low_weight"], prompt[low_node]["inputs"]["strength_model"])
                self.assertEqual(["283", 0], prompt[high_nodes[0]]["inputs"]["model"])
                self.assertEqual(["284", 0], prompt[low_nodes[0]]["inputs"]["model"])
                if len(names) == 2:
                    self.assertEqual([high_nodes[0], 0], prompt[high_nodes[1]]["inputs"]["model"])
                    self.assertEqual([low_nodes[0], 0], prompt[low_nodes[1]]["inputs"]["model"])

    def test_client_submits_presets(self):
        client = GenerateVideoClient("test-endpoint", "test-key")
        client.submit_job = Mock(return_value="test-job")
        client.wait_for_completion = Mock(return_value={"status": "COMPLETED"})
        presets = [{"name": "airblow"}]
        client.create_video_from_image(image=b"test-image", lora_presets=presets)
        submitted = client.submit_job.call_args.args[0]
        self.assertEqual(presets, submitted["lora_presets"])
        self.assertNotIn("lora_pairs", submitted)


if __name__ == "__main__":
    unittest.main()
