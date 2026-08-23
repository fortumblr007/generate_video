#!/usr/bin/env python3
"""
LoRA chain test script.
Checks that LoRAs are wired into the model chain correctly.
"""

import json
import os
import sys
from typing import Dict, Any, List, Optional

# So handler.py can be imported from this directory
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Mock runpod so importing handler does not start the serverless worker
import types
mock_runpod = types.ModuleType('runpod')
mock_runpod.serverless = types.ModuleType('serverless')
mock_runpod.serverless.start = lambda x: None
mock_runpod.serverless.utils = types.ModuleType('utils')
mock_runpod.serverless.utils.rp_upload = lambda x: None

sys.modules['runpod'] = mock_runpod
sys.modules['runpod.serverless'] = mock_runpod.serverless
sys.modules['runpod.serverless.utils'] = mock_runpod.serverless.utils

from handler import (
    load_workflow,
    apply_lora_chain,
    get_next_available_node_id
)

def print_node_chain(prompt: Dict[str, Any], start_node_id: str, chain_type: str = ""):
    """Walk and print a LoRA node chain from start_node_id."""
    print(f"\n{'='*60}")
    print(f"{chain_type} LoRA chain (start node: {start_node_id})")
    print(f"{'='*60}")
    
    visited = set()
    current_id = start_node_id
    chain = []
    
    while current_id and current_id not in visited:
        visited.add(current_id)
        
        if current_id not in prompt:
            print(f"Node {current_id} not found.")
            break
        
        node = prompt[current_id]
        node_type = node.get("class_type", "Unknown")
        
        chain.append({
            "node_id": current_id,
            "type": node_type,
            "inputs": node.get("inputs", {})
        })
        
        if node_type == "LoraLoaderModelOnly":
            model_input = node.get("inputs", {}).get("model")
            if model_input and isinstance(model_input, list):
                next_id = str(model_input[0])
                lora_name = node.get("inputs", {}).get("lora_name", "N/A")
                strength = node.get("inputs", {}).get("strength_model", "N/A")
                print(f"  node {current_id}: {node_type}")
                print(f"    LoRA: {lora_name}")
                print(f"    strength: {strength}")
                print(f"    input: node {next_id}")
                current_id = next_id
            else:
                print(f"  node {current_id}: {node_type}")
                print(f"    LoRA: {node.get('inputs', {}).get('lora_name', 'N/A')}")
                print(f"    strength: {node.get('inputs', {}).get('strength_model', 'N/A')}")
                print(f"    input: none (end of chain)")
                break
        else:
            print(f"  node {current_id}: {node_type}")
            break
    
    print(f"\nChain length: {len(chain)} nodes")
    return chain

def test_lora_chain(test_name: str, lora_pairs: List[Dict[str, Any]], is_flf2v: bool = False):
    """Run one LoRA-chain case and dump before/after workflow JSON."""
    print(f"\n{'#'*80}")
    print(f"# Test: {test_name}")
    print(f"{'#'*80}")
    print(f"LoRA count: {len(lora_pairs)}")
    print(f"Workflow: {'FLF2V' if is_flf2v else 'single image'}")
    
    workflow_file = "/wan22_flf2v_api.json" if is_flf2v else "/wan22_api.json"
    workflow_path = os.path.join(os.path.dirname(__file__), workflow_file.lstrip("/"))
    
    if not os.path.exists(workflow_path):
        print(f"Workflow file not found: {workflow_path}")
        return None
    
    import copy
    prompt_before = load_workflow(workflow_path)
    prompt = copy.deepcopy(prompt_before)
    
    if is_flf2v:
        high_lora_node_id = "91"
        low_lora_node_id = "92"
        high_sampling_node_id = "54"
        low_sampling_node_id = "55"
    else:
        high_lora_node_id = "101"
        low_lora_node_id = "102"
        high_sampling_node_id = "104"
        low_sampling_node_id = "103"
    
    print(f"\n{'='*60}")
    print("State before applying LoRAs")
    print(f"{'='*60}")
    print(f"HIGH LoRA node ({high_lora_node_id}):")
    if high_lora_node_id in prompt:
        high_node = prompt[high_lora_node_id]
        print(f"  LoRA: {high_node.get('inputs', {}).get('lora_name', 'N/A')}")
        print(f"  strength: {high_node.get('inputs', {}).get('strength_model', 'N/A')}")
        print(f"  input: {high_node.get('inputs', {}).get('model', 'N/A')}")
    
    print(f"\nLOW LoRA node ({low_lora_node_id}):")
    if low_lora_node_id in prompt:
        low_node = prompt[low_lora_node_id]
        print(f"  LoRA: {low_node.get('inputs', {}).get('lora_name', 'N/A')}")
        print(f"  strength: {low_node.get('inputs', {}).get('strength_model', 'N/A')}")
        print(f"  input: {low_node.get('inputs', {}).get('model', 'N/A')}")
    
    if lora_pairs:
        try:
            apply_lora_chain(
                prompt,
                lora_pairs,
                high_lora_node_id,
                low_lora_node_id,
                high_sampling_node_id,
                low_sampling_node_id,
                is_flf2v
            )
        except Exception as e:
            print(f"Failed to apply LoRAs: {e}")
            import traceback
            traceback.print_exc()
            return None
    
    print(f"\n{'='*60}")
    print("State after applying LoRAs")
    print(f"{'='*60}")
    
    high_chain = print_node_chain(prompt, high_sampling_node_id, "HIGH")
    low_chain = print_node_chain(prompt, low_sampling_node_id, "LOW")
    
    before_file = f"test_before_{test_name.replace(' ', '_').lower()}.json"
    before_path = os.path.join(os.path.dirname(__file__), before_file)
    with open(before_path, 'w', encoding='utf-8') as f:
        json.dump(prompt_before, f, indent=2, ensure_ascii=False)
    
    after_file = f"test_after_{test_name.replace(' ', '_').lower()}.json"
    after_path = os.path.join(os.path.dirname(__file__), after_file)
    with open(after_path, 'w', encoding='utf-8') as f:
        json.dump(prompt, f, indent=2, ensure_ascii=False)
    
    print(f"\nSaved workflows:")
    print(f"   before: {before_path}")
    print(f"   after: {after_path}")
    
    print(f"\n{'='*60}")
    print("Key node comparison")
    print(f"{'='*60}")
    
    if is_flf2v:
        high_lora_node_id = "91"
        low_lora_node_id = "92"
        high_sampling_node_id = "54"
        low_sampling_node_id = "55"
    else:
        high_lora_node_id = "101"
        low_lora_node_id = "102"
        high_sampling_node_id = "104"
        low_sampling_node_id = "103"
    
    print(f"\nHIGH LoRA node ({high_lora_node_id}):")
    print(f"  before: {prompt_before.get(high_lora_node_id, {}).get('inputs', {}).get('model', 'N/A')}")
    print(f"  after: {prompt.get(high_lora_node_id, {}).get('inputs', {}).get('model', 'N/A')}")
    
    print(f"\nHIGH ModelSamplingSD3 node ({high_sampling_node_id}):")
    print(f"  before: {prompt_before.get(high_sampling_node_id, {}).get('inputs', {}).get('model', 'N/A')}")
    print(f"  after: {prompt.get(high_sampling_node_id, {}).get('inputs', {}).get('model', 'N/A')}")
    
    print(f"\nLOW LoRA node ({low_lora_node_id}):")
    print(f"  before: {prompt_before.get(low_lora_node_id, {}).get('inputs', {}).get('model', 'N/A')}")
    print(f"  after: {prompt.get(low_lora_node_id, {}).get('inputs', {}).get('model', 'N/A')}")
    
    print(f"\nLOW ModelSamplingSD3 node ({low_sampling_node_id}):")
    print(f"  before: {prompt_before.get(low_sampling_node_id, {}).get('inputs', {}).get('model', 'N/A')}")
    print(f"  after: {prompt.get(low_sampling_node_id, {}).get('inputs', {}).get('model', 'N/A')}")
    
    new_nodes = set(prompt.keys()) - set(prompt_before.keys())
    if new_nodes:
        print(f"\nNewly created nodes: {sorted(new_nodes, key=int)}")
        for node_id in sorted(new_nodes, key=int):
            node = prompt[node_id]
            if node.get("class_type") == "LoraLoaderModelOnly":
                print(f"  node {node_id}: {node.get('inputs', {}).get('lora_name', 'N/A')}")
                print(f"    input: {node.get('inputs', {}).get('model', 'N/A')}")
    
    return prompt

def main():
    """Run the LoRA chain cases."""
    print("="*80)
    print("Starting LoRA chain tests")
    print("="*80)
    
    test_lora_chain(
        "no LoRA",
        [],
        is_flf2v=False
    )
    
    test_lora_chain(
        "1 LoRA",
        [
            {
                "high": "test_lora1_high.safetensors",
                "low": "test_lora1_low.safetensors",
                "high_weight": 1.0,
                "low_weight": 1.0
            }
        ],
        is_flf2v=False
    )
    
    test_lora_chain(
        "2 LoRAs",
        [
            {
                "high": "test_lora1_high.safetensors",
                "low": "test_lora1_low.safetensors",
                "high_weight": 1.0,
                "low_weight": 1.0
            },
            {
                "high": "test_lora2_high.safetensors",
                "low": "test_lora2_low.safetensors",
                "high_weight": 0.8,
                "low_weight": 0.8
            }
        ],
        is_flf2v=False
    )
    
    test_lora_chain(
        "3 LoRAs",
        [
            {
                "high": "test_lora1_high.safetensors",
                "low": "test_lora1_low.safetensors",
                "high_weight": 1.0,
                "low_weight": 1.0
            },
            {
                "high": "test_lora2_high.safetensors",
                "low": "test_lora2_low.safetensors",
                "high_weight": 0.8,
                "low_weight": 0.8
            },
            {
                "high": "test_lora3_high.safetensors",
                "low": "test_lora3_low.safetensors",
                "high_weight": 0.5,
                "low_weight": 0.5
            }
        ],
        is_flf2v=False
    )
    
    test_lora_chain(
        "FLF2V 2 LoRAs",
        [
            {
                "high": "test_lora1_high.safetensors",
                "low": "test_lora1_low.safetensors",
                "high_weight": 1.0,
                "low_weight": 1.0
            },
            {
                "high": "test_lora2_high.safetensors",
                "low": "test_lora2_low.safetensors",
                "high_weight": 0.8,
                "low_weight": 0.8
            }
        ],
        is_flf2v=True
    )
    
    print("\n" + "="*80)
    print("All tests finished!")
    print("="*80)
    print("\nInspect the generated JSON files to confirm the LoRA chain:")
    print("  - test_before_*.json: workflow before applying LoRAs")
    print("  - test_after_*.json: workflow after applying LoRAs")
    print("\nHow to compare:")
    print("  1. Open the before/after files and compare node links")
    print("  2. Confirm new nodes are wired into the chain")
    print("  3. Confirm ModelSamplingSD3 points at the last LoRA")

if __name__ == "__main__":
    main()
