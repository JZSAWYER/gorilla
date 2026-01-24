#!/usr/bin/env python3
"""
Build Combined WMFT and SFT Datasets for BFCL (RC-GRPO Aligned)

This script combines:
1. Ground truth SFT data (expert trajectories with 100% success)
2. Model-generated RL trajectories (with varying success rates)

To create comprehensive datasets for:
- WMFT: World Model Fine-Tuning (all trajectories with binary RC tokens)
- SFT: Supervised Fine-Tuning (success-only, no RC tokens)

The key insight is that ground truth data provides high-quality success examples,
while model-generated trajectories provide diverse failure examples for WMFT.

Usage:
    python build_combined_wmft_sft.py \
        --gt-sft-train ../generated/sft/bfcl_glaive_dataset_train.json \
        --gt-sft-test ../generated/sft/bfcl_glaive_dataset_test.json \
        --rl-trajectories ../generated/qwen2.5-7b-instruct/rl/rl_trajectories.json \
        --reward-results ../generated/evaluation/trajectory_reward_results.json \
        --output-dir ../generated/combined
"""

import argparse
import copy
import json
import logging
import random
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# RC-GRPO: Binary reward conditioning tokens
HIGH_REWARD_TOKEN = "<|high_reward|>"
LOW_REWARD_TOKEN = "<|low_reward|>"

SUCCESS_THRESHOLD = 1.0


def get_binary_reward_token(final_reward: float) -> str:
    """Map trajectory-level reward to binary token."""
    return HIGH_REWARD_TOKEN if final_reward >= SUCCESS_THRESHOLD else LOW_REWARD_TOKEN


def format_reward_prompt(token: str) -> str:
    """Format reward conditioning prompt suffix."""
    return f"\n\n[Reward Goal: {token}]"


def remove_existing_reward_tokens(text: str) -> str:
    """Remove any existing reward conditioning tokens from text."""
    import re
    return re.sub(r'\n\n\[Reward Goal: <\|[a-z_]+\|>\]', '', text)


def find_first_user_message_index(conversations: List[Dict], skip_env_setup: bool = True) -> int:
    """Find the index of the first substantive user message."""
    user_message_count = 0
    for i, conv in enumerate(conversations):
        if conv.get("from") == "human":
            if skip_env_setup and user_message_count == 0:
                user_message_count += 1
                continue
            return i
    return -1


def inject_reward_token_glaive(
    sample: Dict[str, Any],
    final_reward: float,
    skip_env_setup: bool = True
) -> Dict[str, Any]:
    """Inject binary reward token into FIRST user message (Glaive format)."""
    modified = copy.deepcopy(sample)
    conversations = modified.get("conversations", [])
    
    target_idx = find_first_user_message_index(conversations, skip_env_setup)
    
    if target_idx >= 0:
        reward_token = get_binary_reward_token(final_reward)
        reward_suffix = format_reward_prompt(reward_token)
        
        # Remove existing tokens first
        conversations[target_idx]["value"] = remove_existing_reward_tokens(
            conversations[target_idx]["value"]
        )
        # Add new token
        conversations[target_idx]["value"] = conversations[target_idx]["value"] + reward_suffix
    
    return modified


def strip_reward_tokens_glaive(sample: Dict[str, Any]) -> Dict[str, Any]:
    """Remove all reward conditioning tokens from a Glaive sample."""
    modified = copy.deepcopy(sample)
    conversations = modified.get("conversations", [])
    
    for conv in conversations:
        if conv.get("from") == "human" and "value" in conv:
            conv["value"] = remove_existing_reward_tokens(conv["value"])
    
    return modified


def convert_glaive_to_openai(sample: Dict[str, Any]) -> Dict[str, Any]:
    """Convert Glaive format to OpenAI format."""
    messages = []
    conversations = sample.get("conversations", [])
    tools_json = sample.get("tools", "[]")
    
    try:
        tools = json.loads(tools_json) if isinstance(tools_json, str) else tools_json
    except json.JSONDecodeError:
        tools = []
    
    # Convert to OpenAI function format
    tools_openai = []
    for tool in tools:
        tools_openai.append({
            "type": "function",
            "function": {
                "name": tool.get("name", ""),
                "description": tool.get("description", ""),
                "parameters": tool.get("parameters", {})
            }
        })
    
    i = 0
    while i < len(conversations):
        conv = conversations[i]
        role_from = conv.get("from", "")
        value = conv.get("value", "")
        
        if role_from == "human":
            messages.append({"role": "user", "content": value})
            i += 1
            
        elif role_from == "gpt":
            messages.append({"role": "assistant", "content": value})
            i += 1
            
        elif role_from == "function_call":
            tool_calls = []
            tool_results = []
            
            while i < len(conversations):
                curr = conversations[i]
                curr_role = curr.get("from", "")
                
                if curr_role == "function_call":
                    try:
                        fc_data = json.loads(curr["value"])
                        tool_call_id = f"call_{len(tool_calls)}"
                        tool_calls.append({
                            "id": tool_call_id,
                            "type": "function",
                            "function": {
                                "name": fc_data.get("name", ""),
                                "arguments": json.dumps(fc_data.get("arguments", {}))
                            }
                        })
                    except json.JSONDecodeError:
                        pass
                    i += 1
                    
                elif curr_role == "observation":
                    tool_call_id = f"call_{len(tool_results)}"
                    tool_results.append({
                        "tool_call_id": tool_call_id,
                        "content": curr["value"]
                    })
                    i += 1
                else:
                    break
            
            if tool_calls:
                messages.append({
                    "role": "assistant",
                    "content": "",
                    "tool_calls": tool_calls
                })
            
            for result in tool_results:
                messages.append({
                    "role": "tool",
                    "content": result["content"],
                    "tool_call_id": result["tool_call_id"]
                })
                
        elif role_from == "observation":
            messages.append({
                "role": "tool",
                "content": value,
                "tool_call_id": "call_0"
            })
            i += 1
        else:
            i += 1
    
    return {
        "messages": messages,
        "tools": tools_openai
    }


def fix_message_alternation(messages: List[Dict]) -> List[Dict]:
    """Fix message alternation for LLaMA-Factory compatibility."""
    if not messages:
        return messages
    
    fixed = []
    
    if messages[0].get("role") == "system":
        fixed.append(messages[0])
        messages = messages[1:]
    
    for msg in messages:
        role = msg.get("role")
        effective = "user" if role in ("user", "tool") else "assistant"
        
        if not fixed or fixed[-1].get("role") == "system":
            if effective != "user":
                fixed.append({"role": "user", "content": "[Start]"})
            fixed.append(msg)
        else:
            last_role = fixed[-1].get("role")
            last_effective = "user" if last_role in ("user", "tool") else "assistant"
            
            if effective == last_effective:
                if effective == "assistant":
                    fixed.append({"role": "user", "content": "[Continue]"})
                else:
                    fixed.append({
                        "role": "assistant",
                        "content": "I understand.",
                        "tool_calls": []
                    })
            
            fixed.append(msg)
    
    content_messages = fixed[1:] if fixed[0].get("role") == "system" else fixed
    
    if len(content_messages) % 2 != 0:
        last_role = content_messages[-1].get("role")
        if last_role in ("user", "tool"):
            fixed.append({
                "role": "assistant",
                "content": "Task completed.",
                "tool_calls": []
            })
        else:
            fixed.append({"role": "user", "content": "[End]"})
    
    return fixed


def load_reward_results(results_path: Path) -> Dict[int, float]:
    """Load reward results and compute trajectory-level rewards."""
    logger.info(f"Loading reward results from: {results_path}")
    
    with open(results_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    trajectory_rewards = {}
    
    for sample in data.get("samples", []):
        sample_idx = sample["sample_idx"]
        turns = sample.get("turns", [])
        
        if turns:
            rewards = [t.get("normalized_reward", 0.0) for t in turns]
            trajectory_rewards[sample_idx] = sum(rewards) / len(rewards)
        else:
            trajectory_rewards[sample_idx] = 0.0
    
    logger.info(f"Loaded rewards for {len(trajectory_rewards)} samples")
    return trajectory_rewards


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Build combined WMFT and SFT datasets from GT-SFT and RL trajectories"
    )
    parser.add_argument(
        "--gt-sft-train",
        type=str,
        required=True,
        help="Path to ground truth SFT training data (Glaive format)"
    )
    parser.add_argument(
        "--gt-sft-test",
        type=str,
        required=True,
        help="Path to ground truth SFT test data (Glaive format)"
    )
    parser.add_argument(
        "--rl-trajectories",
        type=str,
        required=True,
        help="Path to RL trajectories file (Glaive format)"
    )
    parser.add_argument(
        "--reward-results",
        type=str,
        required=True,
        help="Path to trajectory_reward_results.json"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        required=True,
        help="Output directory for generated datasets"
    )
    parser.add_argument(
        "--format",
        type=str,
        choices=["glaive", "openai"],
        default="openai",
        help="Output format (default: openai)"
    )
    parser.add_argument(
        "--rl-train-ratio",
        type=float,
        default=0.8,
        help="Train split ratio for RL data (default: 0.8)"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility"
    )
    parser.add_argument(
        "--include-rl-success",
        action="store_true",
        help="Include successful RL trajectories in SFT (in addition to GT)"
    )
    
    args = parser.parse_args()
    random.seed(args.seed)
    
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load ground truth SFT data
    logger.info(f"Loading GT SFT train from: {args.gt_sft_train}")
    with open(args.gt_sft_train, 'r', encoding='utf-8') as f:
        gt_sft_train = json.load(f)
    logger.info(f"Loaded {len(gt_sft_train)} GT SFT train samples")
    
    logger.info(f"Loading GT SFT test from: {args.gt_sft_test}")
    with open(args.gt_sft_test, 'r', encoding='utf-8') as f:
        gt_sft_test = json.load(f)
    logger.info(f"Loaded {len(gt_sft_test)} GT SFT test samples")
    
    # Load RL trajectories
    logger.info(f"Loading RL trajectories from: {args.rl_trajectories}")
    with open(args.rl_trajectories, 'r', encoding='utf-8') as f:
        rl_samples = json.load(f)
    logger.info(f"Loaded {len(rl_samples)} RL trajectory samples")
    
    # Load reward results
    trajectory_rewards = load_reward_results(Path(args.reward_results))
    
    # Split RL data into train/test
    rl_indices = list(range(len(rl_samples)))
    random.shuffle(rl_indices)
    rl_train_size = int(len(rl_indices) * args.rl_train_ratio)
    rl_train_indices = set(rl_indices[:rl_train_size])
    rl_test_indices = set(rl_indices[rl_train_size:])
    
    logger.info(f"RL split: {len(rl_train_indices)} train, {len(rl_test_indices)} test")
    
    # Count success/failure in RL data
    rl_success_count = sum(1 for i in range(len(rl_samples)) 
                          if trajectory_rewards.get(i, 0.0) >= SUCCESS_THRESHOLD)
    rl_failure_count = len(rl_samples) - rl_success_count
    logger.info(f"RL reward distribution: {rl_success_count} success, {rl_failure_count} failure")
    
    # ============================================================
    # Build WMFT datasets (all trajectories with RC tokens)
    # ============================================================
    logger.info("=" * 60)
    logger.info("Building WMFT datasets...")
    
    wmft_train = []
    wmft_test = []
    
    # Add GT SFT data with high_reward token
    for sample in gt_sft_train:
        modified = inject_reward_token_glaive(sample, SUCCESS_THRESHOLD)
        if args.format == "openai":
            modified = convert_glaive_to_openai(modified)
            modified["messages"] = fix_message_alternation(modified["messages"])
        modified["source"] = "gt_sft"
        modified["final_reward"] = SUCCESS_THRESHOLD
        wmft_train.append(modified)
    
    for sample in gt_sft_test:
        modified = inject_reward_token_glaive(sample, SUCCESS_THRESHOLD)
        if args.format == "openai":
            modified = convert_glaive_to_openai(modified)
            modified["messages"] = fix_message_alternation(modified["messages"])
        modified["source"] = "gt_sft"
        modified["final_reward"] = SUCCESS_THRESHOLD
        wmft_test.append(modified)
    
    # Add RL trajectories with appropriate reward tokens
    for idx in rl_train_indices:
        sample = rl_samples[idx]
        reward = trajectory_rewards.get(idx, 0.0)
        modified = inject_reward_token_glaive(sample, reward)
        if args.format == "openai":
            modified = convert_glaive_to_openai(modified)
            modified["messages"] = fix_message_alternation(modified["messages"])
        modified["source"] = "rl"
        modified["final_reward"] = reward
        wmft_train.append(modified)
    
    for idx in rl_test_indices:
        sample = rl_samples[idx]
        reward = trajectory_rewards.get(idx, 0.0)
        modified = inject_reward_token_glaive(sample, reward)
        if args.format == "openai":
            modified = convert_glaive_to_openai(modified)
            modified["messages"] = fix_message_alternation(modified["messages"])
        modified["source"] = "rl"
        modified["final_reward"] = reward
        wmft_test.append(modified)
    
    # Shuffle
    random.shuffle(wmft_train)
    random.shuffle(wmft_test)
    
    # Count statistics
    wmft_train_success = sum(1 for s in wmft_train if s.get("final_reward", 0) >= SUCCESS_THRESHOLD)
    wmft_train_failure = len(wmft_train) - wmft_train_success
    wmft_test_success = sum(1 for s in wmft_test if s.get("final_reward", 0) >= SUCCESS_THRESHOLD)
    wmft_test_failure = len(wmft_test) - wmft_test_success
    
    logger.info(f"WMFT train: {len(wmft_train)} samples ({wmft_train_success} success, {wmft_train_failure} failure)")
    logger.info(f"WMFT test: {len(wmft_test)} samples ({wmft_test_success} success, {wmft_test_failure} failure)")
    
    # Save WMFT
    wmft_train_path = output_dir / "combined_wm_train.json"
    wmft_test_path = output_dir / "combined_wm_test.json"
    
    with open(wmft_train_path, 'w', encoding='utf-8') as f:
        json.dump(wmft_train, f, ensure_ascii=False, indent=2)
    logger.info(f"Saved WMFT train to: {wmft_train_path}")
    
    with open(wmft_test_path, 'w', encoding='utf-8') as f:
        json.dump(wmft_test, f, ensure_ascii=False, indent=2)
    logger.info(f"Saved WMFT test to: {wmft_test_path}")
    
    # ============================================================
    # Build SFT datasets (success-only, no RC tokens)
    # ============================================================
    logger.info("=" * 60)
    logger.info("Building SFT datasets (success-only, no RC tokens)...")
    
    sft_train = []
    sft_test = []
    
    # Add GT SFT data (all success, no RC tokens)
    for sample in gt_sft_train:
        modified = strip_reward_tokens_glaive(sample)
        if args.format == "openai":
            modified = convert_glaive_to_openai(modified)
            modified["messages"] = fix_message_alternation(modified["messages"])
        modified["source"] = "gt_sft"
        sft_train.append(modified)
    
    for sample in gt_sft_test:
        modified = strip_reward_tokens_glaive(sample)
        if args.format == "openai":
            modified = convert_glaive_to_openai(modified)
            modified["messages"] = fix_message_alternation(modified["messages"])
        modified["source"] = "gt_sft"
        sft_test.append(modified)
    
    # Optionally add successful RL trajectories
    if args.include_rl_success:
        for idx in rl_train_indices:
            reward = trajectory_rewards.get(idx, 0.0)
            if reward >= SUCCESS_THRESHOLD:
                sample = rl_samples[idx]
                modified = strip_reward_tokens_glaive(sample)
                if args.format == "openai":
                    modified = convert_glaive_to_openai(modified)
                    modified["messages"] = fix_message_alternation(modified["messages"])
                modified["source"] = "rl_success"
                sft_train.append(modified)
        
        for idx in rl_test_indices:
            reward = trajectory_rewards.get(idx, 0.0)
            if reward >= SUCCESS_THRESHOLD:
                sample = rl_samples[idx]
                modified = strip_reward_tokens_glaive(sample)
                if args.format == "openai":
                    modified = convert_glaive_to_openai(modified)
                    modified["messages"] = fix_message_alternation(modified["messages"])
                modified["source"] = "rl_success"
                sft_test.append(modified)
    
    # Shuffle
    random.shuffle(sft_train)
    random.shuffle(sft_test)
    
    logger.info(f"SFT train: {len(sft_train)} samples (all success)")
    logger.info(f"SFT test: {len(sft_test)} samples (all success)")
    
    # Save SFT
    sft_train_path = output_dir / "combined_sft_train.json"
    sft_test_path = output_dir / "combined_sft_test.json"
    
    with open(sft_train_path, 'w', encoding='utf-8') as f:
        json.dump(sft_train, f, ensure_ascii=False, indent=2)
    logger.info(f"Saved SFT train to: {sft_train_path}")
    
    with open(sft_test_path, 'w', encoding='utf-8') as f:
        json.dump(sft_test, f, ensure_ascii=False, indent=2)
    logger.info(f"Saved SFT test to: {sft_test_path}")
    
    # ============================================================
    # Final summary
    # ============================================================
    logger.info("=" * 60)
    logger.info("COMBINED WMFT/SFT DATASET GENERATION COMPLETE")
    logger.info("=" * 60)
    logger.info(f"Output format: {args.format}")
    logger.info(f"Output directory: {output_dir}")
    logger.info("")
    logger.info("WMFT (World Model Fine-Tuning):")
    logger.info(f"  Train: {len(wmft_train)} ({wmft_train_success} success / {wmft_train_failure} failure)")
    logger.info(f"  Test:  {len(wmft_test)} ({wmft_test_success} success / {wmft_test_failure} failure)")
    logger.info(f"  Success rate: {(wmft_train_success + wmft_test_success) / (len(wmft_train) + len(wmft_test)) * 100:.1f}%")
    logger.info("")
    logger.info("SFT (Supervised Fine-Tuning):")
    logger.info(f"  Train: {len(sft_train)} (all success)")
    logger.info(f"  Test:  {len(sft_test)} (all success)")
    logger.info("")
    logger.info("Files created:")
    logger.info(f"  {wmft_train_path}")
    logger.info(f"  {wmft_test_path}")
    logger.info(f"  {sft_train_path}")
    logger.info(f"  {sft_test_path}")
    logger.info("=" * 60)
    
    return 0


if __name__ == "__main__":
    exit(main())
