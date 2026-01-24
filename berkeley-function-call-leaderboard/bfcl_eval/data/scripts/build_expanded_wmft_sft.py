#!/usr/bin/env python3
"""
Build Expanded WMFT and SFT Datasets for BFCL (RC-GRPO Aligned)

Creates datasets aligned with RL training format used in verl/examples/bfcl:
- WMFT: ALL trajectories (success + failure) with binary reward conditioning
- SFT: Success-only trajectories, NO reward conditioning tokens

Key differences from previous `build_reward_conditioned_data.py`:
1. Binary reward tokens (high/low only) vs tertile (high/mid/low)
2. Trajectory-level reward (final score) vs per-turn rewards
3. Reward token in FIRST user message only (after env setup) vs every turn
4. Supports OpenAI format output (for LLaMA-Factory compatibility)

Format Options:
- Glaive: conversations with from/value, tools as JSON string
- OpenAI: messages with role/content/tool_calls, tools array

Usage:
    # Generate WMFT and SFT datasets
    python build_expanded_wmft_sft.py \
        --rl-trajectories ../generated/qwen2.5-7b-instruct/rl/rl_trajectories.json \
        --reward-results ../generated/evaluation/trajectory_reward_results.json \
        --output-dir ../generated/expanded \
        --format openai  # or glaive

    # Generate with train/test split
    python build_expanded_wmft_sft.py \
        --rl-trajectories ../generated/qwen2.5-7b-instruct/rl/rl_trajectories_train.json \
        --reward-results ../generated/evaluation/trajectory_reward_results.json \
        --output-dir ../generated/expanded \
        --format openai \
        --split --train-ratio 0.8
"""

import argparse
import copy
import json
import logging
import random
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# RC-GRPO: Binary reward conditioning tokens (matching unified agent loop)
HIGH_REWARD_TOKEN = "<|high_reward|>"
LOW_REWARD_TOKEN = "<|low_reward|>"

# RC-GRPO: Binary reward threshold (1.0 = success, < 1.0 = failure)
SUCCESS_THRESHOLD = 1.0


def get_binary_reward_token(final_reward: float) -> str:
    """
    Map trajectory-level reward to binary token.
    
    RC-GRPO uses binary classification:
    - reward >= 1.0 → high_reward (success)
    - reward < 1.0 → low_reward (failure)
    
    Args:
        final_reward: Final trajectory reward score
        
    Returns:
        Binary reward token string
    """
    if final_reward >= SUCCESS_THRESHOLD:
        return HIGH_REWARD_TOKEN
    else:
        return LOW_REWARD_TOKEN


def format_reward_prompt(token: str) -> str:
    """Format reward conditioning prompt suffix."""
    return f"\n\n[Reward Goal: {token}]"


def calculate_trajectory_reward(turn_rewards: List[Dict[str, Any]]) -> float:
    """
    Calculate final trajectory-level reward from turn rewards.
    
    Uses mean of normalized rewards across all turns, matching RL training.
    
    Args:
        turn_rewards: List of turn reward dicts from trajectory_reward_results.json
        
    Returns:
        Final trajectory reward in [0, 1]
    """
    if not turn_rewards:
        return 0.0
    
    # Calculate mean normalized reward across all turns
    rewards = [t.get("normalized_reward", 0.0) for t in turn_rewards]
    return sum(rewards) / len(rewards) if rewards else 0.0


def load_reward_results(results_path: Path) -> Tuple[Dict[int, float], Dict[int, List[Dict]]]:
    """
    Load reward results and compute trajectory-level rewards.
    
    Args:
        results_path: Path to trajectory_reward_results.json
        
    Returns:
        Tuple of (trajectory_rewards dict, turn_rewards dict) indexed by sample_idx
    """
    logger.info(f"Loading reward results from: {results_path}")
    
    with open(results_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    trajectory_rewards = {}
    turn_rewards_by_sample = {}
    
    for sample in data.get("samples", []):
        sample_idx = sample["sample_idx"]
        turns = sample.get("turns", [])
        
        # Store turn-level rewards for reference
        turn_rewards_by_sample[sample_idx] = turns
        
        # Calculate trajectory-level reward
        trajectory_rewards[sample_idx] = calculate_trajectory_reward(turns)
    
    logger.info(f"Loaded rewards for {len(trajectory_rewards)} samples")
    
    # Log reward distribution
    high_count = sum(1 for r in trajectory_rewards.values() if r >= SUCCESS_THRESHOLD)
    low_count = len(trajectory_rewards) - high_count
    logger.info(f"Reward distribution: {high_count} success (>={SUCCESS_THRESHOLD}), {low_count} failure")
    
    return trajectory_rewards, turn_rewards_by_sample


def find_first_user_message_index(conversations: List[Dict], skip_env_setup: bool = True) -> int:
    """
    Find the index of the first substantive user message.
    
    If skip_env_setup is True, skips the environment setup message (first human message).
    
    Args:
        conversations: List of conversation turns
        skip_env_setup: Whether to skip the first user message (env setup)
        
    Returns:
        Index of the first user message to inject reward token, or -1 if not found
    """
    user_message_count = 0
    for i, conv in enumerate(conversations):
        if conv.get("from") == "human":
            if skip_env_setup and user_message_count == 0:
                # Skip first user message (environment setup)
                user_message_count += 1
                continue
            return i
    return -1


def inject_reward_token_glaive(
    sample: Dict[str, Any],
    final_reward: float,
    skip_env_setup: bool = True
) -> Dict[str, Any]:
    """
    Inject binary reward token into FIRST user message (Glaive format).
    
    Args:
        sample: Glaive format sample with conversations
        final_reward: Trajectory-level final reward
        skip_env_setup: Whether to skip env setup message
        
    Returns:
        Modified sample with reward token
    """
    modified = copy.deepcopy(sample)
    conversations = modified.get("conversations", [])
    
    # Find first user message (after env setup if applicable)
    target_idx = find_first_user_message_index(conversations, skip_env_setup)
    
    if target_idx >= 0:
        reward_token = get_binary_reward_token(final_reward)
        reward_suffix = format_reward_prompt(reward_token)
        
        # Append to user message
        conversations[target_idx]["value"] = conversations[target_idx]["value"] + reward_suffix
    
    return modified


def remove_existing_reward_tokens(text: str) -> str:
    """Remove any existing reward conditioning tokens from text."""
    import re
    # Remove [Reward Goal: <|...|>] patterns
    return re.sub(r'\n\n\[Reward Goal: <\|[a-z_]+\|>\]', '', text)


def strip_reward_tokens_glaive(sample: Dict[str, Any]) -> Dict[str, Any]:
    """
    Remove all reward conditioning tokens from a Glaive sample.
    
    Args:
        sample: Glaive format sample
        
    Returns:
        Sample with all reward tokens removed
    """
    modified = copy.deepcopy(sample)
    conversations = modified.get("conversations", [])
    
    for conv in conversations:
        if conv.get("from") == "human" and "value" in conv:
            conv["value"] = remove_existing_reward_tokens(conv["value"])
    
    return modified


def convert_glaive_to_openai(sample: Dict[str, Any]) -> Dict[str, Any]:
    """
    Convert Glaive format to OpenAI format with proper message structure.
    
    Glaive format:
    - conversations: [{from: "human"|"gpt"|"function_call"|"observation", value: "..."}]
    - tools: JSON string
    
    OpenAI format:
    - messages: [{role: "system"|"user"|"assistant"|"tool", content: "...", tool_calls?: [...]}]
    - tools: Array of tool definitions
    
    Args:
        sample: Glaive format sample
        
    Returns:
        OpenAI format sample
    """
    messages = []
    conversations = sample.get("conversations", [])
    tools_json = sample.get("tools", "[]")
    
    # Parse tools from JSON string to array
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
    
    # Process conversations
    i = 0
    while i < len(conversations):
        conv = conversations[i]
        role_from = conv.get("from", "")
        value = conv.get("value", "")
        
        if role_from == "human":
            messages.append({
                "role": "user",
                "content": value
            })
            i += 1
            
        elif role_from == "gpt":
            messages.append({
                "role": "assistant",
                "content": value
            })
            i += 1
            
        elif role_from == "function_call":
            # Collect consecutive function_call/observation pairs
            tool_calls = []
            tool_results = []
            
            while i < len(conversations):
                curr = conversations[i]
                curr_role = curr.get("from", "")
                
                if curr_role == "function_call":
                    # Parse the function call JSON
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
                    # Match with previous tool call
                    tool_call_id = f"call_{len(tool_results)}"
                    tool_results.append({
                        "tool_call_id": tool_call_id,
                        "content": curr["value"]
                    })
                    i += 1
                    
                else:
                    break
            
            # Add assistant message with tool_calls
            if tool_calls:
                messages.append({
                    "role": "assistant",
                    "content": "",
                    "tool_calls": tool_calls
                })
            
            # Add tool responses
            for result in tool_results:
                messages.append({
                    "role": "tool",
                    "content": result["content"],
                    "tool_call_id": result["tool_call_id"]
                })
                
        elif role_from == "observation":
            # Standalone observation (shouldn't happen normally)
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
    """
    Fix message alternation for LLaMA-Factory compatibility.
    
    LLaMA-Factory expects strict alternation: user/tool → assistant → user/tool → ...
    
    Args:
        messages: List of OpenAI format messages
        
    Returns:
        Fixed messages with proper alternation
    """
    if not messages:
        return messages
    
    fixed = []
    
    # Extract system message if present
    if messages[0].get("role") == "system":
        fixed.append(messages[0])
        messages = messages[1:]
    
    for msg in messages:
        role = msg.get("role")
        
        # Map effective role for alternation checking
        if role == "assistant":
            effective = "assistant" if msg.get("tool_calls") else "assistant"
        elif role in ("user", "tool"):
            effective = "user"  # tool responses count as user-side
        else:
            effective = role
        
        if not fixed or fixed[-1].get("role") == "system":
            # First content message - should be user/tool
            if effective != "user":
                fixed.append({"role": "user", "content": "[Start]"})
            fixed.append(msg)
        else:
            # Check alternation
            last_role = fixed[-1].get("role")
            last_effective = "user" if last_role in ("user", "tool") else "assistant"
            
            if effective == last_effective:
                # Need to insert filler
                if effective == "assistant":
                    fixed.append({"role": "user", "content": "[Continue]"})
                else:
                    fixed.append({
                        "role": "assistant",
                        "content": "I understand.",
                        "tool_calls": []
                    })
            
            fixed.append(msg)
    
    # Ensure even count after system
    content_messages = fixed[1:] if fixed[0].get("role") == "system" else fixed
    start_idx = 1 if fixed[0].get("role") == "system" else 0
    
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


def process_samples(
    samples: List[Dict[str, Any]],
    trajectory_rewards: Dict[int, float],
    output_format: str = "glaive",
    add_reward_tokens: bool = True
) -> Tuple[List[Dict[str, Any]], Dict[str, int]]:
    """
    Process trajectory samples with optional reward token injection.
    
    Args:
        samples: List of Glaive format samples
        trajectory_rewards: Dict mapping sample_idx to trajectory reward
        output_format: "glaive" or "openai"
        add_reward_tokens: Whether to add reward conditioning tokens
        
    Returns:
        Tuple of (processed samples, statistics dict)
    """
    processed = []
    stats = {
        "total": 0,
        "success": 0,
        "failure": 0,
        "missing_reward": 0
    }
    
    for sample_idx, sample in enumerate(samples):
        stats["total"] += 1
        
        # Get trajectory reward
        final_reward = trajectory_rewards.get(sample_idx)
        
        if final_reward is None:
            stats["missing_reward"] += 1
            # Default to low reward for missing
            final_reward = 0.0
        
        # Track success/failure
        if final_reward >= SUCCESS_THRESHOLD:
            stats["success"] += 1
        else:
            stats["failure"] += 1
        
        # Process sample
        if add_reward_tokens:
            # Strip existing tokens first, then add new binary token
            clean_sample = strip_reward_tokens_glaive(sample)
            modified = inject_reward_token_glaive(clean_sample, final_reward)
        else:
            # SFT mode: just strip existing tokens
            modified = strip_reward_tokens_glaive(sample)
        
        # Convert format if needed
        if output_format == "openai":
            modified = convert_glaive_to_openai(modified)
            modified["messages"] = fix_message_alternation(modified["messages"])
            # Add metadata
            modified["final_reward"] = final_reward
            modified["sample_idx"] = sample_idx
        else:
            # Keep Glaive format, add metadata
            modified["final_reward"] = final_reward
            modified["sample_idx"] = sample_idx
        
        processed.append(modified)
    
    return processed, stats


def filter_success_only(
    samples: List[Dict[str, Any]],
    trajectory_rewards: Dict[int, float]
) -> List[Dict[str, Any]]:
    """
    Filter to only success trajectories (reward >= 1.0).
    
    Args:
        samples: List of samples
        trajectory_rewards: Dict mapping sample_idx to trajectory reward
        
    Returns:
        Filtered list of success-only samples
    """
    filtered = []
    for sample_idx, sample in enumerate(samples):
        reward = trajectory_rewards.get(sample_idx, 0.0)
        if reward >= SUCCESS_THRESHOLD:
            filtered.append((sample_idx, sample))
    
    return [(idx, s) for idx, s in filtered]


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Build expanded WMFT and SFT datasets aligned with RC-GRPO training"
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
        help="Output format: 'glaive' or 'openai' (default: openai)"
    )
    parser.add_argument(
        "--split",
        action="store_true",
        help="Generate train/test split"
    )
    parser.add_argument(
        "--train-ratio",
        type=float,
        default=0.8,
        help="Train split ratio (default: 0.8)"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility"
    )
    parser.add_argument(
        "--wmft-only",
        action="store_true",
        help="Only generate WMFT dataset (skip SFT)"
    )
    parser.add_argument(
        "--sft-only",
        action="store_true",
        help="Only generate SFT dataset (skip WMFT)"
    )
    
    args = parser.parse_args()
    
    # Set random seed
    random.seed(args.seed)
    
    # Load data
    rl_path = Path(args.rl_trajectories)
    rewards_path = Path(args.reward_results)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Validate inputs
    if not rl_path.exists():
        logger.error(f"RL trajectories file not found: {rl_path}")
        return 1
    if not rewards_path.exists():
        logger.error(f"Reward results file not found: {rewards_path}")
        return 1
    
    # Load reward results
    trajectory_rewards, turn_rewards = load_reward_results(rewards_path)
    
    # Load RL trajectories
    logger.info(f"Loading RL trajectories from: {rl_path}")
    with open(rl_path, 'r', encoding='utf-8') as f:
        rl_samples = json.load(f)
    logger.info(f"Loaded {len(rl_samples)} trajectory samples")
    
    # Process datasets
    if args.split:
        # Generate train/test split
        logger.info(f"Generating train/test split (ratio={args.train_ratio}, seed={args.seed})")
        
        # Shuffle indices
        indices = list(range(len(rl_samples)))
        random.shuffle(indices)
        
        # Split
        train_size = int(len(indices) * args.train_ratio)
        train_indices = set(indices[:train_size])
        test_indices = set(indices[train_size:])
        
        train_samples = [(i, rl_samples[i]) for i in train_indices]
        test_samples = [(i, rl_samples[i]) for i in test_indices]
        
        logger.info(f"Split: {len(train_samples)} train, {len(test_samples)} test")
        
        # Process train
        if not args.sft_only:
            logger.info("=" * 60)
            logger.info("Processing WMFT TRAIN dataset...")
            wmft_train, wmft_train_stats = process_samples(
                [s for _, s in train_samples],
                {new_idx: trajectory_rewards.get(orig_idx, 0.0) 
                 for new_idx, (orig_idx, _) in enumerate(train_samples)},
                output_format=args.format,
                add_reward_tokens=True
            )
            wmft_train_path = output_dir / f"expanded_wm_train.json"
            with open(wmft_train_path, 'w', encoding='utf-8') as f:
                json.dump(wmft_train, f, ensure_ascii=False, indent=2)
            logger.info(f"WMFT train: {wmft_train_stats}")
            logger.info(f"Saved to: {wmft_train_path}")
        
        if not args.sft_only:
            logger.info("=" * 60)
            logger.info("Processing WMFT TEST dataset...")
            wmft_test, wmft_test_stats = process_samples(
                [s for _, s in test_samples],
                {new_idx: trajectory_rewards.get(orig_idx, 0.0) 
                 for new_idx, (orig_idx, _) in enumerate(test_samples)},
                output_format=args.format,
                add_reward_tokens=True
            )
            wmft_test_path = output_dir / f"expanded_wm_test.json"
            with open(wmft_test_path, 'w', encoding='utf-8') as f:
                json.dump(wmft_test, f, ensure_ascii=False, indent=2)
            logger.info(f"WMFT test: {wmft_test_stats}")
            logger.info(f"Saved to: {wmft_test_path}")
        
        # Process SFT (success-only)
        if not args.wmft_only:
            logger.info("=" * 60)
            logger.info("Processing SFT TRAIN dataset (success-only)...")
            sft_train_filtered = [(new_idx, s) for new_idx, (orig_idx, s) in enumerate(train_samples)
                                  if trajectory_rewards.get(orig_idx, 0.0) >= SUCCESS_THRESHOLD]
            sft_train, sft_train_stats = process_samples(
                [s for _, s in sft_train_filtered],
                {i: SUCCESS_THRESHOLD for i in range(len(sft_train_filtered))},  # All success
                output_format=args.format,
                add_reward_tokens=False  # No reward tokens for SFT
            )
            sft_train_path = output_dir / f"expanded_sft_train.json"
            with open(sft_train_path, 'w', encoding='utf-8') as f:
                json.dump(sft_train, f, ensure_ascii=False, indent=2)
            logger.info(f"SFT train: {len(sft_train)} samples (success only)")
            logger.info(f"Saved to: {sft_train_path}")
        
        if not args.wmft_only:
            logger.info("=" * 60)
            logger.info("Processing SFT TEST dataset (success-only)...")
            sft_test_filtered = [(new_idx, s) for new_idx, (orig_idx, s) in enumerate(test_samples)
                                 if trajectory_rewards.get(orig_idx, 0.0) >= SUCCESS_THRESHOLD]
            sft_test, sft_test_stats = process_samples(
                [s for _, s in sft_test_filtered],
                {i: SUCCESS_THRESHOLD for i in range(len(sft_test_filtered))},
                output_format=args.format,
                add_reward_tokens=False
            )
            sft_test_path = output_dir / f"expanded_sft_test.json"
            with open(sft_test_path, 'w', encoding='utf-8') as f:
                json.dump(sft_test, f, ensure_ascii=False, indent=2)
            logger.info(f"SFT test: {len(sft_test)} samples (success only)")
            logger.info(f"Saved to: {sft_test_path}")
    
    else:
        # No split - process all samples
        if not args.sft_only:
            logger.info("=" * 60)
            logger.info("Processing WMFT dataset (all trajectories)...")
            wmft_all, wmft_stats = process_samples(
                rl_samples,
                trajectory_rewards,
                output_format=args.format,
                add_reward_tokens=True
            )
            wmft_path = output_dir / f"expanded_wm.json"
            with open(wmft_path, 'w', encoding='utf-8') as f:
                json.dump(wmft_all, f, ensure_ascii=False, indent=2)
            logger.info(f"WMFT: {wmft_stats}")
            logger.info(f"Saved to: {wmft_path}")
        
        if not args.wmft_only:
            logger.info("=" * 60)
            logger.info("Processing SFT dataset (success-only)...")
            # Filter to success only
            success_samples = [(i, s) for i, s in enumerate(rl_samples)
                               if trajectory_rewards.get(i, 0.0) >= SUCCESS_THRESHOLD]
            sft_all, sft_stats = process_samples(
                [s for _, s in success_samples],
                {new_idx: SUCCESS_THRESHOLD for new_idx in range(len(success_samples))},
                output_format=args.format,
                add_reward_tokens=False
            )
            sft_path = output_dir / f"expanded_sft.json"
            with open(sft_path, 'w', encoding='utf-8') as f:
                json.dump(sft_all, f, ensure_ascii=False, indent=2)
            logger.info(f"SFT: {len(sft_all)} samples (success only)")
            logger.info(f"Saved to: {sft_path}")
    
    # Final summary
    logger.info("=" * 60)
    logger.info("EXPANDED WMFT/SFT DATASET GENERATION COMPLETE")
    logger.info("=" * 60)
    logger.info(f"Output format: {args.format}")
    logger.info(f"Output directory: {output_dir}")
    logger.info(f"Binary reward threshold: {SUCCESS_THRESHOLD}")
    logger.info(f"Reward token placement: FIRST user message only")
    if args.split:
        logger.info(f"Train/test split: {args.train_ratio}/{1-args.train_ratio:.2f}")
    logger.info("=" * 60)
    
    return 0


if __name__ == "__main__":
    exit(main())
