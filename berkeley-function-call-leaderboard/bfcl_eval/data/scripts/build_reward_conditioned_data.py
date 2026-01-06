#!/usr/bin/env python3
"""
Build Reward-Conditioned Training Data

Combines RL trajectories with SFT training data, appending reward conditioning
tokens to each turn based on computed reward scores.

Uses tertile thresholds (0.33 / 0.67):
- <|low_reward|>: reward < 0.33
- <|mid_reward|>: 0.33 <= reward < 0.67  
- <|high_reward|>: reward >= 0.67

Usage:
    python build_reward_conditioned_data.py \
        --rl-trajectories ../generated/rl/rl_trajectories.json \
        --reward-results ../generated/evaluation/trajectory_reward_results.json \
        --sft-data ../generated/sft/bfcl_glaive_dataset_train.json \
        --output ../generated/rl/rl_conditioned_combined.json
"""

import argparse
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

# Reward conditioning tokens (matching build_bfcl_sft.py)
HIGH_REWARD_TOKEN = "<|high_reward|>"
MID_REWARD_TOKEN = "<|mid_reward|>"
LOW_REWARD_TOKEN = "<|low_reward|>"

# Tertile thresholds
LOW_THRESHOLD = 0.33
HIGH_THRESHOLD = 0.67


def get_reward_prompt(reward_level: str) -> str:
    """
    Generate reward-conditioned prompt suffix for user messages.
    
    Args:
        reward_level: One of "high", "mid", or "low"
        
    Returns:
        Formatted reward prompt string to append to user messages
    """
    token = {
        "high": HIGH_REWARD_TOKEN,
        "mid": MID_REWARD_TOKEN,
        "low": LOW_REWARD_TOKEN
    }.get(reward_level, HIGH_REWARD_TOKEN)
    
    return f"\n\n[Reward Goal: {token}]"


def classify_reward(reward: float) -> str:
    """
    Classify a reward score into low/mid/high category using tertile thresholds.
    
    Args:
        reward: Normalized reward score in [0, 1]
        
    Returns:
        One of "low", "mid", or "high"
    """
    if reward < LOW_THRESHOLD:
        return "low"
    elif reward < HIGH_THRESHOLD:
        return "mid"
    else:
        return "high"


def load_reward_results(results_path: Path) -> Dict[int, List[Dict[str, Any]]]:
    """
    Load reward results and index by sample.
    
    Args:
        results_path: Path to trajectory_reward_results.json
        
    Returns:
        Dictionary mapping sample_idx to list of turn results
    """
    logger.info(f"Loading reward results from: {results_path}")
    
    with open(results_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # Index by sample_idx
    results_by_sample = {}
    for sample in data.get("samples", []):
        sample_idx = sample["sample_idx"]
        results_by_sample[sample_idx] = sample["turns"]
    
    logger.info(f"Loaded rewards for {len(results_by_sample)} samples")
    return results_by_sample


def add_reward_tokens_to_sample(
    sample: Dict[str, Any],
    turn_rewards: List[Dict[str, Any]]
) -> Tuple[Dict[str, Any], Dict[str, int]]:
    """
    Add reward conditioning tokens to human messages in a sample.
    
    Skips the first human message (environment setup, turn 0).
    
    Args:
        sample: Original sample with conversations
        turn_rewards: List of turn reward results from evaluation
        
    Returns:
        Tuple of (modified sample, stats dict with counts)
    """
    import copy
    modified_sample = copy.deepcopy(sample)
    conversations = modified_sample.get("conversations", [])
    
    stats = {"low": 0, "mid": 0, "high": 0, "skipped": 0}
    
    # Track which turn index we're on (turns start at each human message)
    turn_idx = 0
    
    for conv in conversations:
        if conv.get("from") == "human":
            if turn_idx == 0:
                # Skip environment setup (first human message)
                stats["skipped"] += 1
                turn_idx += 1
                continue
            
            # Get reward for this turn
            # turn_idx in reward results corresponds to: 0=env setup, 1=first task, etc.
            if turn_idx < len(turn_rewards):
                reward = turn_rewards[turn_idx].get("normalized_reward", 0.0)
            else:
                # Fallback if turn_rewards doesn't have this turn
                logger.warning(f"Missing reward for turn {turn_idx}, defaulting to 0.0")
                reward = 0.0
            
            # Classify and append token
            reward_level = classify_reward(reward)
            reward_suffix = get_reward_prompt(reward_level)
            
            # Append to the human message value
            conv["value"] = conv["value"] + reward_suffix
            
            stats[reward_level] += 1
            turn_idx += 1
    
    return modified_sample, stats


def process_rl_trajectories(
    rl_path: Path,
    rewards_by_sample: Dict[int, List[Dict[str, Any]]]
) -> Tuple[List[Dict[str, Any]], Dict[str, int]]:
    """
    Process all RL trajectories and add reward conditioning tokens.
    
    Args:
        rl_path: Path to rl_trajectories.json
        rewards_by_sample: Dictionary of reward results indexed by sample
        
    Returns:
        Tuple of (list of modified samples, aggregate stats)
    """
    logger.info(f"Loading RL trajectories from: {rl_path}")
    
    with open(rl_path, 'r', encoding='utf-8') as f:
        rl_data = json.load(f)
    
    logger.info(f"Processing {len(rl_data)} RL samples")
    
    modified_samples = []
    total_stats = {"low": 0, "mid": 0, "high": 0, "skipped": 0, "missing_rewards": 0}
    
    for sample_idx, sample in enumerate(rl_data):
        if sample_idx in rewards_by_sample:
            turn_rewards = rewards_by_sample[sample_idx]
        else:
            logger.warning(f"No reward data for sample {sample_idx}")
            total_stats["missing_rewards"] += 1
            # Still process but with empty rewards
            turn_rewards = []
        
        modified_sample, stats = add_reward_tokens_to_sample(sample, turn_rewards)
        modified_samples.append(modified_sample)
        
        # Aggregate stats
        for key in ["low", "mid", "high", "skipped"]:
            total_stats[key] += stats[key]
    
    return modified_samples, total_stats


def load_sft_data(sft_path: Path) -> List[Dict[str, Any]]:
    """
    Load SFT training data (already has reward tokens).
    
    Args:
        sft_path: Path to bfcl_glaive_dataset_train.json
        
    Returns:
        List of SFT samples
    """
    logger.info(f"Loading SFT data from: {sft_path}")
    
    with open(sft_path, 'r', encoding='utf-8') as f:
        sft_data = json.load(f)
    
    logger.info(f"Loaded {len(sft_data)} SFT samples (with existing high_reward tokens)")
    return sft_data


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Build reward-conditioned training data by combining RL trajectories with SFT data"
    )
    parser.add_argument(
        "--rl-trajectories",
        type=str,
        default="../generated/rl/rl_trajectories.json",
        help="Path to RL trajectories file"
    )
    parser.add_argument(
        "--reward-results",
        type=str,
        default="../generated/evaluation/trajectory_reward_results.json",
        help="Path to trajectory reward results file"
    )
    parser.add_argument(
        "--sft-data",
        type=str,
        default="../generated/sft/bfcl_glaive_dataset_train.json",
        help="Path to SFT training data file"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="../generated/rl/rl_conditioned_combined.json",
        help="Path to output combined file"
    )
    parser.add_argument(
        "--low-threshold",
        type=float,
        default=0.33,
        help="Threshold below which rewards are classified as 'low' (default: 0.33)"
    )
    parser.add_argument(
        "--high-threshold",
        type=float,
        default=0.67,
        help="Threshold at or above which rewards are classified as 'high' (default: 0.67)"
    )
    parser.add_argument(
        "--rl-only",
        action="store_true",
        help="Only output RL data with reward tokens, don't combine with SFT"
    )
    parser.add_argument(
        "--shuffle",
        action="store_true",
        default=True,
        help="Shuffle the final output (default: True)"
    )
    parser.add_argument(
        "--no-shuffle",
        action="store_true",
        help="Disable shuffling of the final output"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for shuffling (default: 42)"
    )
    
    args = parser.parse_args()
    
    # Update global thresholds
    global LOW_THRESHOLD, HIGH_THRESHOLD
    LOW_THRESHOLD = args.low_threshold
    HIGH_THRESHOLD = args.high_threshold
    
    logger.info(f"Using thresholds: low < {LOW_THRESHOLD}, mid < {HIGH_THRESHOLD}, high >= {HIGH_THRESHOLD}")
    
    rl_path = Path(args.rl_trajectories)
    rewards_path = Path(args.reward_results)
    sft_path = Path(args.sft_data)
    output_path = Path(args.output)
    
    # Validate input files
    if not rl_path.exists():
        logger.error(f"RL trajectories file not found: {rl_path}")
        return 1
    if not rewards_path.exists():
        logger.error(f"Reward results file not found: {rewards_path}")
        return 1
    if not args.rl_only and not sft_path.exists():
        logger.error(f"SFT data file not found: {sft_path}")
        return 1
    
    # Load reward results
    rewards_by_sample = load_reward_results(rewards_path)
    
    # Process RL trajectories
    rl_samples, rl_stats = process_rl_trajectories(rl_path, rewards_by_sample)
    
    # Combine with SFT data
    if args.rl_only:
        combined_data = rl_samples
        logger.info("RL-only mode: not combining with SFT data")
    else:
        sft_samples = load_sft_data(sft_path)
        combined_data = rl_samples + sft_samples
        logger.info(f"Combined {len(rl_samples)} RL samples + {len(sft_samples)} SFT samples = {len(combined_data)} total")
    
    # Shuffle the combined data
    should_shuffle = args.shuffle and not args.no_shuffle
    if should_shuffle:
        random.seed(args.seed)
        random.shuffle(combined_data)
        logger.info(f"Shuffled {len(combined_data)} samples (seed={args.seed})")
    
    # Save output
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(combined_data, f, ensure_ascii=False, indent=2)
    
    logger.info(f"Output saved to: {output_path}")
    
    # Print summary
    logger.info("=" * 60)
    logger.info("Processing Summary")
    logger.info("=" * 60)
    logger.info(f"RL samples processed: {len(rl_samples)}")
    logger.info(f"Reward token distribution (turn-level):")
    logger.info(f"  - Low (<{LOW_THRESHOLD}): {rl_stats['low']}")
    logger.info(f"  - Mid ({LOW_THRESHOLD}-{HIGH_THRESHOLD}): {rl_stats['mid']}")
    logger.info(f"  - High (>={HIGH_THRESHOLD}): {rl_stats['high']}")
    logger.info(f"  - Skipped (env setup): {rl_stats['skipped']}")
    if rl_stats['missing_rewards'] > 0:
        logger.warning(f"  - Samples with missing rewards: {rl_stats['missing_rewards']}")
    
    total_tokens = rl_stats['low'] + rl_stats['mid'] + rl_stats['high']
    if total_tokens > 0:
        logger.info(f"Percentages:")
        logger.info(f"  - Low: {rl_stats['low']/total_tokens*100:.1f}%")
        logger.info(f"  - Mid: {rl_stats['mid']/total_tokens*100:.1f}%")
        logger.info(f"  - High: {rl_stats['high']/total_tokens*100:.1f}%")
    
    if not args.rl_only:
        logger.info(f"SFT samples (all high_reward): {len(sft_samples)}")
    logger.info(f"Total output samples: {len(combined_data)}")
    logger.info(f"Shuffled: {should_shuffle}" + (f" (seed={args.seed})" if should_shuffle else ""))
    logger.info("=" * 60)
    
    return 0


if __name__ == "__main__":
    exit(main())
