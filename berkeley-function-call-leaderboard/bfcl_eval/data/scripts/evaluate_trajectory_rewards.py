#!/usr/bin/env python3
"""
Evaluate Trajectory Rewards

Compares model-generated trajectories against ground truth from a reference file,
calculates normalized per-turn rewards using the ToolRL formula, stores results
in JSON, and plots a histogram showing reward distribution from 0 to 1.

Usage:
    python evaluate_trajectory_rewards.py \
        --reference bfcl_glaive_dataset.json \
        --input rl_trajectories.json \
        --output trajectory_reward_results.json \
        --plot reward_histogram.png
"""

import argparse
import json
import logging
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# Reward conditioning token pattern to strip from reference prompts
REWARD_GOAL_PATTERN = re.compile(r'\s*\[Reward Goal: <\|(?:high|mid|low)_reward\|>\]\s*$')


def normalize_prompt(prompt: str) -> str:
    """
    Normalize a prompt by stripping the reward goal suffix.
    
    Args:
        prompt: Raw prompt string potentially containing [Reward Goal: <|xxx_reward|>]
        
    Returns:
        Normalized prompt without reward goal suffix
    """
    return REWARD_GOAL_PATTERN.sub('', prompt).strip()


def extract_human_prompts(conversations: List[Dict[str, str]]) -> Tuple[str, ...]:
    """
    Extract and normalize all human prompts from a conversation.
    
    Args:
        conversations: List of conversation turns
        
    Returns:
        Tuple of normalized human prompts (hashable for dict key)
    """
    prompts = []
    for turn in conversations:
        if turn.get("from") == "human":
            prompt = normalize_prompt(turn.get("value", ""))
            prompts.append(prompt)
    return tuple(prompts)


def extract_task_prompts(conversations: List[Dict[str, str]]) -> Tuple[str, ...]:
    """
    Extract and normalize task prompts (non-environment setup) from a conversation.
    
    This skips the first human message (environment setup) and extracts only the
    actual task prompts. Useful for matching when environment states may differ.
    
    Args:
        conversations: List of conversation turns
        
    Returns:
        Tuple of normalized task prompts (hashable for dict key)
    """
    prompts = []
    seen_first = False
    for turn in conversations:
        if turn.get("from") == "human":
            if not seen_first:
                seen_first = True
                continue  # Skip environment setup
            prompt = normalize_prompt(turn.get("value", ""))
            prompts.append(prompt)
    return tuple(prompts)


def extract_turns_with_calls(
    conversations: List[Dict[str, str]]
) -> List[Dict[str, Any]]:
    """
    Parse conversations to extract per-turn function calls.
    
    A turn starts with a human message and includes all subsequent
    function_call entries until the next human message or end.
    
    Args:
        conversations: List of conversation turns
        
    Returns:
        List of turn dictionaries with 'human_prompt' and 'function_calls' keys
    """
    turns = []
    current_turn = None
    
    for conv in conversations:
        msg_from = conv.get("from", "")
        value = conv.get("value", "")
        
        if msg_from == "human":
            # Start a new turn
            if current_turn is not None:
                turns.append(current_turn)
            current_turn = {
                "human_prompt": normalize_prompt(value),
                "function_calls": []
            }
        elif msg_from == "function_call" and current_turn is not None:
            # Add function call to current turn
            try:
                func_call = json.loads(value)
                current_turn["function_calls"].append(func_call)
            except json.JSONDecodeError:
                logger.warning(f"Failed to parse function_call: {value[:100]}...")
                current_turn["function_calls"].append({"_raw": value, "_error": "parse_failed"})
    
    # Don't forget the last turn
    if current_turn is not None:
        turns.append(current_turn)
    
    return turns


def load_and_index_reference(ref_path: Path) -> Tuple[Dict[Tuple[str, ...], List[Dict[str, Any]]], Dict[Tuple[str, ...], List[Dict[str, Any]]]]:
    """
    Load reference file and create indexes by normalized human prompts and task prompts.
    
    Args:
        ref_path: Path to reference JSON file
        
    Returns:
        Tuple of (full_prompts_index, task_prompts_index):
        - full_prompts_index: Dict mapping all prompt tuples to list of turns
        - task_prompts_index: Dict mapping task-only prompts (no env setup) to list of turns
    """
    logger.info(f"Loading reference file: {ref_path}")
    
    with open(ref_path, 'r', encoding='utf-8') as f:
        reference_data = json.load(f)
    
    full_index = {}
    task_index = {}
    
    for sample_idx, sample in enumerate(reference_data):
        conversations = sample.get("conversations", [])
        prompts_key = extract_human_prompts(conversations)
        task_key = extract_task_prompts(conversations)
        turns = extract_turns_with_calls(conversations)
        
        if prompts_key in full_index:
            logger.warning(f"Duplicate prompts found at sample {sample_idx}, overwriting")
        
        full_index[prompts_key] = turns
        task_index[task_key] = turns
    
    logger.info(f"Indexed {len(full_index)} reference samples (full prompts)")
    logger.info(f"Indexed {len(task_index)} reference samples (task prompts only)")
    return full_index, task_index


def calculate_format_reward(predicted_calls: List[Dict[str, Any]]) -> float:
    """
    Calculate format reward: 1.0 if all calls are valid JSON format, 0.0 otherwise.
    
    Args:
        predicted_calls: List of predicted function call dictionaries
        
    Returns:
        Format reward (0.0 or 1.0)
    """
    if not predicted_calls:
        return 1.0  # No calls = valid format
    
    for call in predicted_calls:
        if "_error" in call:
            return 0.0
        if "name" not in call:
            return 0.0
    
    return 1.0


def calculate_tool_name_reward(
    predicted_calls: List[Dict[str, Any]],
    gt_calls: List[Dict[str, Any]]
) -> Tuple[float, float]:
    """
    Calculate tool name matching reward.
    
    Args:
        predicted_calls: List of predicted function calls
        gt_calls: List of ground truth function calls
        
    Returns:
        Tuple of (actual_reward, max_reward)
    """
    if not gt_calls:
        return (0.0, 0.0)
    
    gt_tool_names = [call.get("name", "") for call in gt_calls]
    pred_tool_names = [call.get("name", "") for call in predicted_calls]
    
    # Count matches (order-sensitive matching)
    matches = 0
    pred_copy = pred_tool_names.copy()
    for gt_name in gt_tool_names:
        if gt_name in pred_copy:
            matches += 1
            pred_copy.remove(gt_name)
    
    return (float(matches), float(len(gt_calls)))


def calculate_param_name_reward(
    predicted_calls: List[Dict[str, Any]],
    gt_calls: List[Dict[str, Any]]
) -> Tuple[float, float]:
    """
    Calculate parameter name matching reward using Jaccard similarity.
    
    For each ground truth call, find best matching predicted call and compute
    Jaccard similarity of parameter names.
    
    Args:
        predicted_calls: List of predicted function calls
        gt_calls: List of ground truth function calls
        
    Returns:
        Tuple of (actual_reward, max_reward)
    """
    if not gt_calls:
        return (0.0, 0.0)
    
    total_jaccard = 0.0
    max_reward = float(len(gt_calls))  # Max is 1.0 per call = |G|
    
    # Match calls by tool name first
    pred_by_name = {}
    for call in predicted_calls:
        name = call.get("name", "")
        if name not in pred_by_name:
            pred_by_name[name] = []
        pred_by_name[name].append(call)
    
    for gt_call in gt_calls:
        gt_name = gt_call.get("name", "")
        gt_params = set(gt_call.get("arguments", {}).keys())
        
        # Find matching predicted call
        if gt_name in pred_by_name and pred_by_name[gt_name]:
            pred_call = pred_by_name[gt_name].pop(0)
            pred_params = set(pred_call.get("arguments", {}).keys())
            
            # Calculate Jaccard similarity
            if gt_params or pred_params:
                intersection = len(gt_params & pred_params)
                union = len(gt_params | pred_params)
                jaccard = intersection / union if union > 0 else 0.0
            else:
                jaccard = 1.0  # Both empty = perfect match
            
            total_jaccard += jaccard
        # If no match, jaccard contribution is 0
    
    return (total_jaccard, max_reward)


def calculate_param_value_reward(
    predicted_calls: List[Dict[str, Any]],
    gt_calls: List[Dict[str, Any]]
) -> Tuple[float, float]:
    """
    Calculate parameter value matching reward.
    
    Count exact matches of parameter values for matching calls.
    
    Args:
        predicted_calls: List of predicted function calls
        gt_calls: List of ground truth function calls
        
    Returns:
        Tuple of (actual_reward, max_reward)
    """
    if not gt_calls:
        return (0.0, 0.0)
    
    total_matches = 0
    total_params = 0
    
    # Match calls by tool name first
    pred_by_name = {}
    for call in predicted_calls:
        name = call.get("name", "")
        if name not in pred_by_name:
            pred_by_name[name] = []
        pred_by_name[name].append(call)
    
    for gt_call in gt_calls:
        gt_name = gt_call.get("name", "")
        gt_args = gt_call.get("arguments", {})
        total_params += len(gt_args)
        
        # Find matching predicted call
        if gt_name in pred_by_name and pred_by_name[gt_name]:
            pred_call = pred_by_name[gt_name].pop(0)
            pred_args = pred_call.get("arguments", {})
            
            # Count exact value matches
            for param_name, gt_value in gt_args.items():
                if param_name in pred_args:
                    pred_value = pred_args[param_name]
                    # Compare values (handle type differences)
                    if gt_value == pred_value:
                        total_matches += 1
                    elif str(gt_value) == str(pred_value):
                        total_matches += 1
    
    return (float(total_matches), float(total_params))


def calculate_turn_reward(
    predicted_calls: List[Dict[str, Any]],
    gt_calls: List[Dict[str, Any]]
) -> Dict[str, Any]:
    """
    Calculate complete turn reward comparing predicted vs ground truth.
    
    R_final = R_format + R_correct
    R_correct = r_tool + r_param + r_value
    
    Args:
        predicted_calls: List of predicted function calls
        gt_calls: List of ground truth function calls
        
    Returns:
        Dictionary with reward details and normalized score
    """
    # Calculate individual rewards
    r_format = calculate_format_reward(predicted_calls)
    r_tool, max_tool = calculate_tool_name_reward(predicted_calls, gt_calls)
    r_param, max_param = calculate_param_name_reward(predicted_calls, gt_calls)
    r_value, max_value = calculate_param_value_reward(predicted_calls, gt_calls)
    
    # Total rewards
    raw_reward = r_format + r_tool + r_param + r_value
    max_reward = 1.0 + max_tool + max_param + max_value  # format max is 1.0
    
    # Normalize to [0, 1]
    if max_reward > 0:
        normalized_reward = raw_reward / max_reward
    else:
        normalized_reward = 1.0  # Empty turn = perfect score
    
    return {
        "r_format": r_format,
        "r_tool": r_tool,
        "max_tool": max_tool,
        "r_param": r_param,
        "max_param": max_param,
        "r_value": r_value,
        "max_value": max_value,
        "raw_reward": raw_reward,
        "max_reward": max_reward,
        "normalized_reward": normalized_reward
    }


def evaluate_sample(
    input_turns: List[Dict[str, Any]],
    ref_turns: List[Dict[str, Any]]
) -> List[Dict[str, Any]]:
    """
    Evaluate all turns in a sample against reference.
    
    Args:
        input_turns: List of input turns with function calls
        ref_turns: List of reference turns with function calls
        
    Returns:
        List of turn evaluation results
    """
    results = []
    
    # Match turns by index (they should align)
    max_turns = max(len(input_turns), len(ref_turns))
    
    for turn_idx in range(max_turns):
        # Get turns (or empty if missing)
        input_turn = input_turns[turn_idx] if turn_idx < len(input_turns) else {"function_calls": []}
        ref_turn = ref_turns[turn_idx] if turn_idx < len(ref_turns) else {"function_calls": []}
        
        pred_calls = input_turn.get("function_calls", [])
        gt_calls = ref_turn.get("function_calls", [])
        
        # Calculate reward
        reward_result = calculate_turn_reward(pred_calls, gt_calls)
        
        # Add turn info
        reward_result["turn_idx"] = turn_idx
        reward_result["predicted_calls"] = pred_calls
        reward_result["gt_calls"] = gt_calls
        reward_result["num_predicted"] = len(pred_calls)
        reward_result["num_gt"] = len(gt_calls)
        
        results.append(reward_result)
    
    return results


def plot_histogram(
    rewards: List[float],
    output_path: Path,
    num_bins: int = 10
) -> None:
    """
    Generate histogram showing reward distribution.
    
    Args:
        rewards: List of normalized rewards (0 to 1)
        output_path: Path to save histogram image
        num_bins: Number of bins for histogram
    """
    plt.figure(figsize=(12, 8))
    
    # Create histogram with percentage
    bins = np.linspace(0, 1, num_bins + 1)
    interval_width = 1.0 / num_bins
    counts, bin_edges, patches = plt.hist(
        rewards, 
        bins=bins, 
        weights=np.ones(len(rewards)) / len(rewards) * 100,
        edgecolor='black',
        alpha=0.7,
        color='steelblue'
    )
    
    # Add percentage labels on bars
    for count, patch in zip(counts, patches):
        if count > 0:
            plt.text(
                patch.get_x() + patch.get_width() / 2,
                patch.get_height() + 0.5,
                f'{count:.1f}%',
                ha='center',
                va='bottom',
                fontsize=9 if num_bins > 15 else 10
            )
    
    # Calculate statistics
    mean_reward = np.mean(rewards)
    std_reward = np.std(rewards)
    median_reward = np.median(rewards)
    
    # Add vertical lines for mean and median
    plt.axvline(mean_reward, color='red', linestyle='--', linewidth=2, label=f'Mean: {mean_reward:.3f}')
    plt.axvline(median_reward, color='green', linestyle='-.', linewidth=2, label=f'Median: {median_reward:.3f}')
    
    # Labels and title
    plt.xlabel('Normalized Reward', fontsize=14)
    plt.ylabel('Percentage of Turns (%)', fontsize=14)
    plt.title(
        f'Trajectory Reward Distribution (interval={interval_width:.2f})\n'
        f'Total Turns: {len(rewards)} | Mean: {mean_reward:.3f} | Std: {std_reward:.3f}',
        fontsize=16
    )
    
    # Set x-axis ticks - adjust formatting based on number of bins
    if num_bins <= 10:
        plt.xticks(bins, [f'{b:.1f}' for b in bins], fontsize=10)
    elif num_bins <= 20:
        plt.xticks(bins, [f'{b:.2f}' for b in bins], fontsize=8, rotation=45)
    else:
        # Show fewer ticks for many bins
        tick_step = max(1, num_bins // 10)
        tick_indices = list(range(0, len(bins), tick_step))
        if len(bins) - 1 not in tick_indices:
            tick_indices.append(len(bins) - 1)
        plt.xticks([bins[i] for i in tick_indices], [f'{bins[i]:.2f}' for i in tick_indices], fontsize=8, rotation=45)
    
    plt.xlim(0, 1)
    plt.ylim(0, max(counts) * 1.2 if max(counts) > 0 else 100)
    
    plt.legend(loc='upper left', fontsize=12)
    plt.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    
    # Save
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    logger.info(f"Histogram saved to: {output_path}")


def plot_histogram_by_model_tier(
    rewards_by_tier: Dict[str, List[float]],
    output_path: Path,
    num_bins: int = 10
) -> None:
    """
    Generate stacked bar histogram showing reward distribution for each model tier.
    
    Each bar represents one reward interval, divided into segments showing the
    proportion of strong/intermediate/weak model turns within that interval.
    
    Args:
        rewards_by_tier: Dictionary mapping tier names to lists of rewards
                        Expected keys: "strong", "intermediate", "weak"
        output_path: Path to save histogram image
        num_bins: Number of bins for histogram
    """
    plt.figure(figsize=(14, 10))
    
    # Define colors for each tier (from bottom to top: strong, intermediate, weak)
    tier_colors = {
        "strong": "#2ecc71",       # Green
        "intermediate": "#3498db",  # Blue  
        "weak": "#e74c3c"           # Red
    }
    tier_labels = {
        "strong": "Strong Model",
        "intermediate": "Intermediate Model",
        "weak": "Weak Model"
    }
    
    # Create bins
    bins = np.linspace(0, 1, num_bins + 1)
    interval_width = 1.0 / num_bins
    bar_width = interval_width  # Full width, no gaps between bars
    
    # Calculate raw counts for each tier in each bin
    tier_counts = {}
    tier_means = {}
    for tier_name, rewards in rewards_by_tier.items():
        if rewards:
            counts, _ = np.histogram(rewards, bins=bins)
            tier_counts[tier_name] = counts
            tier_means[tier_name] = np.mean(rewards)
        else:
            tier_counts[tier_name] = np.zeros(num_bins)
            tier_means[tier_name] = 0.0
    
    # Calculate total turns per bin (across all tiers)
    total_per_bin = np.zeros(num_bins)
    for tier_name in ["strong", "intermediate", "weak"]:
        if tier_name in tier_counts:
            total_per_bin += tier_counts[tier_name]
    
    # Calculate total turns across all bins for percentage
    total_turns = sum(len(rewards) for rewards in rewards_by_tier.values())
    
    # Convert to percentage of total turns
    tier_percentages = {}
    for tier_name in ["strong", "intermediate", "weak"]:
        if tier_name in tier_counts:
            tier_percentages[tier_name] = tier_counts[tier_name] / total_turns * 100
        else:
            tier_percentages[tier_name] = np.zeros(num_bins)
    
    # X positions for bars (center of each bin)
    x_positions = bins[:-1] + interval_width / 2
    
    # Plot stacked bars (from bottom to top: strong, intermediate, weak)
    tier_order = ["strong", "intermediate", "weak"]
    bottom = np.zeros(num_bins)
    
    for tier_name in tier_order:
        if tier_name not in tier_percentages:
            continue
        
        percentages = tier_percentages[tier_name]
        n_turns = len(rewards_by_tier.get(tier_name, []))
        mean_val = tier_means.get(tier_name, 0)
        label = f"{tier_labels[tier_name]} (n={n_turns}, μ={mean_val:.3f})"
        
        plt.bar(
            x_positions,
            percentages,
            width=bar_width,
            bottom=bottom,
            color=tier_colors[tier_name],
            edgecolor='black',
            linewidth=0.5,
            label=label
        )
        
        bottom += percentages
    
    # Add percentage labels on top of each bar
    for i, (x, total_pct) in enumerate(zip(x_positions, bottom)):
        if total_pct > 0:
            plt.text(
                x,
                total_pct + 0.3,
                f'{total_pct:.1f}%',
                ha='center',
                va='bottom',
                fontsize=8 if num_bins > 15 else 9
            )
    
    # Calculate overall statistics
    all_rewards = []
    for rewards in rewards_by_tier.values():
        all_rewards.extend(rewards)
    
    if all_rewards:
        overall_mean = np.mean(all_rewards)
        overall_std = np.std(all_rewards)
    else:
        overall_mean = 0
        overall_std = 0
    
    # Labels and title
    plt.xlabel('Normalized Reward', fontsize=14)
    plt.ylabel('Percentage of Total Turns (%)', fontsize=14)
    plt.title(
        f'Trajectory Reward Distribution by Model Tier (interval={interval_width:.2f})\n'
        f'Total Turns: {len(all_rewards)} | Overall Mean: {overall_mean:.3f} | Std: {overall_std:.3f}',
        fontsize=16
    )
    
    # Set x-axis ticks at bin edges
    if num_bins <= 10:
        plt.xticks(bins, [f'{b:.1f}' for b in bins], fontsize=10)
    elif num_bins <= 20:
        plt.xticks(bins, [f'{b:.2f}' for b in bins], fontsize=8, rotation=45)
    else:
        tick_step = max(1, num_bins // 10)
        tick_indices = list(range(0, len(bins), tick_step))
        if len(bins) - 1 not in tick_indices:
            tick_indices.append(len(bins) - 1)
        plt.xticks([bins[i] for i in tick_indices], [f'{bins[i]:.2f}' for i in tick_indices], fontsize=8, rotation=45)
    
    plt.xlim(0, 1)
    
    # Set y-axis limit
    max_height = np.max(bottom) if len(bottom) > 0 else 100
    plt.ylim(0, max_height * 1.15)
    
    # Legend (reverse order so it matches visual stacking: strong at bottom)
    handles, labels = plt.gca().get_legend_handles_labels()
    plt.legend(handles[::-1], labels[::-1], loc='upper left', fontsize=11)
    
    plt.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    
    # Save
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    logger.info(f"Model tier histogram saved to: {output_path}")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Evaluate trajectory rewards against ground truth reference"
    )
    parser.add_argument(
        "--reference",
        type=str,
        default="bfcl_glaive_dataset.json",
        help="Path to reference file with ground truth (e.g., bfcl_glaive_dataset.json)"
    )
    parser.add_argument(
        "--input",  
        type=str,
        default="rl_trajectories.json",
        help="Path to input file with predictions (e.g., rl_trajectories.json)"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="trajectory_reward_results.json",
        help="Path to output JSON file for detailed results"
    )
    parser.add_argument(
        "--plot",
        type=str,
        default="reward_histogram.png",
        help="Path to output histogram image"
    )
    parser.add_argument(
        "--bins",
        type=int,
        default=None,
        help="Number of bins for histogram (overrides --interval if both specified)"
    )
    parser.add_argument(
        "--interval",
        type=float,
        default=0.05,
        help="Interval width for histogram bins (default: 0.1, range 0-1 split into intervals of this size)"
    )
    parser.add_argument(
        "--split-models",
        action="store_true",
        default=False,
        help="Split input into 3 model tiers (strong/intermediate/weak) and plot with different colors. "
             "Assumes input file is ordered: first 1/3 = strong, middle 1/3 = intermediate, last 1/3 = weak"
    )
    
    args = parser.parse_args()
    
    ref_path = Path(args.reference)
    input_path = Path(args.input)
    output_path = Path(args.output)
    plot_path = Path(args.plot)
    
    # Validate input files
    if not ref_path.exists():
        logger.error(f"Reference file not found: {ref_path}")
        return 1
    if not input_path.exists():
        logger.error(f"Input file not found: {input_path}")
        return 1
    
    # Load and index reference file
    ref_full_index, ref_task_index = load_and_index_reference(ref_path)
    
    # Load input file
    logger.info(f"Loading input file: {input_path}")
    with open(input_path, 'r', encoding='utf-8') as f:
        input_data = json.load(f)
    logger.info(f"Loaded {len(input_data)} input samples")
    
    # Process each input sample
    all_results = []
    all_rewards = []
    matched_count = 0
    matched_by_task_count = 0  # Matched using task prompts only (env setup differed)
    unmatched_count = 0
    
    # Track rewards by sample index for model tier splitting
    rewards_by_sample = []
    
    for sample_idx, sample in enumerate(input_data):
        conversations = sample.get("conversations", [])
        
        # Extract prompts for matching
        prompts_key = extract_human_prompts(conversations)
        task_key = extract_task_prompts(conversations)
        
        # Extract turns from input
        input_turns = extract_turns_with_calls(conversations)
        
        # Try to find matching reference (first by full prompts, then by task prompts)
        if prompts_key in ref_full_index:
            ref_turns = ref_full_index[prompts_key]
            matched = True
            matched_count += 1
        elif task_key in ref_task_index:
            # Fallback: match by task prompts only (env setup differs due to state mutation bug)
            ref_turns = ref_task_index[task_key]
            matched = True
            matched_by_task_count += 1
        else:
            # No match found - use empty reference
            ref_turns = []
            matched = False
            unmatched_count += 1
            if unmatched_count <= 5:
                logger.warning(f"No reference match for sample {sample_idx}")
        
        # Evaluate turns
        turn_results = evaluate_sample(input_turns, ref_turns)
        
        # Calculate sample average
        turn_rewards = [t["normalized_reward"] for t in turn_results]
        sample_avg = np.mean(turn_rewards) if turn_rewards else 0.0
        
        # Collect all turn rewards
        all_rewards.extend(turn_rewards)
        
        # Track rewards by sample for model tier splitting
        rewards_by_sample.append(turn_rewards)
        
        # Store result
        sample_result = {
            "sample_idx": sample_idx,
            "matched": matched,
            "num_turns": len(turn_results),
            "turns": turn_results,
            "sample_avg_reward": sample_avg
        }
        all_results.append(sample_result)
    
    # Calculate summary statistics
    total_matched = matched_count + matched_by_task_count
    summary = {
        "total_samples": len(input_data),
        "matched_samples": total_matched,
        "matched_by_full_prompts": matched_count,
        "matched_by_task_prompts": matched_by_task_count,  # Matched after env setup differed
        "unmatched_samples": unmatched_count,
        "total_turns": len(all_rewards),
        "mean_reward": float(np.mean(all_rewards)) if all_rewards else 0.0,
        "std_reward": float(np.std(all_rewards)) if all_rewards else 0.0,
        "median_reward": float(np.median(all_rewards)) if all_rewards else 0.0,
        "min_reward": float(np.min(all_rewards)) if all_rewards else 0.0,
        "max_reward": float(np.max(all_rewards)) if all_rewards else 0.0
    }
    
    # Prepare output
    output_data = {
        "summary": summary,
        "samples": all_results
    }
    
    # Save JSON results
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(output_data, f, ensure_ascii=False, indent=2)
    logger.info(f"Results saved to: {output_path}")
    
    # Generate histogram
    if all_rewards:
        # Determine number of bins: --bins takes precedence, otherwise calculate from --interval
        if args.bins is not None:
            num_bins = args.bins
        else:
            # Calculate bins from interval (0-1 range divided by interval width)
            num_bins = int(1.0 / args.interval)
        
        if args.split_models:
            # Split samples into 3 tiers: strong (first 1/3), intermediate (middle 1/3), weak (last 1/3)
            num_samples = len(rewards_by_sample)
            tier_size = num_samples // 3
            remainder = num_samples % 3
            
            # Calculate tier boundaries
            # First tier gets one extra if remainder >= 1
            # Second tier gets one extra if remainder >= 2
            strong_end = tier_size + (1 if remainder >= 1 else 0)
            intermediate_end = strong_end + tier_size + (1 if remainder >= 2 else 0)
            
            # Collect rewards by tier
            strong_rewards = []
            intermediate_rewards = []
            weak_rewards = []
            
            for sample_idx, sample_rewards in enumerate(rewards_by_sample):
                if sample_idx < strong_end:
                    strong_rewards.extend(sample_rewards)
                elif sample_idx < intermediate_end:
                    intermediate_rewards.extend(sample_rewards)
                else:
                    weak_rewards.extend(sample_rewards)
            
            rewards_by_tier = {
                "strong": strong_rewards,
                "intermediate": intermediate_rewards,
                "weak": weak_rewards
            }
            
            logger.info(f"Model tier split: Strong={len(strong_rewards)} turns ({strong_end} samples), "
                       f"Intermediate={len(intermediate_rewards)} turns ({intermediate_end - strong_end} samples), "
                       f"Weak={len(weak_rewards)} turns ({num_samples - intermediate_end} samples)")
            
            # Add tier info to summary
            summary["model_tiers"] = {
                "strong": {
                    "num_samples": strong_end,
                    "num_turns": len(strong_rewards),
                    "mean_reward": float(np.mean(strong_rewards)) if strong_rewards else 0.0,
                    "std_reward": float(np.std(strong_rewards)) if strong_rewards else 0.0
                },
                "intermediate": {
                    "num_samples": intermediate_end - strong_end,
                    "num_turns": len(intermediate_rewards),
                    "mean_reward": float(np.mean(intermediate_rewards)) if intermediate_rewards else 0.0,
                    "std_reward": float(np.std(intermediate_rewards)) if intermediate_rewards else 0.0
                },
                "weak": {
                    "num_samples": num_samples - intermediate_end,
                    "num_turns": len(weak_rewards),
                    "mean_reward": float(np.mean(weak_rewards)) if weak_rewards else 0.0,
                    "std_reward": float(np.std(weak_rewards)) if weak_rewards else 0.0
                }
            }
            
            # Update output data with tier info
            output_data["summary"] = summary
            
            # Re-save JSON with tier info
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(output_data, f, ensure_ascii=False, indent=2)
            
            plot_histogram_by_model_tier(rewards_by_tier, plot_path, num_bins=num_bins)
        else:
            plot_histogram(all_rewards, plot_path, num_bins=num_bins)
        
        logger.info(f"Histogram bins: {num_bins} (interval width: {1.0/num_bins:.3f})")
    else:
        logger.warning("No rewards to plot")
    
    # Print summary
    logger.info("=" * 60)
    logger.info("Evaluation Summary")
    logger.info("=" * 60)
    logger.info(f"Total samples: {summary['total_samples']}")
    logger.info(f"Matched samples: {summary['matched_samples']}")
    logger.info(f"  - By full prompts: {summary['matched_by_full_prompts']}")
    logger.info(f"  - By task prompts (env setup differed): {summary['matched_by_task_prompts']}")
    logger.info(f"Unmatched samples: {summary['unmatched_samples']}")
    logger.info(f"Total turns: {summary['total_turns']}")
    logger.info(f"Mean reward: {summary['mean_reward']:.4f}")
    logger.info(f"Std reward: {summary['std_reward']:.4f}")
    logger.info(f"Median reward: {summary['median_reward']:.4f}")
    logger.info(f"Min reward: {summary['min_reward']:.4f}")
    logger.info(f"Max reward: {summary['max_reward']:.4f}")
    
    # Print tier-specific statistics if --split-models was used
    if args.split_models and "model_tiers" in summary:
        logger.info("-" * 60)
        logger.info("Model Tier Statistics:")
        for tier_name in ["strong", "intermediate", "weak"]:
            tier_info = summary["model_tiers"][tier_name]
            logger.info(f"  {tier_name.capitalize():12s}: {tier_info['num_samples']:3d} samples, "
                       f"{tier_info['num_turns']:4d} turns, "
                       f"mean={tier_info['mean_reward']:.4f}, std={tier_info['std_reward']:.4f}")
    
    logger.info("=" * 60)
    
    return 0


if __name__ == "__main__":
    exit(main())

