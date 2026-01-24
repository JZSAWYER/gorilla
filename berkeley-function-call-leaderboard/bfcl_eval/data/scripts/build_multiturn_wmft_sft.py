#!/usr/bin/env python3
"""
Build Multi-Turn WMFT and SFT Datasets for BFCL (Official Turn-by-Turn Format)

This script generates datasets that match the **official BFCL evaluation format**:
- Turn-by-turn interaction: user sends message → model responds → user sends next message
- NOT combined "Step 1: ..., Step 2: ..." format (that's for free exploration mode)

Reference: BFCL V3 Multi-Turn evaluation methodology
- "multi-turn function calls, which involve multiple exchanges or function calls between user and assistant"
- See: bfcl_eval/model_handler/base_handler.py inference_multi_turn_FC()

This creates two dataset types:
- WMFT: ALL trajectories (success + failure) with binary reward conditioning tokens
- SFT: Success-only trajectories, NO reward conditioning tokens

Key Differences from build_aligned_wmft_sft.py:
1. This version uses TURN-BY-TURN format (official BFCL)
2. The "aligned" version uses COMBINED format (RC-GRPO free exploration)

Usage:
    python build_multiturn_wmft_sft.py \
        --bfcl-data-dir ../  \
        --output-dir ../generated/multiturn \
        --format openai

Author: RC-GRPO Implementation
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

# Reward conditioning tokens
HIGH_REWARD_TOKEN = "<|high_reward|>"
LOW_REWARD_TOKEN = "<|low_reward|>"

# Class to file mapping for tool documentation
CLASS_TO_DOC_FILE = {
    "GorillaFileSystem": "gorilla_file_system.json",
    "MathAPI": "math_api.json",
    "MessageAPI": "message_api.json",
    "TwitterAPI": "posting_api.json",
    "TicketAPI": "ticket_api.json",
    "TradingBot": "trading_bot.json",
    "TravelAPI": "travel_booking.json",
    "VehicleControlAPI": "vehicle_control.json",
    "WebSearchAPI": "web_search.json",
    "MemoryAPI_kv": "memory_kv.json",
    "MemoryAPI_vector": "memory_vector.json",
    "MemoryAPI_rec_sum": "memory_rec_sum.json",
}


def load_tool_docs(involved_classes: List[str], func_doc_dir: Path) -> List[Dict]:
    """Load tool schemas for involved classes in OpenAI format."""
    tools = []
    
    for class_name in involved_classes:
        doc_file = CLASS_TO_DOC_FILE.get(class_name)
        if not doc_file:
            continue
        
        doc_path = func_doc_dir / doc_file
        if not doc_path.exists():
            continue
        
        with open(doc_path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    tool = json.loads(line)
                    wrapped_tool = {
                        "type": "function",
                        "function": tool
                    }
                    tools.append(wrapped_tool)
                except json.JSONDecodeError:
                    continue
    
    return tools


def load_bfcl_tasks(bfcl_data_dir: Path) -> List[Dict]:
    """Load all BFCL multi-turn tasks."""
    tasks = []
    
    bfcl_files = list(bfcl_data_dir.glob("BFCL_v4_multi_turn_*.json"))
    bfcl_files = [f for f in bfcl_files if "possible_answer" not in str(f)]
    
    for file_path in sorted(bfcl_files):
        source_file = file_path.name
        
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    task = json.loads(line)
                    task['source_file'] = source_file
                    tasks.append(task)
                except json.JSONDecodeError:
                    continue
    
    logger.info(f"Loaded {len(tasks)} BFCL tasks from {len(bfcl_files)} files")
    return tasks


def load_ground_truth(task_id: str, source_file: str, answer_dir: Path) -> Optional[List[List[str]]]:
    """Load ground truth actions for a task."""
    answer_path = answer_dir / source_file
    if not answer_path.exists():
        return None
    
    with open(answer_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                answer = json.loads(line)
                if answer.get('id') == task_id:
                    return answer.get('ground_truth', [])
            except json.JSONDecodeError:
                continue
    
    return None


def convert_function_call_to_openai(func_call: str) -> Dict:
    """Convert Python-style function call to OpenAI tool_calls format."""
    import ast
    
    if '(' not in func_call:
        return {
            "id": f"call_{hash(func_call) % 10000}",
            "type": "function",
            "function": {
                "name": func_call,
                "arguments": "{}"
            }
        }
    
    func_name = func_call[:func_call.index('(')].strip()
    params_str = func_call[func_call.index('(') + 1:func_call.rindex(')')].strip()
    
    arguments = {}
    if params_str:
        current_param = ""
        in_quotes = False
        quote_char = None
        bracket_depth = 0
        
        for char in params_str + ',':
            if char in ('"', "'") and (not in_quotes or char == quote_char):
                if not in_quotes:
                    in_quotes = True
                    quote_char = char
                else:
                    in_quotes = False
                    quote_char = None
                current_param += char
            elif char in ('[', '{'):
                bracket_depth += 1
                current_param += char
            elif char in (']', '}'):
                bracket_depth -= 1
                current_param += char
            elif char == ',' and not in_quotes and bracket_depth == 0:
                if '=' in current_param:
                    key, value = current_param.split('=', 1)
                    key = key.strip()
                    value = value.strip()
                    
                    try:
                        parsed_value = ast.literal_eval(value)
                    except:
                        parsed_value = value
                    
                    arguments[key] = parsed_value
                current_param = ""
            else:
                current_param += char
    
    return {
        "id": f"call_{hash(func_call) % 10000}",
        "type": "function",
        "function": {
            "name": func_name,
            "arguments": json.dumps(arguments)
        }
    }


def inject_reward_token(content: str, reward: float) -> str:
    """Inject reward conditioning token at END of content."""
    token = HIGH_REWARD_TOKEN if reward >= 0.5 else LOW_REWARD_TOKEN
    return f"{content}\n\n[Reward Goal: {token}]"


def create_multiturn_sample(
    task: Dict,
    ground_truth: List[List[str]],
    tools: List[Dict],
    reward: float,
    include_reward_token: bool = True
) -> Dict:
    """
    Create a multi-turn training sample in OFFICIAL BFCL format.
    
    This format has:
    - Turn-by-turn user messages
    - Model responds to each turn separately
    - Matches inference_multi_turn_FC behavior in base_handler.py
    
    Args:
        task: BFCL task dictionary
        ground_truth: Ground truth actions per turn
        tools: Tool schemas in OpenAI format
        reward: Binary reward (1.0 or 0.0)
        include_reward_token: Whether to include RC token
        
    Returns:
        Training sample in OpenAI messages format
    """
    questions = task.get("question", [])  # List[List[Dict]] - turns of user messages
    task_id = task.get("id", "unknown")
    
    # Build messages with turn-by-turn interaction
    messages = []
    
    # System prompt (simple, no environment setup)
    messages.append({
        "role": "system",
        "content": (
            "You are a helpful assistant that completes tasks by calling the appropriate tools. "
            "Analyze the user's request and call the necessary functions to fulfill it."
        )
    })
    
    # Process each turn
    for turn_idx, turn_questions in enumerate(questions):
        # Get user message for this turn
        user_content = ""
        for msg in turn_questions:
            if msg.get("role") == "user":
                user_content = msg.get("content", "")
                break
        
        if not user_content:
            continue
        
        # Inject reward token on FIRST user message only
        if turn_idx == 0 and include_reward_token:
            user_content = inject_reward_token(user_content, reward)
        
        # Add user message
        messages.append({
            "role": "user",
            "content": user_content
        })
        
        # Add assistant response with tool calls for this turn
        if turn_idx < len(ground_truth):
            turn_actions = ground_truth[turn_idx]
            
            if turn_actions:
                tool_calls = []
                for action in turn_actions:
                    tool_call = convert_function_call_to_openai(action)
                    tool_calls.append(tool_call)
                
                # Assistant message with tool calls
                messages.append({
                    "role": "assistant",
                    "content": None,
                    "tool_calls": tool_calls
                })
                
                # Tool response messages
                for tool_call in tool_calls:
                    messages.append({
                        "role": "tool",
                        "tool_call_id": tool_call["id"],
                        "content": '{"status": "success"}'
                    })
            else:
                # No actions for this turn (e.g., miss_func category)
                messages.append({
                    "role": "assistant",
                    "content": "I don't have the required function to complete this task."
                })
    
    return {
        "messages": messages,
        "tools": tools,
        "task_id": task_id,
        "source_file": task.get("source_file", ""),
        "final_reward": reward,
        "num_turns": len(questions)
    }


def main():
    parser = argparse.ArgumentParser(
        description="Build multi-turn WMFT and SFT datasets for BFCL (official turn-by-turn format)"
    )
    parser.add_argument(
        "--bfcl-data-dir",
        type=str,
        required=True,
        help="Path to BFCL data directory"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        required=True,
        help="Output directory"
    )
    parser.add_argument(
        "--train-ratio",
        type=float,
        default=0.8,
        help="Train/test split ratio"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed"
    )
    
    args = parser.parse_args()
    random.seed(args.seed)
    
    bfcl_data_dir = Path(args.bfcl_data_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load all tasks
    tasks = load_bfcl_tasks(bfcl_data_dir)
    
    if not tasks:
        logger.error("No tasks found!")
        return 1
    
    func_doc_dir = bfcl_data_dir / "multi_turn_func_doc"
    
    # Process tasks
    wmft_samples = []
    sft_samples = []
    
    for task in tasks:
        task_id = task.get("id")
        source_file = task.get("source_file")
        involved_classes = task.get("involved_classes", [])
        
        answer_dir = bfcl_data_dir / "possible_answer"
        ground_truth = load_ground_truth(task_id, source_file, answer_dir)
        
        if not ground_truth:
            continue
        
        tools = load_tool_docs(involved_classes, func_doc_dir)
        reward = 1.0  # Ground truth = success
        
        # Create WMFT sample (with reward token)
        wmft_sample = create_multiturn_sample(
            task, ground_truth, tools, reward, include_reward_token=True
        )
        wmft_samples.append(wmft_sample)
        
        # Create SFT sample (without reward token)
        sft_sample = create_multiturn_sample(
            task, ground_truth, tools, reward, include_reward_token=False
        )
        sft_samples.append(sft_sample)
    
    logger.info(f"Created {len(wmft_samples)} WMFT samples")
    logger.info(f"Created {len(sft_samples)} SFT samples")
    
    # Split into train/test
    random.shuffle(wmft_samples)
    random.shuffle(sft_samples)
    
    split_idx = int(len(wmft_samples) * args.train_ratio)
    
    wmft_train = wmft_samples[:split_idx]
    wmft_test = wmft_samples[split_idx:]
    sft_train = sft_samples[:split_idx]
    sft_test = sft_samples[split_idx:]
    
    # Write output files
    outputs = {
        "multiturn_wm_train.json": wmft_train,
        "multiturn_wm_test.json": wmft_test,
        "multiturn_sft_train.json": sft_train,
        "multiturn_sft_test.json": sft_test,
    }
    
    for filename, data in outputs.items():
        output_path = output_dir / filename
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        logger.info(f"Wrote {len(data)} samples to {output_path}")
    
    # Summary
    logger.info("=" * 70)
    logger.info("Multi-Turn Dataset Generation Complete!")
    logger.info("=" * 70)
    logger.info(f"WMFT Train: {len(wmft_train)} samples")
    logger.info(f"WMFT Test: {len(wmft_test)} samples")
    logger.info(f"SFT Train: {len(sft_train)} samples")
    logger.info(f"SFT Test: {len(sft_test)} samples")
    logger.info("")
    logger.info("Format: OFFICIAL BFCL TURN-BY-TURN")
    logger.info("  ✓ Turn-by-turn user messages (matches base_handler.py)")
    logger.info("  ✓ Model responds to each turn separately")
    logger.info("  ✓ RC token on FIRST user message only (for WMFT)")
    logger.info("  ✓ NO environment setup block")
    logger.info("")
    logger.info("NOTE: This is different from 'aligned' format (RC-GRPO free exploration)")
    logger.info("      which combines all steps into single prompt.")
    logger.info("=" * 70)
    
    return 0


if __name__ == "__main__":
    exit(main())
