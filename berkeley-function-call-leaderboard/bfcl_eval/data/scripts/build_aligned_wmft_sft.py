#!/usr/bin/env python3
"""
Build Aligned WMFT and SFT Datasets for BFCL (RC-GRPO Aligned)

CRITICAL: This script generates data that EXACTLY matches the format used during RL training.

Key Alignment Requirements (from verl/experimental/agent_loop/env_adapters/bfcl_adapter.py):
1. NO "Environment setup:" block in user messages - initial_config is HIDDEN from model
2. User message format: Combined task steps ("Step 1: ...\n\nStep 2: ...\n\nComplete all steps...")
3. System prompt: Simple assistant prompt (not detailed tool specs)
4. Tools: OpenAI format array with done() tool added
5. RC token: Injected at END of first user message as "\n\n[Reward Goal: <|xxx_reward|>]"

This creates two dataset types:
- WMFT: ALL trajectories (success + failure) with binary reward conditioning tokens
- SFT: Success-only trajectories, NO reward conditioning tokens

Usage:
    python build_aligned_wmft_sft.py \
        --bfcl-data-dir ../  \
        --output-dir ../generated/aligned \
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

# Reward conditioning tokens (must match verl/experimental/agent_loop/src_unified_agent_loop.py)
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


def get_system_prompt() -> str:
    """
    Get the system prompt matching BFCLEnvironmentAdapter._get_system_prompt().
    
    CRITICAL: Must match exactly what the RL environment uses.
    """
    return (
        "You are a helpful assistant that completes tasks by calling the appropriate tools. "
        "Analyze the task requirements carefully, then execute the necessary tool calls in the correct order. "
        "You can call multiple tools as needed. When you have completed all required actions, "
        "call the done() function to finish the task."
    )


def get_done_tool() -> Dict:
    """
    Get the done() tool schema matching BFCLEnvironmentAdapter._get_done_tool().
    
    CRITICAL: Must match exactly what the RL environment uses.
    """
    return {
        "type": "function",
        "function": {
            "name": "done",
            "description": "Call this function when you have completed all the required tasks. This will end the episode.",
            "parameters": {
                "type": "object",
                "properties": {},
                "required": []
            }
        }
    }


def build_initial_task_prompt(questions: List) -> str:
    """
    Combine all turn questions into a single task description.
    
    CRITICAL: Must match BFCLEnvironmentAdapter._build_initial_task_prompt() exactly.
    
    Args:
        questions: List of turn questions from BFCL task
        
    Returns:
        Combined task prompt string
    """
    if not questions:
        return "Complete the task using the available tools. Call done() when finished."
    
    # Flatten all questions into task requirements
    task_parts = []
    for i, turn_q in enumerate(questions, 1):
        if isinstance(turn_q, list) and turn_q:
            q = turn_q[0]
            if isinstance(q, dict):
                content = q.get("content", str(q))
            else:
                content = str(q)
            task_parts.append(f"Step {i}: {content}")
        elif isinstance(turn_q, str):
            task_parts.append(f"Step {i}: {turn_q}")
    
    combined = "\n\n".join(task_parts)
    combined += "\n\nComplete all the steps above using the available tools. Call done() when you have finished all tasks."
    return combined


def inject_reward_token(content: str, reward: float) -> str:
    """
    Inject reward conditioning token at END of content.
    
    CRITICAL: Must match _inject_src_token format in agent loops.
    
    Args:
        content: Original message content
        reward: Binary reward (1.0 = success, 0.0 = failure)
        
    Returns:
        Content with reward token appended
    """
    token = HIGH_REWARD_TOKEN if reward >= 0.5 else LOW_REWARD_TOKEN
    return f"{content}\n\n[Reward Goal: {token}]"


def load_tool_docs(involved_classes: List[str], func_doc_dir: Path) -> List[Dict]:
    """
    Load tool schemas for involved classes in OpenAI format.
    
    Args:
        involved_classes: List of class names
        func_doc_dir: Path to multi_turn_func_doc directory
        
    Returns:
        List of tool schemas in OpenAI format
    """
    tools = []
    
    for class_name in involved_classes:
        doc_file = CLASS_TO_DOC_FILE.get(class_name)
        if not doc_file:
            logger.warning(f"No doc file mapping for class: {class_name}")
            continue
        
        doc_path = func_doc_dir / doc_file
        if not doc_path.exists():
            logger.warning(f"Doc file not found: {doc_path}")
            continue
        
        # Load tool schemas (JSONL format)
        with open(doc_path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    tool = json.loads(line)
                    # Wrap in OpenAI function format
                    wrapped_tool = {
                        "type": "function",
                        "function": tool
                    }
                    tools.append(wrapped_tool)
                except json.JSONDecodeError:
                    continue
    
    return tools


def load_bfcl_tasks(bfcl_data_dir: Path) -> List[Dict]:
    """
    Load all BFCL multi-turn tasks.
    
    Args:
        bfcl_data_dir: Path to BFCL data directory
        
    Returns:
        List of task dictionaries with source_file added
    """
    tasks = []
    
    # Find all multi-turn BFCL files
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
    """
    Load ground truth actions for a task.
    
    Args:
        task_id: Task ID
        source_file: Source file name
        answer_dir: Path to possible_answer directory
        
    Returns:
        Ground truth as list of turns, each containing list of actions
    """
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
    """
    Convert Python-style function call to OpenAI tool_calls format.
    
    Args:
        func_call: Function call string like "cd(folder='temp')"
        
    Returns:
        OpenAI tool_calls format dict
    """
    import re
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
    
    # Extract function name and arguments
    func_name = func_call[:func_call.index('(')].strip()
    params_str = func_call[func_call.index('(') + 1:func_call.rindex(')')].strip()
    
    # Parse parameters
    arguments = {}
    if params_str:
        # Parse key=value pairs
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
                    
                    # Parse value
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


def create_aligned_sample(
    task: Dict,
    ground_truth: List[List[str]],
    tools: List[Dict],
    reward: float,
    include_reward_token: bool = True
) -> Dict:
    """
    Create a single aligned training sample.
    
    CRITICAL: This format MUST match what the RL environment produces.
    
    Args:
        task: BFCL task dictionary
        ground_truth: Ground truth actions
        tools: Tool schemas in OpenAI format
        reward: Binary reward (1.0 or 0.0)
        include_reward_token: Whether to include RC token
        
    Returns:
        Training sample in OpenAI messages format
    """
    questions = task.get("question", [])
    task_id = task.get("id", "unknown")
    
    # Build initial task prompt (NO environment setup!)
    task_prompt = build_initial_task_prompt(questions)
    
    # Optionally inject reward token
    if include_reward_token:
        task_prompt = inject_reward_token(task_prompt, reward)
    
    # Build messages
    messages = []
    
    # System prompt
    messages.append({
        "role": "system",
        "content": get_system_prompt()
    })
    
    # User message with combined task (NO environment setup)
    messages.append({
        "role": "user",
        "content": task_prompt
    })
    
    # Add assistant responses with tool calls
    # For SFT, we show the expert trajectory
    for turn_idx, turn_actions in enumerate(ground_truth):
        if not turn_actions:
            continue
        
        # Convert actions to tool_calls format
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
        
        # Tool response messages (mock observations)
        for tool_call in tool_calls:
            messages.append({
                "role": "tool",
                "tool_call_id": tool_call["id"],
                "content": '{"status": "success"}'
            })
    
    # Final done() call
    done_call = {
        "id": "call_done",
        "type": "function",
        "function": {
            "name": "done",
            "arguments": "{}"
        }
    }
    messages.append({
        "role": "assistant",
        "content": None,
        "tool_calls": [done_call]
    })
    
    # Add done() to tools
    tools_with_done = tools + [get_done_tool()]
    
    return {
        "messages": messages,
        "tools": tools_with_done,
        "task_id": task_id,
        "source_file": task.get("source_file", ""),
        "final_reward": reward
    }


def main():
    parser = argparse.ArgumentParser(
        description="Build aligned WMFT and SFT datasets for BFCL (RC-GRPO aligned)"
    )
    parser.add_argument(
        "--bfcl-data-dir",
        type=str,
        required=True,
        help="Path to BFCL data directory containing BFCL_v4_multi_turn_*.json files"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        required=True,
        help="Output directory for generated datasets"
    )
    parser.add_argument(
        "--train-ratio",
        type=float,
        default=0.8,
        help="Train/test split ratio (default: 0.8)"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility"
    )
    parser.add_argument(
        "--format",
        type=str,
        choices=["openai"],
        default="openai",
        help="Output format (only openai supported for alignment)"
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
    
    # Load tool documentation
    func_doc_dir = bfcl_data_dir / "multi_turn_func_doc"
    
    # Process tasks
    wmft_samples = []
    sft_samples = []
    
    success_count = 0
    failure_count = 0
    
    for task in tasks:
        task_id = task.get("id")
        source_file = task.get("source_file")
        involved_classes = task.get("involved_classes", [])
        
        # Load ground truth
        answer_dir = bfcl_data_dir / "possible_answer"
        ground_truth = load_ground_truth(task_id, source_file, answer_dir)
        
        if not ground_truth:
            logger.warning(f"No ground truth for {task_id}")
            continue
        
        # Load tools
        tools = load_tool_docs(involved_classes, func_doc_dir)
        
        # For ground truth data, reward is always 1.0 (success)
        reward = 1.0
        success_count += 1
        
        # Create WMFT sample (with reward token)
        wmft_sample = create_aligned_sample(
            task, ground_truth, tools, reward, include_reward_token=True
        )
        wmft_samples.append(wmft_sample)
        
        # Create SFT sample (without reward token)
        sft_sample = create_aligned_sample(
            task, ground_truth, tools, reward, include_reward_token=False
        )
        sft_samples.append(sft_sample)
    
    logger.info(f"Created {len(wmft_samples)} WMFT samples (all success)")
    logger.info(f"Created {len(sft_samples)} SFT samples (all success)")
    
    # Split into train/test
    random.shuffle(wmft_samples)
    random.shuffle(sft_samples)
    
    split_idx_wm = int(len(wmft_samples) * args.train_ratio)
    split_idx_sft = int(len(sft_samples) * args.train_ratio)
    
    wmft_train = wmft_samples[:split_idx_wm]
    wmft_test = wmft_samples[split_idx_wm:]
    sft_train = sft_samples[:split_idx_sft]
    sft_test = sft_samples[split_idx_sft:]
    
    # Write output files
    outputs = {
        "aligned_wm_train.json": wmft_train,
        "aligned_wm_test.json": wmft_test,
        "aligned_sft_train.json": sft_train,
        "aligned_sft_test.json": sft_test,
    }
    
    for filename, data in outputs.items():
        output_path = output_dir / filename
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        logger.info(f"Wrote {len(data)} samples to {output_path}")
    
    # Summary
    logger.info("=" * 60)
    logger.info("Aligned Dataset Generation Complete!")
    logger.info("=" * 60)
    logger.info(f"WMFT Train: {len(wmft_train)} samples")
    logger.info(f"WMFT Test: {len(wmft_test)} samples")
    logger.info(f"SFT Train: {len(sft_train)} samples")
    logger.info(f"SFT Test: {len(sft_test)} samples")
    logger.info("")
    logger.info("Format Alignment:")
    logger.info("  ✓ NO 'Environment setup:' block (hidden from model)")
    logger.info("  ✓ Combined task prompt ('Step 1: ...\\n\\nStep 2: ...')")
    logger.info("  ✓ Simple system prompt (matches RL adapter)")
    logger.info("  ✓ done() tool included")
    logger.info("  ✓ RC token at END of first user message")
    logger.info("=" * 60)
    
    return 0


if __name__ == "__main__":
    exit(main())
