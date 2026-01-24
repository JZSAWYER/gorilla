#!/usr/bin/env python3
"""
Build Expert Trajectories from BFCL Ground Truth.

This script creates expert demonstration trajectories by:
1. Loading BFCL tasks (questions + initial_config)
2. Loading ground truth function calls from possible_answer files
3. Executing ground truth calls to get observations
4. Converting to OpenAI messages format

Output: 800 expert trajectories that can be used for:
- SFT training (all success)
- WM training (combined with failure trajectories for 50-50)
"""

import json
import random
import sys
from pathlib import Path
from collections import defaultdict

# Add BFCL eval path
BFCL_EVAL_DIR = Path(__file__).parent.parent.parent
sys.path.insert(0, str(BFCL_EVAL_DIR))

try:
    from eval_checker.multi_turn_eval.multi_turn_utils import execute_multi_turn_func_call
    EXEC_AVAILABLE = True
except ImportError:
    EXEC_AVAILABLE = False
    print("Warning: Cannot import execution utils, will use placeholder observations")

# Paths
BFCL_DATA_DIR = Path(__file__).parent.parent
OUTPUT_DIR = BFCL_DATA_DIR / "generated/aligned"
RL_TRAIN_PATH = Path("/home/dataset-local/projects/verl/examples/bfcl/data/bfcl_rl_train.json")
RL_TEST_PATH = Path("/home/dataset-local/projects/verl/examples/bfcl/data/bfcl_rl_test.json")

# Source files
SOURCE_FILES = [
    ('BFCL_v4_multi_turn_base.json', 'possible_answer/BFCL_v4_multi_turn_base.json'),
    ('BFCL_v4_multi_turn_long_context.json', 'possible_answer/BFCL_v4_multi_turn_long_context.json'),
    ('BFCL_v4_multi_turn_miss_func.json', 'possible_answer/BFCL_v4_multi_turn_miss_func.json'),
    ('BFCL_v4_multi_turn_miss_param.json', 'possible_answer/BFCL_v4_multi_turn_miss_param.json'),
]

SYSTEM_PROMPT = """You are a helpful assistant that completes tasks by calling the appropriate tools. Analyze the task requirements carefully, then execute the necessary tool calls in the correct order. You can call multiple tools as needed. When you have completed all required actions, call the done() function to finish the task."""


def load_bfcl_data():
    """Load BFCL tasks and ground truth."""
    tasks = {}
    ground_truths = {}
    
    for task_file, gt_file in SOURCE_FILES:
        task_path = BFCL_DATA_DIR / task_file
        gt_path = BFCL_DATA_DIR / gt_file
        
        # Load tasks
        with open(task_path) as f:
            for line in f:
                task = json.loads(line.strip())
                tasks[task['id']] = task
        
        # Load ground truth
        with open(gt_path) as f:
            for line in f:
                gt = json.loads(line.strip())
                ground_truths[gt['id']] = gt['ground_truth']
    
    return tasks, ground_truths


def parse_function_call(call_str):
    """Parse function call string like "func_name(arg1='val1', arg2='val2')"."""
    # Handle class.method format
    if '.' in call_str.split('(')[0]:
        func_name = call_str.split('(')[0].split('.')[-1]
    else:
        func_name = call_str.split('(')[0]
    
    # Extract arguments
    args_str = call_str[call_str.find('(')+1:call_str.rfind(')')]
    
    # Parse arguments
    args = {}
    if args_str.strip():
        # Simple parsing - this handles most cases
        current_key = ""
        current_val = ""
        in_string = False
        string_char = None
        depth = 0
        
        i = 0
        while i < len(args_str):
            c = args_str[i]
            
            if c in '"\'':
                if not in_string:
                    in_string = True
                    string_char = c
                elif c == string_char and args_str[i-1] != '\\':
                    in_string = False
                current_val += c
            elif c in '[{':
                depth += 1
                current_val += c
            elif c in ']}':
                depth -= 1
                current_val += c
            elif c == '=' and not in_string and depth == 0 and not current_key:
                current_key = current_val.strip()
                current_val = ""
            elif c == ',' and not in_string and depth == 0:
                if current_key:
                    # Clean up the value
                    val = current_val.strip()
                    if val.startswith(("'", '"')) and val.endswith(("'", '"')):
                        val = val[1:-1]
                    args[current_key] = val
                current_key = ""
                current_val = ""
            else:
                current_val += c
            i += 1
        
        # Don't forget the last argument
        if current_key:
            val = current_val.strip()
            if val.startswith(("'", '"')) and val.endswith(("'", '"')):
                val = val[1:-1]
            args[current_key] = val
    
    return func_name, args


def create_expert_trajectory(task, ground_truth, task_id, source_file, add_reward_token=True, is_success=True):
    """Create an expert trajectory from task and ground truth."""
    messages = []
    
    # System message
    messages.append({
        "role": "system",
        "content": SYSTEM_PROMPT
    })
    
    # Get questions (user turns)
    questions = task.get('question', [])
    
    # Build combined task prompt (RC-GRPO style)
    task_steps = []
    for i, q_turn in enumerate(questions, 1):
        if q_turn and isinstance(q_turn, list) and q_turn[0].get('content'):
            task_steps.append(f"Step {i}: {q_turn[0]['content']}")
    
    combined_prompt = "\n\n".join(task_steps)
    combined_prompt += "\n\nComplete all the steps above using the available tools. Call done() when you have finished all tasks."
    
    # Add reward token if requested
    if add_reward_token:
        reward_token = "<|high_reward|>" if is_success else "<|low_reward|>"
        combined_prompt += f"\n\n[Reward Goal: {reward_token}]"
    
    messages.append({
        "role": "user",
        "content": combined_prompt
    })
    
    # Add ground truth function calls and observations
    tool_call_id_counter = 1000
    
    for turn_idx, turn_calls in enumerate(ground_truth):
        if not turn_calls:
            continue
            
        # Parse function calls for this turn
        if isinstance(turn_calls, str):
            turn_calls = [turn_calls]
        
        tool_calls = []
        for call_str in turn_calls:
            func_name, args = parse_function_call(call_str)
            tool_call_id = f"call_{tool_call_id_counter}"
            tool_call_id_counter += 1
            
            tool_calls.append({
                "id": tool_call_id,
                "type": "function",
                "function": {
                    "name": func_name,
                    "arguments": json.dumps(args)
                }
            })
        
        # Add assistant message with tool calls
        # Use empty string instead of None for content (LLaMA-Factory compatibility)
        messages.append({
            "role": "assistant",
            "content": "",
            "tool_calls": tool_calls
        })
        
        # Add tool responses (placeholder since we can't execute)
        for tc in tool_calls:
            messages.append({
                "role": "tool",
                "tool_call_id": tc["id"],
                "content": '{"status": "success"}'
            })
    
    # Add final done() call
    done_id = f"call_{tool_call_id_counter}"
    messages.append({
        "role": "assistant",
        "content": "",  # Use empty string instead of None (LLaMA-Factory compatibility)
        "tool_calls": [{
            "id": done_id,
            "type": "function",
            "function": {
                "name": "done",
                "arguments": "{}"
            }
        }]
    })
    messages.append({
        "role": "tool",
        "tool_call_id": done_id,
        "content": '{"status": "completed"}'
    })
    
    # Get tools from task
    tools = []
    # Add done tool
    tools.append({
        "type": "function",
        "function": {
            "name": "done",
            "description": "Call this function when all tasks are completed.",
            "parameters": {"type": "object", "properties": {}}
        }
    })
    
    return {
        "messages": messages,
        "tools": json.dumps(tools, ensure_ascii=False),  # Serialize to JSON string for consistency
        "task_id": task_id,
        "source_file": source_file,
        "final_reward": 1.0 if is_success else 0.0
    }


def main():
    print("=" * 60)
    print("Building Expert Trajectories from BFCL Ground Truth")
    print("=" * 60)
    
    # Load data
    print("\nLoading BFCL data...")
    tasks, ground_truths = load_bfcl_data()
    print(f"  Tasks: {len(tasks)}")
    print(f"  Ground truths: {len(ground_truths)}")
    
    # Load RL split
    print("\nLoading RL split...")
    with open(RL_TRAIN_PATH) as f:
        rl_train = json.load(f)
    with open(RL_TEST_PATH) as f:
        rl_test = json.load(f)
    
    rl_train_ids = set(s['task_id'] for s in rl_train)
    rl_test_ids = set(s['task_id'] for s in rl_test)
    print(f"  RL train tasks: {len(rl_train_ids)}")
    print(f"  RL test tasks: {len(rl_test_ids)}")
    
    # Create expert trajectories
    print("\nCreating expert trajectories...")
    expert_train = []
    expert_test = []
    
    for task_id, task in tasks.items():
        if task_id not in ground_truths:
            continue
        
        gt = ground_truths[task_id]
        
        # Determine source file
        source_file = None
        for sf, _ in SOURCE_FILES:
            if task_id.startswith(sf.replace('BFCL_v4_', '').replace('.json', '')):
                source_file = sf
                break
        if not source_file:
            # Try to match by category
            if 'base' in task_id:
                source_file = 'BFCL_v4_multi_turn_base.json'
            elif 'long_context' in task_id:
                source_file = 'BFCL_v4_multi_turn_long_context.json'
            elif 'miss_func' in task_id:
                source_file = 'BFCL_v4_multi_turn_miss_func.json'
            elif 'miss_param' in task_id:
                source_file = 'BFCL_v4_multi_turn_miss_param.json'
            else:
                source_file = 'unknown'
        
        # Create trajectory (with reward token for WM)
        traj = create_expert_trajectory(task, gt, task_id, source_file, add_reward_token=True, is_success=True)
        
        # Split based on RL split
        if task_id in rl_train_ids:
            expert_train.append(traj)
        elif task_id in rl_test_ids:
            expert_test.append(traj)
    
    print(f"  Expert train: {len(expert_train)}")
    print(f"  Expert test: {len(expert_test)}")
    
    # Load existing failure trajectories from the RL data
    print("\nLoading failure trajectories...")
    
    # Load reward data
    with open(BFCL_DATA_DIR / "generated/evaluation/trajectory_reward_results.json") as f:
        reward_data = json.load(f)
    samples = reward_data['samples']
    
    # Load raw trajectories
    with open(BFCL_DATA_DIR / "generated/qwen2.5-7b-instruct/rl/rl_trajectories.json") as f:
        raw_trajs = json.load(f)
    
    # Build question to task_id mapping
    question_to_task = {}
    for task_id, task in tasks.items():
        if task.get('question') and task['question'][0]:
            first_q = task['question'][0][0].get('content', '')[:200]
            question_to_task[first_q] = task_id
    
    # Collect failure trajectories
    failure_train = []
    failure_test = []
    
    for idx, (traj, sample) in enumerate(zip(raw_trajs, samples)):
        reward = sample['sample_avg_reward']
        
        # Only failures (reward < 1.0)
        if reward >= 1.0:
            continue
        
        # Find task_id
        task_id = None
        for msg in traj['conversations']:
            if msg.get('from') == 'human' and not msg['value'].startswith('Environment setup'):
                key = msg['value'][:200]
                task_id = question_to_task.get(key)
                break
        
        if not task_id:
            continue
        
        # Convert trajectory
        messages = []
        messages.append({"role": "system", "content": SYSTEM_PROMPT})
        
        first_user = False
        tool_call_id_counter = 1000
        
        for msg in traj['conversations']:
            role = msg.get('from', '')
            content = msg.get('value', '')
            
            if role == 'human':
                if content.startswith('Environment setup:'):
                    continue
                if not first_user:
                    content += "\n\n[Reward Goal: <|low_reward|>]"
                    first_user = True
                messages.append({"role": "user", "content": content})
            elif role == 'gpt':
                if content.startswith("I understand"):
                    continue
                messages.append({"role": "assistant", "content": content})
            elif role == 'function_call':
                try:
                    func_data = json.loads(content)
                    tc_id = f"call_{tool_call_id_counter}"
                    tool_call_id_counter += 1
                    messages.append({
                        "role": "assistant",
                        "content": "",  # Use empty string instead of None (LLaMA-Factory compatibility)
                        "tool_calls": [{
                            "id": tc_id,
                            "type": "function",
                            "function": {
                                "name": func_data.get("name", "unknown"),
                                "arguments": json.dumps(func_data.get("arguments", {}))
                            }
                        }]
                    })
                except:
                    messages.append({"role": "assistant", "content": content})
            elif role == 'observation':
                messages.append({
                    "role": "tool",
                    "tool_call_id": f"call_{tool_call_id_counter-1}",
                    "content": content
                })
        
        # Ensure tools is JSON string for consistency
        tools_data = traj.get('tools', [])
        if isinstance(tools_data, list):
            tools_str = json.dumps(tools_data, ensure_ascii=False)
        elif isinstance(tools_data, str):
            tools_str = tools_data
        else:
            tools_str = "[]"
        
        failure_item = {
            "messages": messages,
            "tools": tools_str,  # JSON string
            "task_id": task_id,
            "source_file": "rl_trajectory",
            "final_reward": reward
        }
        
        if task_id in rl_train_ids:
            failure_train.append(failure_item)
        elif task_id in rl_test_ids:
            failure_test.append(failure_item)
    
    print(f"  Failure train: {len(failure_train)}")
    print(f"  Failure test: {len(failure_test)}")
    
    # Balance to 50-50
    print("\nBalancing datasets...")
    random.seed(42)
    
    # Sample failures to match expert count
    random.shuffle(failure_train)
    random.shuffle(failure_test)
    
    failure_train_sampled = failure_train[:len(expert_train)]
    failure_test_sampled = failure_test[:len(expert_test)]
    
    # Combine
    wm_train = expert_train + failure_train_sampled
    wm_test = expert_test + failure_test_sampled
    random.shuffle(wm_train)
    random.shuffle(wm_test)
    
    # Create SFT (expert only, no reward token)
    sft_train = []
    sft_test = []
    
    for task_id, task in tasks.items():
        if task_id not in ground_truths:
            continue
        gt = ground_truths[task_id]
        
        # Get source file
        source_file = 'unknown'
        for sf, _ in SOURCE_FILES:
            if task_id.startswith(sf.replace('BFCL_v4_', '').replace('.json', '')):
                source_file = sf
                break
        
        traj = create_expert_trajectory(task, gt, task_id, source_file, add_reward_token=False, is_success=True)
        
        if task_id in rl_train_ids:
            sft_train.append(traj)
        elif task_id in rl_test_ids:
            sft_test.append(traj)
    
    # Save
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    with open(OUTPUT_DIR / "aligned_wm_train.json", 'w') as f:
        json.dump(wm_train, f, indent=2, ensure_ascii=False)
    with open(OUTPUT_DIR / "aligned_wm_test.json", 'w') as f:
        json.dump(wm_test, f, indent=2, ensure_ascii=False)
    with open(OUTPUT_DIR / "aligned_sft_train.json", 'w') as f:
        json.dump(sft_train, f, indent=2, ensure_ascii=False)
    with open(OUTPUT_DIR / "aligned_sft_test.json", 'w') as f:
        json.dump(sft_test, f, indent=2, ensure_ascii=False)
    
    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    
    wm_train_high = sum(1 for s in wm_train if '<|high_reward|>' in str(s))
    wm_train_low = sum(1 for s in wm_train if '<|low_reward|>' in str(s))
    wm_test_high = sum(1 for s in wm_test if '<|high_reward|>' in str(s))
    wm_test_low = sum(1 for s in wm_test if '<|low_reward|>' in str(s))
    
    print(f"""
WM Train: {len(wm_train)} samples
  - Success (<|high_reward|>): {wm_train_high} ({100*wm_train_high/len(wm_train):.1f}%)
  - Failure (<|low_reward|>): {wm_train_low} ({100*wm_train_low/len(wm_train):.1f}%)

WM Test: {len(wm_test)} samples  
  - Success (<|high_reward|>): {wm_test_high} ({100*wm_test_high/len(wm_test):.1f}%)
  - Failure (<|low_reward|>): {wm_test_low} ({100*wm_test_low/len(wm_test):.1f}%)

SFT Train: {len(sft_train)} samples (all expert)
SFT Test: {len(sft_test)} samples (all expert)

Expert trajectories from: BFCL ground truth (possible_answer files)
Failure trajectories from: RL exploration (qwen2.5-7b-instruct)

Saved to: {OUTPUT_DIR}
""")


if __name__ == "__main__":
    main()
