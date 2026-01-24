#!/usr/bin/env python3
"""
Build balanced WM data with ~50-50 success/failure from RL trajectories.

Uses reward threshold to split into success/failure, aligned with RL train/test split.
"""

import json
import random
from pathlib import Path
from collections import defaultdict

# Paths
TRAJ_PATH = Path(__file__).parent.parent / "generated/qwen2.5-7b-instruct/rl/rl_trajectories.json"
REWARD_PATH = Path(__file__).parent.parent / "generated/evaluation/trajectory_reward_results.json"
BFCL_DIR = Path(__file__).parent.parent
RL_TRAIN_PATH = Path("/home/dataset-local/projects/verl/examples/bfcl/data/bfcl_rl_train.json")
RL_TEST_PATH = Path("/home/dataset-local/projects/verl/examples/bfcl/data/bfcl_rl_test.json")
OUTPUT_DIR = Path(__file__).parent.parent / "generated/aligned"

# BFCL source files
SOURCE_FILES = [
    'BFCL_v4_multi_turn_base.json',
    'BFCL_v4_multi_turn_long_context.json',
    'BFCL_v4_multi_turn_miss_func.json',
    'BFCL_v4_multi_turn_miss_param.json'
]

# Reward threshold for success/failure (1.0 = RL binary style: success=1.0, failure<1.0)
REWARD_THRESHOLD = 1.0


def build_question_to_task_mapping():
    """Build mapping from first question to task_id."""
    question_to_task = {}
    task_to_source = {}
    
    for src_file in SOURCE_FILES:
        src_path = BFCL_DIR / src_file
        if not src_path.exists():
            print(f"Warning: {src_file} not found")
            continue
        
        with open(src_path) as f:
            for line in f:
                task = json.loads(line.strip())
                task_id = task['id']
                first_q = task['question'][0][0]['content'] if task['question'] else ''
                key = first_q[:200]
                question_to_task[key] = task_id
                task_to_source[task_id] = src_file
    
    return question_to_task, task_to_source


def convert_to_openai_format(traj, task_id, source_file, reward, is_success):
    """Convert trajectory to OpenAI messages format with reward token."""
    messages = []
    first_user_processed = False
    
    # System message
    messages.append({
        "role": "system",
        "content": "You are a helpful assistant that completes tasks by calling the appropriate tools. Analyze the task requirements carefully, then execute the necessary tool calls in the correct order. You can call multiple tools as needed. When you have completed all required actions, call the done() function to finish the task."
    })
    
    # Convert conversations
    tool_call_id_counter = 1000
    pending_tool_calls = []
    
    for msg in traj['conversations']:
        role = msg.get('from', '')
        content = msg.get('value', '')
        
        if role == 'human':
            # Skip environment setup, keep task instructions
            if content.startswith('Environment setup:'):
                continue
            
            # Add reward token to first actual user message (not env setup)
            if not first_user_processed:
                reward_token = "<|high_reward|>" if is_success else "<|low_reward|>"
                content = f"{content}\n\n[Reward Goal: {reward_token}]"
                first_user_processed = True
            
            messages.append({"role": "user", "content": content})
            
        elif role == 'gpt':
            # Skip the acknowledgment message that follows env setup
            if content.startswith("I understand the environment"):
                continue
            messages.append({"role": "assistant", "content": content})
            
        elif role == 'function_call':
            # Parse function call
            try:
                func_data = json.loads(content)
                tool_call_id = f"call_{tool_call_id_counter}"
                tool_call_id_counter += 1
                
                tool_call = {
                    "id": tool_call_id,
                    "type": "function",
                    "function": {
                        "name": func_data.get("name", "unknown"),
                        "arguments": json.dumps(func_data.get("arguments", {}))
                    }
                }
                pending_tool_calls.append(tool_call)
                
                messages.append({
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [tool_call]
                })
            except:
                messages.append({"role": "assistant", "content": content})
                
        elif role == 'observation':
            # Tool response
            if pending_tool_calls:
                tool_call = pending_tool_calls.pop(0)
                messages.append({
                    "role": "tool",
                    "tool_call_id": tool_call["id"],
                    "content": content
                })
            else:
                messages.append({
                    "role": "tool",
                    "tool_call_id": f"call_{tool_call_id_counter}",
                    "content": content
                })
                tool_call_id_counter += 1
    
    return {
        "messages": messages,
        "tools": traj.get('tools', []),
        "task_id": task_id,
        "source_file": source_file,
        "final_reward": reward
    }


def main():
    print("=" * 60)
    print("Building Balanced WM Data (~50-50 Success/Failure)")
    print("=" * 60)
    
    # Load data
    print("\nLoading data...")
    with open(TRAJ_PATH) as f:
        trajs = json.load(f)
    print(f"  Trajectories: {len(trajs)}")
    
    with open(REWARD_PATH) as f:
        reward_data = json.load(f)
    samples = reward_data['samples']
    print(f"  Reward samples: {len(samples)}")
    
    with open(RL_TRAIN_PATH) as f:
        rl_train = json.load(f)
    with open(RL_TEST_PATH) as f:
        rl_test = json.load(f)
    
    rl_train_ids = set(s['task_id'] for s in rl_train)
    rl_test_ids = set(s['task_id'] for s in rl_test)
    print(f"  RL train tasks: {len(rl_train_ids)}")
    print(f"  RL test tasks: {len(rl_test_ids)}")
    
    # Build task mapping
    print("\nBuilding task mapping...")
    question_to_task, task_to_source = build_question_to_task_mapping()
    print(f"  Mapped {len(question_to_task)} tasks")
    
    # Map trajectories to tasks
    print("\nMapping trajectories to tasks...")
    traj_data = []
    for idx, traj in enumerate(trajs):
        reward = samples[idx]['sample_avg_reward']
        
        # Find task_id
        task_id = None
        for msg in traj['conversations']:
            if msg.get('from') == 'human' and not msg['value'].startswith('Environment setup'):
                key = msg['value'][:200]
                task_id = question_to_task.get(key)
                break
        
        if task_id:
            source_file = task_to_source.get(task_id, "unknown")
            traj_data.append({
                'idx': idx,
                'task_id': task_id,
                'source_file': source_file,
                'reward': reward,
                'traj': traj
            })
    
    print(f"  Mapped {len(traj_data)} trajectories")
    
    # Group by task_id
    by_task = defaultdict(list)
    for item in traj_data:
        by_task[item['task_id']].append(item)
    
    print(f"  Unique tasks: {len(by_task)}")
    
    # Split into train/test based on RL split
    train_items = []
    test_items = []
    
    for task_id, items in by_task.items():
        if task_id in rl_train_ids:
            train_items.extend(items)
        elif task_id in rl_test_ids:
            test_items.extend(items)
    
    print(f"\n  Train items: {len(train_items)}")
    print(f"  Test items: {len(test_items)}")
    
    # Balance success/failure using threshold
    def balance_dataset(items, name):
        success = [i for i in items if i['reward'] >= REWARD_THRESHOLD]
        failure = [i for i in items if i['reward'] < REWARD_THRESHOLD]
        
        print(f"\n{name} before balancing:")
        print(f"  Success (>= {REWARD_THRESHOLD}): {len(success)}")
        print(f"  Failure (< {REWARD_THRESHOLD}): {len(failure)}")
        
        # Balance to 50-50
        min_count = min(len(success), len(failure))
        random.seed(42)
        random.shuffle(success)
        random.shuffle(failure)
        
        balanced = success[:min_count] + failure[:min_count]
        random.shuffle(balanced)
        
        print(f"  After balancing: {len(balanced)} ({min_count} success + {min_count} failure)")
        return balanced
    
    train_balanced = balance_dataset(train_items, "Train")
    test_balanced = balance_dataset(test_items, "Test")
    
    # Convert to OpenAI format
    print("\nConverting to OpenAI format...")
    
    def convert_items(items):
        result = []
        for item in items:
            is_success = item['reward'] >= REWARD_THRESHOLD
            converted = convert_to_openai_format(
                item['traj'],
                item['task_id'],
                item['source_file'],
                item['reward'],
                is_success
            )
            result.append(converted)
        return result
    
    wm_train = convert_items(train_balanced)
    wm_test = convert_items(test_balanced)
    
    # Also create SFT (success only, no reward tokens)
    sft_train = []
    sft_test = []
    
    for item in train_balanced:
        if item['reward'] >= REWARD_THRESHOLD:
            converted = convert_to_openai_format(
                item['traj'], item['task_id'], item['source_file'],
                item['reward'], True
            )
            # Remove reward token for SFT
            for msg in converted['messages']:
                if msg.get('role') == 'user' and '[Reward Goal:' in msg.get('content', ''):
                    msg['content'] = msg['content'].split('\n\n[Reward Goal:')[0]
            sft_train.append(converted)
    
    for item in test_balanced:
        if item['reward'] >= REWARD_THRESHOLD:
            converted = convert_to_openai_format(
                item['traj'], item['task_id'], item['source_file'],
                item['reward'], True
            )
            for msg in converted['messages']:
                if msg.get('role') == 'user' and '[Reward Goal:' in msg.get('content', ''):
                    msg['content'] = msg['content'].split('\n\n[Reward Goal:')[0]
            sft_test.append(converted)
    
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
    print(f"WM Train: {len(wm_train)} samples (50% success, 50% failure)")
    print(f"WM Test: {len(wm_test)} samples (50% success, 50% failure)")
    print(f"SFT Train: {len(sft_train)} samples (success only)")
    print(f"SFT Test: {len(sft_test)} samples (success only)")
    print(f"\nSaved to: {OUTPUT_DIR}")
    
    # Verify
    wm_train_high = sum(1 for s in wm_train if '<|high_reward|>' in str(s))
    wm_train_low = sum(1 for s in wm_train if '<|low_reward|>' in str(s))
    print(f"\nVerification - WM Train reward tokens:")
    print(f"  <|high_reward|>: {wm_train_high}")
    print(f"  <|low_reward|>: {wm_train_low}")


if __name__ == "__main__":
    main()
