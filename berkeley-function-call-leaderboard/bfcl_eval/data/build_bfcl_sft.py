#!/usr/bin/env python3
"""
Build BFCL SFT Dataset

Converts BFCL v4 multi-turn JSON files into alpaca-like SFT dataset format.

Key behavior: Each multi-turn BFCL object is expanded into MULTIPLE SFT records,
one per step/turn. For example, a 4-turn task creates 4 SFT samples:
- Step 1: context=(turn 1 + initial state), output=(action for turn 1)
- Step 2: context=(turns 1-2 + action 1 + state), output=(action for turn 2)
- Step 3: context=(turns 1-3 + actions 1-2 + state), output=(action for turn 3)
- Step 4: context=(turns 1-4 + actions 1-3 + state), output=(action for turn 4)

Each SFT record contains:
- instruction: High-level task description (same across all steps from one example)
- input: Full cumulative context up to step t (user requests, initial state, 
         previous actions, available tools, answer format instructions)
- output: Python-style list of actions for step t, e.g., "[action1, action2, action3]"
- system: Generic role description
- history: List of [user_request, model_output] tuples from all previous turns (empty for turn 1)
"""

import argparse
import json
import logging
from pathlib import Path
from typing import Dict, List, Any, Optional


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


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


def load_tool_docs(involved_classes: List[str], func_doc_dir: Path) -> Dict[str, Any]:
    """
    Load tool documentation for involved classes.
    
    Args:
        involved_classes: List of class names (e.g., ["GorillaFileSystem", "TwitterAPI"])
        func_doc_dir: Path to multi_turn_func_doc directory
        
    Returns:
        Dictionary mapping tool names to their documentation
    """
    tool_docs = {}
    
    for class_name in involved_classes:
        # Handle special case for MemoryAPI variants
        doc_file = CLASS_TO_DOC_FILE.get(class_name)
        if not doc_file and class_name.startswith("MemoryAPI"):
            # Try generic MemoryAPI
            doc_file = CLASS_TO_DOC_FILE.get(f"MemoryAPI_{class_name.split('_')[-1]}")
        
        if not doc_file:
            logger.warning(f"No documentation file mapping for class: {class_name}")
            continue
            
        doc_path = func_doc_dir / doc_file
        if not doc_path.exists():
            logger.warning(f"Documentation file not found: {doc_path}")
            continue
            
        try:
            with open(doc_path, 'r', encoding='utf-8') as f:
                # Each line is a separate JSON object (tool definition)
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    tool_def = json.loads(line)
                    tool_name = tool_def.get('name')
                    if tool_name:
                        # Store with both simple name and class.name format
                        tool_docs[tool_name] = tool_def
                        tool_docs[f"{class_name}.{tool_name}"] = tool_def
        except Exception as e:
            logger.error(f"Error loading tool docs from {doc_path}: {e}")
            
    return tool_docs


def load_ground_truth(obj_id: str, input_file_path: Path) -> Optional[List[List[str]]]:
    """
    Load ground truth for a specific object ID.
    
    Args:
        obj_id: Object ID (e.g., "multi_turn_base_0")
        input_file_path: Path to the input BFCL file
        
    Returns:
        Ground truth list of turns, each turn containing a list of function calls
    """
    # Construct the corresponding possible_answer file path
    # E.g., BFCL_v4_multi_turn_base.json -> possible_answer/BFCL_v4_multi_turn_base.json
    input_filename = input_file_path.name
    possible_answer_dir = input_file_path.parent / "possible_answer"
    ground_truth_path = possible_answer_dir / input_filename
    
    if not ground_truth_path.exists():
        logger.warning(f"Ground truth file not found: {ground_truth_path}")
        return None
        
    try:
        with open(ground_truth_path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                gt_obj = json.loads(line)
                if gt_obj.get('id') == obj_id:
                    return gt_obj.get('ground_truth', [])
    except Exception as e:
        logger.error(f"Error loading ground truth from {ground_truth_path}: {e}")
        
    return None


def format_initial_config(config: Dict[str, Any]) -> str:
    """
    Convert initial_config dict to natural language summary.
    
    Args:
        config: Initial configuration dictionary
        
    Returns:
        Natural language description of the initial state
    """
    parts = []
    
    for class_name, class_config in config.items():
        if class_name == "GorillaFileSystem":
            parts.append(format_filesystem(class_config))
        elif class_name == "TwitterAPI":
            parts.append(format_twitter_state(class_config))
        elif class_name == "TicketAPI":
            parts.append(format_ticket_state(class_config))
        elif class_name == "MessageAPI":
            parts.append(format_message_state(class_config))
        elif class_name == "TravelAPI":
            parts.append(format_travel_state(class_config))
        elif class_name == "TradingBot":
            parts.append(format_trading_state(class_config))
        elif class_name == "VehicleControlAPI":
            parts.append(format_vehicle_state(class_config))
        elif "MemoryAPI" in class_name:
            parts.append(format_memory_state(class_config, class_name))
        else:
            # Generic fallback
            parts.append(f"**{class_name}**: {json.dumps(class_config, indent=2)}")
    
    return "\n\n".join(parts) if parts else "No initial configuration provided."


def format_filesystem_compact(fs_config: Dict[str, Any]) -> List[str]:
    """Format GorillaFileSystem configuration compactly (for environment summary)."""
    items = []
    
    def collect_items(node: Dict, path: str = ""):
        """Recursively collect file/dir info."""
        if "root" in node:
            return collect_items(node["root"], "/")
            
        for name, item in node.items():
            current_path = f"{path}{name}" if path == "/" else f"{path}/{name}"
            
            if isinstance(item, dict):
                if item.get("type") == "directory":
                    contents = item.get("contents", {})
                    if contents:
                        collect_items(contents, current_path + "/")
                elif item.get("type") == "file":
                    content = item.get("content", "")
                    # Very short preview
                    preview = content[:30] + "..." if len(content) > 30 else content
                    items.append(f"{current_path} (file)")
                else:
                    # Generic dict - treat as directory
                    collect_items(item, current_path + "/")
    
    collect_items(fs_config)
    
    # Return concise summary
    if len(items) <= 5:
        return items
    else:
        # Show first few and count
        return items[:3] + [f"... and {len(items) - 3} more files"]


def format_twitter_state_compact(twitter_config: Dict[str, Any]) -> str:
    """Format TwitterAPI state compactly."""
    username = twitter_config.get("username", "N/A")
    authenticated = twitter_config.get("authenticated", False)
    tweet_count = twitter_config.get("tweet_counter", 0)
    tweets = twitter_config.get("tweets", {})
    
    parts = [f"Twitter: logged in as {username}"]
    if tweets:
        parts.append(f"{len(tweets)} existing tweets")
    
    return ", ".join(parts)


def format_environment_summary_compact(config: Dict[str, Any]) -> List[str]:
    """
    Create a compact environment summary (1-3 bullet lines).
    
    Args:
        config: Initial configuration dictionary
        
    Returns:
        List of concise bullet points describing the environment
    """
    lines = []
    
    for class_name, class_config in config.items():
        if class_name == "GorillaFileSystem":
            file_items = format_filesystem_compact(class_config)
            if file_items:
                lines.append(f"Filesystem: {', '.join(file_items[:2])}" + 
                           (f", and {len(file_items) - 2} more" if len(file_items) > 2 else ""))
        elif class_name == "TwitterAPI":
            twitter_summary = format_twitter_state_compact(class_config)
            lines.append(twitter_summary)
        elif class_name in ["TicketAPI", "MessageAPI", "TravelAPI", "TradingBot", 
                           "VehicleControlAPI"] or "MemoryAPI" in class_name:
            lines.append(f"{class_name} is available")
    
    return lines if lines else ["No initial environment configuration"]


def format_available_tools_compact(
    tool_paths: List[str],
    tool_docs: Dict[str, Any]
) -> List[str]:
    """
    Format available tools compactly (one short line per tool).
    
    Args:
        tool_paths: List of tool paths (e.g., ["GorillaFileSystem.mv", ...])
        tool_docs: Tool documentation dictionary
        
    Returns:
        List of tool descriptions (one line each)
    """
    lines = []
    
    for tool_path in tool_paths:
        # Get tool documentation
        tool_doc = tool_docs.get(tool_path)
        if not tool_doc:
            # Try without class prefix
            tool_name = tool_path.split('.')[-1]
            tool_doc = tool_docs.get(tool_name)
        
        if tool_doc:
            tool_name = tool_doc.get('name', tool_path.split('.')[-1])
            description = tool_doc.get('description', '')
            
            # Extract core description and shorten
            if "Tool description:" in description:
                description = description.split("Tool description:")[-1].strip()
            
            # Take only first sentence and lowercase
            first_sentence = description.split('.')[0] if '.' in description else description
            first_sentence = first_sentence.strip().lower()
            
            # Very short gloss
            if len(first_sentence) > 60:
                first_sentence = first_sentence[:57] + "..."
            
            lines.append(f"{tool_name}: {first_sentence}")
        else:
            tool_name = tool_path.split('.')[-1]
            lines.append(f"{tool_name}: (no description)")
    
    return lines


def format_action_list(func_calls: List[str]) -> str:
    """
    Format a list of function calls as a Python-style list string.
    
    Args:
        func_calls: List of function calls for this turn
        
    Returns:
        String representation of a list of actions, e.g., "[action1, action2, action3]"
    """
    if not func_calls:
        return "[]"
    
    # Simply join the function calls with commas and wrap in brackets
    return "[" + ", ".join(func_calls) + "]"


def format_action_history_compact(
    previous_actions: List[List[str]]
) -> List[str]:
    """
    Format the history of previous actions taken (very brief).
    
    Args:
        previous_actions: List of previous turns' actions
        
    Returns:
        List of brief action summaries
    """
    lines = []
    
    for turn_idx, turn_actions in enumerate(previous_actions, 1):
        if not turn_actions:
            lines.append(f"Turn {turn_idx}: no action")
            continue
        
        # Very brief summary - just list the function names
        func_names = []
        for func_call in turn_actions:
            func_name = func_call.split('(')[0] if '(' in func_call else func_call
            func_names.append(func_name)
        
        if len(func_names) == 1:
            lines.append(f"Turn {turn_idx}: called {func_names[0]}")
        else:
            lines.append(f"Turn {turn_idx}: called {', '.join(func_names)}")
    
    return lines


def parse_function_call(func_call: str) -> tuple[str, str]:
    """
    Parse a function call string into name and arguments.
    
    Args:
        func_call: Function call string (e.g., "cd(folder='document')")
        
    Returns:
        Tuple of (function_name, arguments_string)
    """
    if '(' not in func_call:
        return func_call, ""
    
    func_name = func_call[:func_call.index('(')]
    args_str = func_call[func_call.index('('):]
    
    return func_name, args_str


def format_user_requests_compact(questions: List[List[Dict[str, str]]], up_to_turn: Optional[int] = None) -> List[str]:
    """
    Format user questions from turns compactly (numbered list, no markdown).
    
    Args:
        questions: List of turns, each containing list of message dicts
        up_to_turn: If specified, only include turns 1 through up_to_turn (1-indexed)
        
    Returns:
        List of user requests (one per turn)
    """
    lines = []
    turns_to_show = questions[:up_to_turn] if up_to_turn else questions
    
    for turn_idx, turn in enumerate(turns_to_show, 1):
        for msg in turn:
            if msg.get("role") == "user":
                content = msg.get("content", "")
                lines.append(f"{turn_idx}) {content}")
                break
    
    return lines


def extract_task_summary(questions: List[List[Dict[str, str]]]) -> str:
    """
    Extract a short task summary from the user questions.
    
    Args:
        questions: List of turns with user messages
        
    Returns:
        One-sentence summary of the overall task
    """
    # Use the first user message as a base
    if questions and len(questions) > 0:
        for msg in questions[0]:
            if msg.get("role") == "user":
                first_request = msg.get("content", "")
                # Shorten if too long
                if len(first_request) > 80:
                    return first_request[:77] + "..."
                return first_request
    
    return "Complete the requested multi-step operations."


def extract_user_request_for_turn(questions: List[List[Dict[str, str]]], turn_idx: int) -> str:
    """
    Extract the user request for a specific turn (0-indexed).
    
    Args:
        questions: List of turns with user messages
        turn_idx: Turn index (0-indexed)
        
    Returns:
        User request string for that turn
    """
    if turn_idx >= len(questions):
        return "Continue with the task."
    
    for msg in questions[turn_idx]:
        if msg.get("role") == "user":
            return msg.get("content", "Continue with the task.")
    
    return "Continue with the task."


def create_sft_records(
    bfcl_obj: Dict[str, Any],
    tool_docs: Dict[str, Any],
    ground_truth: Optional[List[List[str]]],
    input_file_path: Path
) -> List[Dict[str, Any]]:
    """
    Convert a BFCL object into multiple SFT training records (one per step).
    Uses compact plain-text template format for input field.
    Output is a Python-style list of actions: [action1, action2, action3]
    
    History field is populated with [user_request, model_output] tuples from all previous turns.
    For turn 1, history is empty. For turn N, history contains turns 1 through N-1.
    
    Args:
        bfcl_obj: BFCL object dictionary
        tool_docs: Tool documentation dictionary
        ground_truth: Ground truth function calls
        input_file_path: Path to input file (for context)
        
    Returns:
        List of SFT record dictionaries (one per step/turn)
    """
    obj_id = bfcl_obj.get("id")
    questions = bfcl_obj.get("question", [])
    initial_config = bfcl_obj.get("initial_config", {})
    tool_paths = bfcl_obj.get("path", [])
    involved_classes = bfcl_obj.get("involved_classes", [])
    excluded_funcs = bfcl_obj.get("excluded_function", [])
    
    # Validate required fields
    if not questions:
        logger.warning(f"Skipping {obj_id}: No questions found")
        return []
    
    if not ground_truth:
        logger.warning(f"Skipping {obj_id}: No ground truth found")
        return []
    
    # Create instruction (same for all steps from this example)
    instruction = "Complete the multi-step file operations and API calls as requested by the user."
    
    # Create system prompt (same for all steps)
    system_prompt = "You are a tool-using assistant that can operate on a virtual filesystem and external APIs as described in the context."
    
    # Pre-compute components that are the same for all steps
    task_summary = extract_task_summary(questions)
    env_summary = format_environment_summary_compact(initial_config)
    tools_list = format_available_tools_compact(tool_paths, tool_docs)
    total_turns = len(ground_truth)
    
    # Generate one SFT record per step
    sft_records = []
    
    for step_idx in range(total_turns):
        # step_idx is 0-indexed, but turn numbers are 1-indexed
        current_turn = step_idx + 1
        
        # Build input using the compact template
        input_sections = []
        
        # 1. Task
        input_sections.append(f"Task:\n{task_summary}")
        
        # 2. Current turn
        input_sections.append(f"Current turn:\nTurn {current_turn} of {total_turns}")
        
        # 3. User requests so far
        user_requests = format_user_requests_compact(questions, up_to_turn=current_turn)
        input_sections.append("User requests so far:\n" + "\n".join(user_requests))
        
        # 4. Previous actions (omit at Turn 1)
        if step_idx > 0:
            previous_actions = ground_truth[:step_idx]
            action_history = format_action_history_compact(previous_actions)
            input_sections.append("Previous actions:\n- " + "\n- ".join(action_history))
        
        # 5. Environment summary
        input_sections.append("Environment summary:\n- " + "\n- ".join(env_summary))
        
        # 6. Available tools
        input_sections.append("Available tools:\n- " + "\n- ".join(tools_list))
        
        # 7. Goal for this step
        input_sections.append("Goal for this step:\nDecide the single next action to take at the current turn.")
        
        # 8. Answer format instruction
        answer_format = (
            "Answer format:\n"
            "Output the next action or sequence of actions as a Python-style list of tool calls, for example:\n"
            "[tool-calling1, tool-calling2, tool-calling3]\n"
            "where each tool-calling is written as TOOL_NAME(arg1=value1, arg2=value2, ...), "
            "and you do NOT add any extra explanations or text outside the list."
        )
        input_sections.append(answer_format)
        
        # Join all sections with double newline
        input_text = "\n\n".join(input_sections)
        
        # Create output (list of actions for this step)
        current_actions = ground_truth[step_idx]
        output_text = format_action_list(current_actions)
        
        # Build history from previous turns (empty for first turn)
        history = []
        for prev_turn_idx in range(step_idx):
            # Get the user request for this previous turn
            prev_instruction = extract_user_request_for_turn(questions, prev_turn_idx)
            # Get the model output for this previous turn
            prev_output = format_action_list(ground_truth[prev_turn_idx])
            # Add as a tuple [instruction, output]
            history.append([prev_instruction, prev_output])
        
        # Create SFT record for this step
        sft_record = {
            "instruction": instruction,
            "input": input_text,
            "output": output_text,
            "system": system_prompt,
            "history": history
        }
        
        sft_records.append(sft_record)
    
    return sft_records


def process_bfcl_file(
    file_path: Path,
    tool_docs: Dict[str, Any],
    stats: Dict[str, int]
) -> List[Dict[str, Any]]:
    """
    Process a single BFCL JSON file.
    
    Args:
        file_path: Path to BFCL JSON file
        tool_docs: Tool documentation dictionary
        stats: Statistics dictionary to update
        
    Returns:
        List of SFT records
    """
    sft_records = []
    stats['files_processed'] += 1
    
    logger.info(f"Processing file: {file_path.name}")
    
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            # Handle JSONL format (one object per line)
            for line_num, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue
                
                try:
                    bfcl_obj = json.loads(line)
                    obj_id = bfcl_obj.get("id", f"unknown_{line_num}")
                    
                    # Load ground truth
                    ground_truth = load_ground_truth(obj_id, file_path)
                    
                    # Load tool docs for this object's involved classes
                    involved_classes = bfcl_obj.get("involved_classes", [])
                    
                    # Create SFT records (one per step)
                    step_records = create_sft_records(bfcl_obj, tool_docs, ground_truth, file_path)
                    
                    if step_records:
                        sft_records.extend(step_records)
                        stats['records_created'] += len(step_records)
                    else:
                        stats['objects_skipped'] += 1
                        
                except json.JSONDecodeError as e:
                    logger.error(f"JSON decode error in {file_path.name} line {line_num}: {e}")
                    stats['objects_skipped'] += 1
                except Exception as e:
                    logger.error(f"Error processing object in {file_path.name} line {line_num}: {e}")
                    stats['objects_skipped'] += 1
                    
    except Exception as e:
        logger.error(f"Error reading file {file_path}: {e}")
    
    return sft_records


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Convert BFCL v4 multi-turn JSON files to SFT dataset format"
    )
    parser.add_argument(
        "--input_dir",
        type=str,
        required=True,
        help="Directory containing BFCL_v4_multi_turn_*.json files"
    )
    parser.add_argument(
        "--output_path",
        type=str,
        required=True,
        help="Output path for SFT dataset JSON file"
    )
    
    args = parser.parse_args()
    
    input_dir = Path(args.input_dir)
    output_path = Path(args.output_path)
    
    # Validate input directory
    if not input_dir.exists():
        logger.error(f"Input directory does not exist: {input_dir}")
        return 1
    
    # Find all BFCL multi-turn files
    bfcl_files = list(input_dir.rglob("BFCL_v4_multi_turn_*.json"))
    
    # Exclude possible_answer directory and unused_datasets directory
    bfcl_files = [f for f in bfcl_files if "possible_answer" not in str(f) and "unused_datasets" not in str(f)]
    
    if not bfcl_files:
        logger.error(f"No BFCL_v4_multi_turn_*.json files found in {input_dir}")
        return 1
    
    logger.info(f"Found {len(bfcl_files)} BFCL files to process")
    
    # Load tool documentation (use directory from first file)
    func_doc_dir = input_dir / "multi_turn_func_doc"
    if not func_doc_dir.exists():
        # Try relative to script location
        script_dir = Path(__file__).parent.parent
        func_doc_dir = script_dir / "berkeley-function-call-leaderboard/bfcl_eval/data/multi_turn_func_doc"
    
    if not func_doc_dir.exists():
        logger.error(f"Tool documentation directory not found: {func_doc_dir}")
        return 1
    
    # Load all tool documentation (we'll load all classes)
    all_classes = list(CLASS_TO_DOC_FILE.keys())
    tool_docs = load_tool_docs(all_classes, func_doc_dir)
    logger.info(f"Loaded documentation for {len(tool_docs)} tools")
    
    # Statistics
    stats = {
        'files_processed': 0,
        'records_created': 0,
        'objects_skipped': 0
    }
    
    # Process all files
    all_sft_records = []
    for bfcl_file in sorted(bfcl_files):
        records = process_bfcl_file(bfcl_file, tool_docs, stats)
        all_sft_records.extend(records)
    
    # Write output
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(all_sft_records, f, ensure_ascii=False, indent=2)
    
    # Log final statistics
    logger.info("=" * 60)
    logger.info("Conversion complete!")
    logger.info(f"Files processed: {stats['files_processed']}")
    logger.info(f"SFT records created: {stats['records_created']} (multi-step expansion)")
    logger.info(f"Objects skipped: {stats['objects_skipped']}")
    logger.info(f"Output written to: {output_path}")
    logger.info(f"Note: Each multi-turn example was expanded into multiple SFT records (one per step)")
    logger.info("=" * 60)
    
    return 0


if __name__ == "__main__":
    exit(main())

