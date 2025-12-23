#!/usr/bin/env python3
"""
Build BFCL Glaive Dataset

Converts BFCL v4 multi-turn JSON files into Glaive toolcall dataset format.

Supports two output modes (selectable via --mode argument):

1. Cumulative mode (default):
   Each N-turn BFCL object is converted into N cumulative Glaive records,
   where each record contains the conversation history up to that turn. This enables
   training models to predict tool calls for each turn given the full context so far.
   
   For example, a 4-turn conversation generates:
   - Record 1: Environment setup + Turn 1
   - Record 2: Environment setup + Turn 1 + Turn 2
   - Record 3: Environment setup + Turn 1 + Turn 2 + Turn 3
   - Record 4: Environment setup + Turn 1 + Turn 2 + Turn 3 + Turn 4

2. Single mode:
   Each N-turn BFCL object is converted into 1 Glaive record with all turns merged
   into a single conversation. This is useful for training models on complete
   multi-turn conversations.
   
   For example, a 4-turn conversation generates:
   - Record 1: Environment setup + Turn 1 + Turn 2 + Turn 3 + Turn 4

Each Glaive record contains:
- conversations: Array of conversation turns with "from" (human/gpt/function_call/observation) 
                 and "value" fields. Includes:
  * Initial environment context (if present)
  * For each turn: human message, function_call(s), observation(s), gpt response
- tools: JSON string of tool definitions in OpenAI format

Tool documentation is loaded from berkeley-function-call-leaderboard/bfcl_eval/data/multi_turn_func_doc/
which contains detailed descriptions and parameter schemas for all available tools.

Function calls are converted from Python-style format (e.g., "cd(folder='temp')")
to JSON format (e.g., {"name": "cd", "arguments": {"folder": "temp"}}).
"""

import argparse
import json
import logging
from pathlib import Path
from typing import Dict, List, Any, Optional

# Import the environment for generating real observations
import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from bfcl_eval.env import ToolCallingEnvironment


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
        # Use compact JSON formatting without markdown, matching Glaive plain text style
        parts.append(f"{class_name}: {json.dumps(class_config)}")
    
    return "\n".join(parts) if parts else "No initial configuration provided."


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
    
    # Return all items without truncation
    return items


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
                # Include all file items without truncation
                lines.append(f"Filesystem: {', '.join(file_items)}")
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


def format_tool_full_spec(tool_doc: Dict[str, Any]) -> str:
    """
    Format a single tool's full specification including all parameters.
    
    Args:
        tool_doc: Tool documentation dictionary
        
    Returns:
        Complete formatted tool specification
    """
    lines = []
    
    # Tool name
    tool_name = tool_doc.get('name', 'unknown')
    lines.append(f"{tool_name}")
    
    # Description
    description = tool_doc.get('description', '')
    if "Tool description:" in description:
        description = description.split("Tool description:")[-1].strip()
    lines.append(f"  Description: {description}")
    
    # Parameters
    params = tool_doc.get('parameters', {})
    properties = params.get('properties', {})
    required = params.get('required', [])
    
    if properties:
        lines.append("  Parameters:")
        for param_name, param_info in properties.items():
            param_type = param_info.get('type', 'any')
            param_desc = param_info.get('description', 'No description')
            param_default = param_info.get('default')
            is_required = param_name in required
            
            # Format parameter line
            param_line = f"    - {param_name} ({param_type})"
            if is_required:
                param_line += " [required]"
            else:
                param_line += " [optional"
                if param_default is not None:
                    param_line += f", default={param_default}"
                param_line += "]"
            param_line += f": {param_desc}"
            lines.append(param_line)
    else:
        lines.append("  Parameters: None")
    
    return "\n".join(lines)


def format_available_tools_full(
    tool_paths: List[str],
    tool_docs: Dict[str, Any]
) -> str:
    """
    Format available tools with complete specifications for system prompt.
    
    Args:
        tool_paths: List of tool paths (e.g., ["GorillaFileSystem.mv", ...])
        tool_docs: Tool documentation dictionary
        
    Returns:
        Complete formatted tool list as a single string
    """
    tool_specs = []
    
    for tool_path in tool_paths:
        # Get tool documentation
        tool_doc = tool_docs.get(tool_path)
        if not tool_doc:
            # Try without class prefix
            tool_name = tool_path.split('.')[-1]
            tool_doc = tool_docs.get(tool_name)
        
        if tool_doc:
            tool_specs.append(format_tool_full_spec(tool_doc))
        else:
            tool_name = tool_path.split('.')[-1]
            tool_specs.append(f"{tool_name}\n  Description: No documentation available")
    
    return "\n\n".join(tool_specs)


def format_tools_as_json_string(
    tool_paths: List[str],
    tool_docs: Dict[str, Any]
) -> str:
    """
    Format available tools as a JSON string in Glaive format.
    
    Args:
        tool_paths: List of tool paths (e.g., ["GorillaFileSystem.mv", ...])
        tool_docs: Tool documentation dictionary loaded from multi_turn_func_doc/
        
    Returns:
        JSON string representation of tools array in Glaive format
        Example: '[{"name": "cd", "description": "...", "parameters": {...}}, ...]'
    """
    tools_list = []
    
    for tool_path in tool_paths:
        # Get tool documentation
        tool_doc = tool_docs.get(tool_path)
        if not tool_doc:
            # Try without class prefix
            tool_name = tool_path.split('.')[-1]
            tool_doc = tool_docs.get(tool_name)
        
        if tool_doc:
            # Convert BFCL tool doc format to Glaive format
            glaive_tool = {
                "name": tool_doc.get("name", tool_path.split('.')[-1]),
                "description": tool_doc.get("description", "")
            }
            
            # Convert parameters schema
            params = tool_doc.get("parameters", {})
            if params:
                # Convert "type": "dict" to "type": "object" for Glaive format
                glaive_params = {
                    "type": "object",
                    "properties": params.get("properties", {}),
                }
                
                # Add required fields if present
                if "required" in params:
                    glaive_params["required"] = params["required"]
                
                glaive_tool["parameters"] = glaive_params
            else:
                # No parameters
                glaive_tool["parameters"] = {
                    "type": "object",
                    "properties": {}
                }
            
            tools_list.append(glaive_tool)
        else:
            # Fallback for missing documentation
            tool_name = tool_path.split('.')[-1]
            tools_list.append({
                "name": tool_name,
                "description": "No documentation available",
                "parameters": {
                    "type": "object",
                    "properties": {}
                }
            })
    
    # Return as JSON string
    return json.dumps(tools_list)


def parse_tool_call(call_str: str) -> tuple[str, dict]:
    """
    Parse a tool call string into tool name and parameters.
    
    Args:
        call_str: Tool call string, e.g., "cd(folder='temp')"
        
    Returns:
        Tuple of (tool_name, parameters_dict)
    """
    if '(' not in call_str:
        return call_str, {}
    
    tool_name = call_str[:call_str.index('(')]
    params_str = call_str[call_str.index('(') + 1:call_str.rindex(')')]
    
    # Parse parameters
    params = {}
    if params_str.strip():
        # Simple parsing - split by comma, then by =
        for param in params_str.split(','):
            if '=' in param:
                key, value = param.split('=', 1)
                params[key.strip()] = value.strip()
    
    return tool_name, params


def convert_function_call_to_glaive(func_call: str, tool_docs: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """
    Convert a Python-style function call to Glaive JSON format.
    
    Handles both keyword arguments (arg=value) and positional arguments (value).
    For positional arguments, uses tool documentation to map to parameter names.
    
    Args:
        func_call: Function call string, e.g., "cd(folder='temp')" or "get_city('NYC')"
        tool_docs: Optional tool documentation to resolve positional argument names
        
    Returns:
        Dictionary with "name" and "arguments" keys in Glaive format
        Example: {"name": "cd", "arguments": {"folder": "temp"}}
    """
    if '(' not in func_call:
        # Function with no parameters
        return {"name": func_call.strip(), "arguments": {}}
    
    # Extract function name
    func_name = func_call[:func_call.index('(')].strip()
    params_str = func_call[func_call.index('(') + 1:func_call.rindex(')')].strip()
    
    # Parse parameters into a dictionary
    arguments = {}
    if params_str:
        # First pass: parse all parameters (both keyword and positional)
        parsed_params = []  # List of (key, value) tuples; key is None for positional
        current_param = ""
        in_quotes = False
        quote_char = None
        bracket_depth = 0
        
        for char in params_str + ',':  # Add comma at end to process last parameter
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
                # End of parameter
                if current_param.strip():
                    if '=' in current_param:
                        # Keyword argument
                        key, value = current_param.split('=', 1)
                        parsed_params.append((key.strip(), value.strip()))
                    else:
                        # Positional argument
                        parsed_params.append((None, current_param.strip()))
                current_param = ""
            else:
                current_param += char
        
        # Second pass: resolve positional arguments to parameter names
        positional_param_names = None
        if any(key is None for key, _ in parsed_params):
            # We have positional arguments - need to resolve parameter names
            if tool_docs:
                # Try to find tool documentation for this function
                tool_doc = tool_docs.get(func_name)
                if tool_doc and 'parameters' in tool_doc:
                    params_schema = tool_doc['parameters']
                    if 'properties' in params_schema:
                        # Get parameter names in order (use required first, then optional)
                        required_params = params_schema.get('required', [])
                        all_params = list(params_schema['properties'].keys())
                        # Order: required params first, then remaining params
                        positional_param_names = required_params + [p for p in all_params if p not in required_params]
        
        # Build final arguments dictionary
        positional_index = 0
        for key, value in parsed_params:
            if key is not None:
                # Keyword argument - use as-is
                arguments[key] = parse_parameter_value(value)
            else:
                # Positional argument - map to parameter name
                if positional_param_names and positional_index < len(positional_param_names):
                    param_name = positional_param_names[positional_index]
                    arguments[param_name] = parse_parameter_value(value)
                    positional_index += 1
                else:
                    # Fallback: use generic names if we can't resolve
                    arguments[f"arg{positional_index}"] = parse_parameter_value(value)
                    positional_index += 1
    
    return {"name": func_name, "arguments": arguments}


def parse_parameter_value(value: str) -> Any:
    """
    Parse a parameter value string and convert to appropriate Python type.
    
    Args:
        value: String representation of parameter value
        
    Returns:
        Parsed value with appropriate type (str, int, float, bool, list, dict)
    """
    value = value.strip()
    
    # Handle quoted strings
    if (value.startswith("'") and value.endswith("'")) or \
       (value.startswith('"') and value.endswith('"')):
        return value[1:-1]  # Remove quotes
    
    # Handle booleans
    if value == 'True':
        return True
    if value == 'False':
        return False
    
    # Handle None
    if value == 'None':
        return None
    
    # Handle numbers
    try:
        if '.' in value:
            return float(value)
        else:
            return int(value)
    except ValueError:
        pass
    
    # Handle lists
    if value.startswith('[') and value.endswith(']'):
        try:
            # Use json.loads for safe parsing
            import ast
            return ast.literal_eval(value)
        except:
            return value
    
    # Handle dicts
    if value.startswith('{') and value.endswith('}'):
        try:
            import ast
            return ast.literal_eval(value)
        except:
            return value
    
    # Default: return as string
    return value


def convert_positional_to_keyword_args(func_call: str, tool_docs: Optional[Dict[str, Any]] = None) -> str:
    """
    Convert positional arguments to keyword arguments in a function call string.
    
    This is needed for the environment's step() method which expects keyword arguments.
    
    Args:
        func_call: Function call string with potential positional args, 
                  e.g., "get_city('NYC')" or "add(5, 10)"
        tool_docs: Tool documentation to resolve parameter names
        
    Returns:
        Function call string with all arguments as keyword arguments,
        e.g., "get_city(city='NYC')" or "add(a=5, b=10)"
    """
    if '(' not in func_call:
        # No arguments
        return func_call
    
    # Extract function name
    func_name = func_call[:func_call.index('(')].strip()
    params_str = func_call[func_call.index('(') + 1:func_call.rindex(')')].strip()
    
    if not params_str:
        # Empty arguments
        return func_call
    
    # Check if already all keyword arguments
    # Parse parameters to detect positional vs keyword
    parsed_params = []  # List of (key, value) tuples; key is None for positional
    current_param = ""
    in_quotes = False
    quote_char = None
    bracket_depth = 0
    
    for char in params_str + ',':  # Add comma at end to process last parameter
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
            # End of parameter
            if current_param.strip():
                if '=' in current_param:
                    # Already keyword argument
                    key, value = current_param.split('=', 1)
                    parsed_params.append((key.strip(), value.strip()))
                else:
                    # Positional argument
                    parsed_params.append((None, current_param.strip()))
            current_param = ""
        else:
            current_param += char
    
    # Check if any positional arguments exist
    has_positional = any(key is None for key, _ in parsed_params)
    if not has_positional:
        # All arguments are already keyword arguments
        return func_call
    
    # Need to resolve positional arguments to parameter names
    positional_param_names = None
    if tool_docs:
        # Try to find tool documentation for this function
        tool_doc = tool_docs.get(func_name)
        if tool_doc and 'parameters' in tool_doc:
            params_schema = tool_doc['parameters']
            if 'properties' in params_schema:
                # Get parameter names in order (use required first, then optional)
                required_params = params_schema.get('required', [])
                all_params = list(params_schema['properties'].keys())
                # Order: required params first, then remaining params
                positional_param_names = required_params + [p for p in all_params if p not in required_params]
    
    # Reconstruct function call with keyword arguments
    new_params = []
    positional_index = 0
    
    for key, value in parsed_params:
        if key is not None:
            # Already keyword argument
            new_params.append(f"{key}={value}")
        else:
            # Positional argument - convert to keyword
            if positional_param_names and positional_index < len(positional_param_names):
                param_name = positional_param_names[positional_index]
                new_params.append(f"{param_name}={value}")
                positional_index += 1
            else:
                # Fallback: keep as positional (shouldn't happen if tool_docs is complete)
                new_params.append(value)
                positional_index += 1
    
    return f"{func_name}({', '.join(new_params)})"


def calculate_format_reward(actions: List[str]) -> float:
    """
    Calculate format reward based on ToolRL paper design.
    
    Format reward checks if the output follows the correct format:
    - Correct list format: [action1, action2, ...]
    - Each action properly formatted: TOOL_NAME(arg1=value1, arg2=value2, ...)
    
    For SFT dataset generation with ground truth, format is always correct.
    Returns 1.0 (binary score).
    
    Args:
        actions: List of action strings for this turn
        
    Returns:
        Format reward score (1.0 for correct format)
    """
    # For ground truth data, format is always correct
    return 1.0


def calculate_tool_name_reward(actions: List[str]) -> float:
    """
    Calculate tool name matching reward.
    
    For ground truth data, all tool names are correct.
    Returns |G| where G is the set of tool calls.
    
    Args:
        actions: List of action strings for this turn
        
    Returns:
        Tool name reward = |G| (number of tool calls)
    """
    # For ground truth, all tool names match
    return float(len(actions))


def calculate_param_name_reward(actions: List[str]) -> float:
    """
    Calculate parameter name matching reward based on ToolRL paper formula:
    
    r_param = Σ |keys(P_G) ∩ keys(P_P)| / |keys(P_G) ∪ keys(P_P)| ∈ [0, |G|]
             G_j∈G
    
    where keys(P_G) and keys(P_P) are parameter names of predicted and ground-truth.
    This is Jaccard similarity summed over all tool calls.
    
    For ground truth data, intersection = union, so Jaccard = 1.0 for each call.
    
    Args:
        actions: List of action strings for this turn
        
    Returns:
        Parameter name reward ∈ [0, |G|]
    """
    total_score = 0.0
    
    for action in actions:
        # For ground truth, perfect match: Jaccard = 1.0
        total_score += 1.0
    
    return total_score


def calculate_param_content_reward(actions: List[str]) -> float:
    """
    Calculate parameter content matching reward based on ToolRL paper formula:
    
    r_value = Σ     Σ      𝟙[P_G[k] = P_P[k]] ∈ [0, Σ |keys(G_j)|]
             G_j∈G  k∈keys(G_j)                    G_j∈G
    
    where P_G[k] and P_P[k] are parameter values.
    This counts the number of matching parameter values across all tool calls.
    
    For ground truth data, all parameter values match.
    
    Args:
        actions: List of action strings for this turn
        
    Returns:
        Parameter content reward = total number of parameters
    """
    total_params = 0
    
    for action in actions:
        _, params = parse_tool_call(action)
        # Count parameters for this call
        total_params += len(params)
    
    # For ground truth, all parameters match
    return float(total_params)


def calculate_correctness_reward(actions: List[str]) -> float:
    """
    Calculate total correctness reward based on ToolRL paper design.
    
    R_correct = r_tool + r_param + r_value
    
    Where:
    - r_tool: Tool name matching (range [0, |G|])
    - r_param: Parameter name matching (range [0, |G|])
    - r_value: Parameter content matching (range [0, Σ|keys(G_j)|])
    
    Args:
        actions: List of action strings for this turn
        
    Returns:
        Total correctness reward
    """
    r_tool = calculate_tool_name_reward(actions)
    r_param = calculate_param_name_reward(actions)
    r_value = calculate_param_content_reward(actions)
    
    return r_tool + r_param + r_value


def calculate_turn_reward(actions: List[str]) -> float:
    """
    Calculate total reward for a single turn.
    
    Following ToolRL paper: R_final = R_correct + R_format
    
    Args:
        actions: List of action strings for this turn
        
    Returns:
        Total reward for the turn
    """
    r_format = calculate_format_reward(actions)
    r_correct = calculate_correctness_reward(actions)
    
    return r_format + r_correct


def calculate_return_to_go(
    current_turn_idx: int,
    ground_truth: List[List[str]],
    total_turns: int
) -> float:
    """
    Calculate the return-to-go (sum of future rewards) for the current turn.
    
    Based on ToolRL paper's reward design formulas:
    - R_final = R_correct + R_format
    - R_format = 1.0 (binary score for correct format)
    - R_correct = r_tool + r_param + r_value where:
      • r_tool = |G| (number of tool calls)
      • r_param = Σ (Jaccard similarity of param names) ∈ [0, |G|]
      • r_value = Σ (count of matching param values) ∈ [0, Σ|keys(G_j)|]
    
    Return-to-go = Sum of R_final for current turn and all future turns
    
    Args:
        current_turn_idx: Current turn index (0-indexed)
        ground_truth: Ground truth actions for all turns
        total_turns: Total number of turns in the episode
        
    Returns:
        Return-to-go value for the current turn
    """
    rtg = 0.0
    
    # Sum rewards from current turn to end
    for turn_idx in range(current_turn_idx, total_turns):
        turn_actions = ground_truth[turn_idx]
        if not turn_actions:
            turn_actions = []
        
        # Calculate reward for this turn using actual actions
        turn_reward = calculate_turn_reward(turn_actions)
        rtg += turn_reward
    
    return rtg


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
    
    # Pre-compute components that are the same for all steps
    env_summary = format_environment_summary_compact(initial_config)
    total_turns = len(ground_truth)
    
    # Create system prompt with available tools (same for all steps from this example)
    system_prompt_parts = []
    system_prompt_parts.append(
        "You are a helpful dialogue assistant capable of leveraging tool calls to solve user tasks and provide structured responses."
    )
    
    system_prompt_parts.append("\nAvailable Tools")
    system_prompt_parts.append("In your response, you can use the following tools:\n")
    # Use full tool specifications (not compact)
    tools_full_spec = format_available_tools_full(tool_paths, tool_docs)
    system_prompt_parts.append(tools_full_spec)
    
    system_prompt_parts.append("\n\nOutput Format")
    system_prompt_parts.append(
        "Your output should be a Python-style list of tool calls for the current turn, formatted as:\n"
        "[tool_call_1, tool_call_2, ...]\n"
        "where each tool_call is written as TOOL_NAME(arg1=value1, arg2=value2, ...). "
        "Do NOT include any extra explanations or text outside the list."
    )
    
    system_prompt_parts.append("\n\nReward Design")
    system_prompt_parts.append(
        "Your response will be evaluated based on the following reward formula:\n"
        "\n"
        "R_final = R_correct + R_format\n"
        "\n"
        "where:\n"
        "\n"
        "1. Format Reward (R_format):\n"
        "   - Binary score: 1.0 if output follows correct format, 0.0 otherwise\n"
        "   - Requirements:\n"
        "     * Output must be a valid Python list: [tool_call_1, tool_call_2, ...]\n"
        "     * Each tool call must follow: TOOL_NAME(arg1=value1, arg2=value2, ...)\n"
        "     * No extra text or explanations outside the list\n"
        "\n"
        "2. Correctness Reward (R_correct = r_tool + r_param + r_value):\n"
        "\n"
        "   a) Tool Name Matching (r_tool):\n"
        "      * Counts number of correct tool selections\n"
        "      * Range: [0, |G|] where |G| = number of tool calls\n"
        "      * Example: 3 correct tool calls gives r_tool = 3.0\n"
        "\n"
        "   b) Parameter Name Matching (r_param):\n"
        "      * Formula: r_param = sum over all calls of (intersection of param names / union of param names)\n"
        "      * This is Jaccard similarity of parameter names, summed over all tool calls\n"
        "      * Range: [0, |G|]\n"
        "      * Example: 3 calls with perfect param names gives r_param = 3.0\n"
        "\n"
        "   c) Parameter Content Matching (r_value):\n"
        "      * Formula: r_value = count of exact parameter value matches across all calls\n"
        "      * For each parameter k in each call, add 1 if predicted value equals ground truth value\n"
        "      * Range: [0, total_params] where total_params = sum of all parameters across calls\n"
        "      * Example: 3 calls with 2, 3, 1 params (all correct) gives r_value = 6.0\n"
        "\n"
        "Key insight: Rewards scale with task complexity. More tool calls and parameters\n"
        "lead to higher potential rewards. This design is from the ToolRL paper."
    )
    
    system_prompt = "\n".join(system_prompt_parts)
    
    # Generate one SFT record per step
    sft_records = []
    
    for step_idx in range(total_turns):
        # step_idx is 0-indexed, but turn numbers are 1-indexed
        current_turn = step_idx + 1
        
        # Build input using the compact template
        input_sections = []
        
        # 1. Current turn
        input_sections.append(f"Current turn:\nTurn {current_turn} of {total_turns}")
        
        # 2. Environment summary
        input_sections.append("Environment summary:\n- " + "\n- ".join(env_summary))
        
        # 3. Goal for this step
        input_sections.append("Goal for this step:\nDecide the next action or sequence of actions to take at the current turn.")

        # 4. Return-to-go
        return_to_go = calculate_return_to_go(step_idx, ground_truth, total_turns)
        input_sections.append(
            f"Expected return-to-go:\n"
            f"Taking the correct action(s) at this turn will result in a return-to-go of {return_to_go:.2f}."
        )
        
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


def create_cumulative_glaive_records(
    bfcl_obj: Dict[str, Any],
    tool_docs: Dict[str, Any],
    ground_truth: Optional[List[List[str]]],
    input_file_path: Path,
    use_real_observations: bool = True
) -> List[Dict[str, Any]]:
    """
    Convert a BFCL object into multiple cumulative Glaive training records.
    
    For an N-turn conversation, this creates N separate records where:
    - Record 1 contains: environment setup + turn 1
    - Record 2 contains: environment setup + turn 1 + turn 2
    - Record 3 contains: environment setup + turn 1 + turn 2 + turn 3
    - And so on...
    
    Each record predicts only the tool calls for the current turn, but includes
    the full conversation history up to that turn for context.
    
    Glaive format has:
    - conversations: Array of conversation turns with "from" (human/gpt/function_call/observation) and "value"
    - tools: JSON string of tool definitions
    
    Args:
        bfcl_obj: BFCL object dictionary
        tool_docs: Tool documentation dictionary
        ground_truth: Ground truth function calls
        input_file_path: Path to input file (for context)
        use_real_observations: If True, use environment to generate real observations
        
    Returns:
        List of Glaive record dictionaries (one per turn), or empty list if invalid
    """
    obj_id = bfcl_obj.get("id")
    questions = bfcl_obj.get("question", [])
    initial_config = bfcl_obj.get("initial_config", {})
    tool_paths = bfcl_obj.get("path", [])
    involved_classes = bfcl_obj.get("involved_classes", [])
    
    # Validate required fields
    if not questions:
        logger.warning(f"Skipping {obj_id}: No questions found")
        return []
    
    if not ground_truth:
        logger.warning(f"Skipping {obj_id}: No ground truth found")
        return []
    
    # Format tools as JSON string (same for all records)
    tools_json = format_tools_as_json_string(tool_paths, tool_docs)
    
    # Initialize environment if using real observations
    env = None
    if use_real_observations and initial_config:
        try:
            # Calculate total number of function calls across all turns
            total_function_calls = sum(len(turn_actions) for turn_actions in ground_truth)
            env = ToolCallingEnvironment()
            env.reset(initial_config, episode_id=obj_id, max_turns=total_function_calls)
        except Exception as e:
            logger.warning(f"Failed to initialize environment for {obj_id}: {e}")
            env = None
    
    # Build conversations array incrementally
    conversations_so_far = []
    cumulative_records = []
    
    # Add initial environment context as first human message (if there's initial config)
    if initial_config:
        env_description = format_initial_config(initial_config)
        if env_description and env_description != "No initial configuration provided.":
            # Add environment setup as context
            env_context = f"Environment setup:\n{env_description}"
            conversations_so_far.append({
                "from": "human",
                "value": env_context
            })
            # Add acknowledgment from assistant
            conversations_so_far.append({
                "from": "gpt",
                "value": "I understand the environment setup. I'm ready to help you with your tasks."
            })
    
    # Process each turn and create a cumulative record
    total_turns = len(ground_truth)
    for turn_idx in range(total_turns):
        # Get user request for this turn
        user_request = extract_user_request_for_turn(questions, turn_idx)
        
        # Add human message
        conversations_so_far.append({
            "from": "human",
            "value": user_request
        })
        
        # Get ground truth actions for this turn
        turn_actions = ground_truth[turn_idx]
        
        # Convert each function call to Glaive format and add to conversations
        if turn_actions:
            for func_call in turn_actions:
                # Convert function call to JSON format
                glaive_func_call = convert_function_call_to_glaive(func_call, tool_docs)
                
                # Add function_call message
                conversations_so_far.append({
                    "from": "function_call",
                    "value": json.dumps(glaive_func_call)
                })
                
                # Generate observation using environment
                observation_value = "{}"
                if env is not None:
                    # Convert positional args to keyword args for environment
                    func_call_for_env = convert_positional_to_keyword_args(func_call, tool_docs)
                    try:
                        obs, done, info = env.step(func_call_for_env)
                        observation_value = obs
                    except Exception as e:
                        logger.warning(f"Error executing {func_call} in {obj_id}: {e}")
                        observation_value = json.dumps({"error": str(e)})
                
                # Add observation
                conversations_so_far.append({
                    "from": "observation",
                    "value": observation_value
                })
            
            # Add assistant response after all function calls for this turn
            if turn_idx < total_turns - 1:
                # Not the last turn - simple acknowledgment
                conversations_so_far.append({
                    "from": "gpt",
                    "value": "I've completed the requested actions. What would you like me to do next?"
                })
            else:
                # Last turn - completion message
                conversations_so_far.append({
                    "from": "gpt",
                    "value": "I've completed all the requested actions."
                })
        else:
            # No actions for this turn
            conversations_so_far.append({
                "from": "gpt",
                "value": "I don't need to take any actions for this step."
            })
        
        # Create a cumulative record with conversation history up to this turn
        # Deep copy the conversations to preserve state
        cumulative_record = {
            "conversations": [conv.copy() for conv in conversations_so_far],
            "tools": tools_json
        }
        cumulative_records.append(cumulative_record)
    
    # Clean up environment
    if env is not None:
        env.close()
    
    return cumulative_records


def create_single_glaive_record(
    bfcl_obj: Dict[str, Any],
    tool_docs: Dict[str, Any],
    ground_truth: Optional[List[List[str]]],
    input_file_path: Path,
    use_real_observations: bool = True
) -> Optional[Dict[str, Any]]:
    """
    Convert a BFCL object into a single Glaive training record with all turns merged.
    
    This is the original behavior where one multi-turn conversation produces
    exactly one training record containing the complete conversation.
    
    Glaive format has:
    - conversations: Array of conversation turns with "from" (human/gpt/function_call/observation) and "value"
    - tools: JSON string of tool definitions
    
    Args:
        bfcl_obj: BFCL object dictionary
        tool_docs: Tool documentation dictionary
        ground_truth: Ground truth function calls
        input_file_path: Path to input file (for context)
        use_real_observations: If True, use environment to generate real observations
        
    Returns:
        Single Glaive record dictionary, or None if invalid
    """
    obj_id = bfcl_obj.get("id")
    questions = bfcl_obj.get("question", [])
    initial_config = bfcl_obj.get("initial_config", {})
    tool_paths = bfcl_obj.get("path", [])
    involved_classes = bfcl_obj.get("involved_classes", [])
    
    # Validate required fields
    if not questions:
        logger.warning(f"Skipping {obj_id}: No questions found")
        return None
    
    if not ground_truth:
        logger.warning(f"Skipping {obj_id}: No ground truth found")
        return None
    
    # Initialize environment if using real observations
    env = None
    if use_real_observations:
        try:
            # Calculate total number of function calls across all turns
            total_function_calls = sum(len(turn_actions) for turn_actions in ground_truth)
            env = ToolCallingEnvironment()
            env.reset(initial_config, episode_id=obj_id, max_turns=total_function_calls)
        except Exception as e:
            logger.error(f"Error initializing environment for {obj_id}: {e}")
            use_real_observations = False
    
    # Format tools as JSON string
    tools_json = format_tools_as_json_string(tool_paths, tool_docs)
    
    # Build complete conversations array (all turns)
    conversations = []
    
    # Add initial environment context as first human message (if there's initial config)
    if initial_config:
        env_description = format_initial_config(initial_config)
        if env_description and env_description != "No initial configuration provided.":
            # Add environment setup as context
            env_context = f"Environment setup:\n{env_description}"
            conversations.append({
                "from": "human",
                "value": env_context
            })
            # Add acknowledgment from assistant
            conversations.append({
                "from": "gpt",
                "value": "I understand the environment setup. I'm ready to help you with your tasks."
            })
    
    # Process ALL turns into one conversations array
    total_turns = len(ground_truth)
    for turn_idx in range(total_turns):
        # Get user request for this turn
        user_request = extract_user_request_for_turn(questions, turn_idx)
        
        # Add human message
        conversations.append({
            "from": "human",
            "value": user_request
        })
        
        # Get ground truth actions for this turn
        turn_actions = ground_truth[turn_idx]
        
        # Convert each function call to Glaive format and add to conversations
        if turn_actions:
            for func_call in turn_actions:
                # Convert function call to JSON format
                glaive_func_call = convert_function_call_to_glaive(func_call, tool_docs)
                
                # Add function_call message
                conversations.append({
                    "from": "function_call",
                    "value": json.dumps(glaive_func_call)
                })
                
                # Generate observation using environment
                observation_value = "{}"
                if env is not None:
                    # Convert positional args to keyword args for environment
                    func_call_for_env = convert_positional_to_keyword_args(func_call, tool_docs)
                    try:
                        obs, done, info = env.step(func_call_for_env)
                        observation_value = obs
                    except Exception as e:
                        logger.warning(f"Error executing {func_call} in {obj_id}: {e}")
                        observation_value = json.dumps({"error": str(e)})
                
                # Add observation
                conversations.append({
                    "from": "observation",
                    "value": observation_value
                })
            
            # Add assistant response after all function calls for this turn
            if turn_idx < total_turns - 1:
                # Not the last turn - simple acknowledgment
                conversations.append({
                    "from": "gpt",
                    "value": "I've completed the requested actions. What would you like me to do next?"
                })
            else:
                # Last turn - completion message
                conversations.append({
                    "from": "gpt",
                    "value": "I've completed all the requested actions."
                })
        else:
            # No actions for this turn
            conversations.append({
                "from": "gpt",
                "value": "I don't need to take any actions for this step."
            })
    
    # Clean up environment
    if env is not None:
        env.close()
    
    # Create Glaive record
    glaive_record = {
        "conversations": conversations,
        "tools": tools_json
    }
    
    return glaive_record


def process_bfcl_file(
    file_path: Path,
    tool_docs: Dict[str, Any],
    stats: Dict[str, int],
    use_real_observations: bool = True,
    mode: str = "cumulative"
) -> List[Dict[str, Any]]:
    """
    Process a single BFCL JSON file.
    
    Args:
        file_path: Path to BFCL JSON file
        tool_docs: Tool documentation dictionary
        stats: Statistics dictionary to update
        use_real_observations: If True, use environment to generate real observations
        mode: Output mode - "cumulative" or "single"
        
    Returns:
        List of Glaive records
    """
    glaive_records = []
    stats['files_processed'] += 1
    
    logger.info(f"Processing file: {file_path.name} (mode: {mode})")
    
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
                    
                    # Route based on mode
                    if mode == "cumulative":
                        # Create cumulative Glaive records (multiple per BFCL object, one per turn)
                        records_list = create_cumulative_glaive_records(
                            bfcl_obj, 
                            tool_docs, 
                            ground_truth, 
                            file_path,
                            use_real_observations=use_real_observations
                        )
                        
                        if records_list:
                            glaive_records.extend(records_list)
                            stats['records_created'] += len(records_list)
                        else:
                            stats['objects_skipped'] += 1
                    else:  # mode == "single"
                        # Create single Glaive record (one per BFCL object with all turns merged)
                        record = create_single_glaive_record(
                            bfcl_obj,
                            tool_docs,
                            ground_truth,
                            file_path,
                            use_real_observations=use_real_observations
                        )
                        
                        if record:
                            glaive_records.append(record)
                            stats['records_created'] += 1
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
    
    return glaive_records


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Convert BFCL v4 multi-turn JSON files to Glaive toolcall dataset format (cumulative records)"
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
        help="Output path for Glaive dataset JSON file"
    )
    parser.add_argument(
        "--use_real_observations",
        action="store_true",
        default=True,
        help="Use environment to generate real observations (default: True)"
    )
    parser.add_argument(
        "--no_real_observations",
        action="store_false",
        dest="use_real_observations",
        help="Disable real observation generation (use empty placeholders)"
    )
    parser.add_argument(
        "--mode",
        type=str,
        choices=["cumulative", "single"],
        default="single",
        help="Output mode: 'cumulative' generates N records per N-turn conversation, "
             "'single' generates 1 record per conversation with all turns merged"
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
    all_glaive_records = []
    logger.info(f"Mode: {args.mode}")
    logger.info(f"Real observations: {'ENABLED' if args.use_real_observations else 'DISABLED'}")
    for bfcl_file in sorted(bfcl_files):
        records = process_bfcl_file(bfcl_file, tool_docs, stats, use_real_observations=args.use_real_observations, mode=args.mode)
        all_glaive_records.extend(records)
    
    # Write output
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(all_glaive_records, f, ensure_ascii=False, indent=2)
    
    # Log final statistics
    logger.info("=" * 60)
    logger.info("Conversion complete!")
    logger.info(f"Mode: {args.mode}")
    logger.info(f"Files processed: {stats['files_processed']}")
    
    if args.mode == "cumulative":
        logger.info(f"Glaive records created: {stats['records_created']} "
                    f"(cumulative: N records per N-turn episode)")
        logger.info(f"Note: Each multi-turn example is separated into cumulative records (one per turn)")
    else:  # single mode
        logger.info(f"Glaive records created: {stats['records_created']} "
                    f"(single: 1 record per multi-turn episode)")
        logger.info(f"Note: Each multi-turn example is converted to one Glaive record with all turns merged")
    
    logger.info(f"Objects skipped: {stats['objects_skipped']}")
    logger.info(f"Output written to: {output_path}")
    logger.info("=" * 60)
    
    return 0


if __name__ == "__main__":
    exit(main())

