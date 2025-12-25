#!/usr/bin/env python3
"""
Build RL Trajectories Dataset

Generates diverse function calling trajectories using multiple model tiers (expert, intermediate, weak)
on BFCL benchmark data with strict in-distribution prompting.

For each BFCL sample, generates 3 separate trajectory records (one per model tier) by:
1. Loading exact prompts from BFCL data (no OOD formatting)
2. Generating tool calls using different model tiers
3. Executing tool calls in ToolCallingEnvironment
4. Capturing observations and building Glaive-format conversations
5. Outputting records matching bfcl_glaive_dataset.json schema
"""

import argparse
import json
import logging
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

# Import the environment for executing tool calls
import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from bfcl_eval.env import ToolCallingEnvironment

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Set logging level from environment variable if present
import os
if os.environ.get('DEBUG'):
    logger.setLevel(logging.DEBUG)


# Class to file mapping for tool documentation (from build_bfcl_sft.py)
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


class ModelGenerator:
    """
    Handles model loading and inference for HuggingFace models.
    
    Generates tool calls from exact prompts without any formatting modifications.
    """
    
    def __init__(
        self,
        model_path: str,
        device: str = "cuda",
        quantization: Optional[str] = None,
        max_new_tokens: int = 512,
        use_auto_device_map: bool = True
    ):
        """
        Initialize model generator.
        
        Args:
            model_path: HuggingFace model path or local path
            device: Device to use (cuda/cpu)
            quantization: Quantization mode (4bit/8bit/none)
            max_new_tokens: Maximum tokens to generate
            use_auto_device_map: Use automatic device mapping for multi-GPU (default: True)
        """
        self.model_path = model_path
        self.device = device
        self.quantization = quantization
        self.max_new_tokens = max_new_tokens
        self.use_auto_device_map = use_auto_device_map
        
        self.model = None
        self.tokenizer = None
        self._load_model()
    
    def _load_model(self):
        """Load model and tokenizer."""
        try:
            logger.info(f"Loading model from {self.model_path}...")
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.model_path,
                trust_remote_code=True
            )
            
            # Set pad token if not present
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            
            # Load model with optional quantization
            if self.quantization == "4bit":
                from transformers import BitsAndBytesConfig
                quantization_config = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_compute_dtype=torch.float16
                )
                self.model = AutoModelForCausalLM.from_pretrained(
                    self.model_path,
                    quantization_config=quantization_config,
                    trust_remote_code=True,
                    device_map="auto"
                )
                logger.info(f"Model loaded with 4-bit quantization using device_map='auto'")
            elif self.quantization == "8bit":
                self.model = AutoModelForCausalLM.from_pretrained(
                    self.model_path,
                    load_in_8bit=True,
                    trust_remote_code=True,
                    device_map="auto"
                )
                logger.info(f"Model loaded with 8-bit quantization using device_map='auto'")
            else:
                # Non-quantized model
                if self.use_auto_device_map and self.device == "cuda":
                    # Use automatic device mapping for multi-GPU parallelism
                    self.model = AutoModelForCausalLM.from_pretrained(
                        self.model_path,
                        trust_remote_code=True,
                        dtype=torch.float16,
                        device_map="auto"  # Automatically distribute across available GPUs
                    )
                    logger.info(f"Model loaded with device_map='auto' (multi-GPU)")
                    
                    # Log device placement
                    if hasattr(self.model, 'hf_device_map'):
                        logger.info(f"Device map: {self.model.hf_device_map}")
                else:
                    # Load to specific device
                    self.model = AutoModelForCausalLM.from_pretrained(
                        self.model_path,
                        trust_remote_code=True,
                        dtype=torch.float16 if self.device == "cuda" else torch.float32
                    )
                    if self.device == "cuda" or self.device.startswith("cuda:"):
                        self.model = self.model.to(self.device)
                    logger.info(f"Model loaded on device: {self.device}")
            
        except Exception as e:
            logger.error(f"Failed to load model from {self.model_path}: {e}")
            raise
    
    def generate(self, messages: List[Dict[str, str]], tools: Optional[str] = None) -> str:
        """
        Generate tool call using tokenizer's chat template (matches SFT training).
        
        Args:
            messages: List of message dicts with "role" and "content" keys
                     (Glaive format: from -> role mapping)
            tools: Optional JSON string of tool definitions
            
        Returns:
            Generated text (tool call portion only)
        """
        try:
            # Use tokenizer's chat template - this is how the model was trained!
            # The template knows how to format conversations and tools
            if hasattr(self.tokenizer, 'apply_chat_template') and self.tokenizer.chat_template:
                # Convert messages to chat template format
                # Some templates support tools parameter
                try:
                    # Try with tools if provided
                    if tools:
                        tools_list = json.loads(tools)
                        prompt = self.tokenizer.apply_chat_template(
                            messages,
                            tools=tools_list,
                            add_generation_prompt=True,
                            tokenize=False
                        )
                    else:
                        prompt = self.tokenizer.apply_chat_template(
                            messages,
                            add_generation_prompt=True,
                            tokenize=False
                        )
                except TypeError:
                    # Template doesn't support tools parameter, fallback
                    prompt = self.tokenizer.apply_chat_template(
                        messages,
                        add_generation_prompt=True,
                        tokenize=False
                    )
            else:
                # Fallback: simple concatenation if no template
                logger.warning("No chat template found, using simple concatenation")
                prompt = "\n\n".join([m.get("content", "") for m in messages])
            
            # Tokenize
            inputs = self.tokenizer(prompt, return_tensors="pt", truncation=True, max_length=4096)
            
            # Move to device
            if hasattr(self.model, 'device'):
                inputs = {k: v.to(self.model.device) for k, v in inputs.items()}
            elif self.device == "cuda" or self.device.startswith("cuda:"):
                inputs = {k: v.to(self.device) for k, v in inputs.items()}
            else:
                first_device = next(self.model.parameters()).device
                inputs = {k: v.to(first_device) for k, v in inputs.items()}
            
            # Generate
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=self.max_new_tokens,
                    do_sample=True,
                    temperature=0.7,
                    pad_token_id=self.tokenizer.pad_token_id,
                    eos_token_id=self.tokenizer.eos_token_id
                )
            
            # Decode only the new tokens
            generated_text = self.tokenizer.decode(
                outputs[0][inputs['input_ids'].shape[1]:],
                skip_special_tokens=True
            )
            
            return generated_text.strip()
        except Exception as e:
            logger.error(f"Generation error: {e}")
            import traceback
            logger.error(traceback.format_exc())
            return ""
    
    def parse_tool_calls(self, output: str) -> List[str]:
        """
        Parse model output to extract tool calls.
        
        Handles multiple formats:
        - OpenAI-style: <tool_call>{"name": "cd", "arguments": {...}}</tool_call>
        - Python-style: cd(folder='temp')
        - JSON blocks: ```json\n{"name": "cd", "arguments": {...}}\n```
        - List format: [cd(folder='temp'), mv(...)]
        
        Args:
            output: Raw model output
            
        Returns:
            List of tool call strings in Python format
        """
        tool_calls = []
        
        # Try to extract OpenAI-style <tool_call> tags first
        tool_call_pattern = r'<tool_call>\s*(\{.*?\})\s*</tool_call>'
        tool_call_matches = re.findall(tool_call_pattern, output, re.DOTALL)
        if tool_call_matches:
            for match in tool_call_matches:
                try:
                    parsed = json.loads(match)
                    if isinstance(parsed, dict) and "name" in parsed:
                        tool_call = self._json_to_python_call(parsed)
                        if tool_call:
                            tool_calls.append(tool_call)
                except json.JSONDecodeError:
                    # Try to clean up the JSON (remove trailing ellipsis, etc.)
                    cleaned_match = match.strip()
                    if cleaned_match.endswith('...'):
                        cleaned_match = cleaned_match[:-3].strip()
                        # Try to close any open braces
                        open_braces = cleaned_match.count('{') - cleaned_match.count('}')
                        if open_braces > 0:
                            cleaned_match += '}' * open_braces
                        try:
                            parsed = json.loads(cleaned_match)
                            if isinstance(parsed, dict) and "name" in parsed:
                                tool_call = self._json_to_python_call(parsed)
                                if tool_call:
                                    tool_calls.append(tool_call)
                        except json.JSONDecodeError:
                            pass
        
        if tool_calls:
            return tool_calls
        
        # Try to extract JSON code blocks first (```json ... ```)
        json_pattern = r'```json\s*(\[.*?\]|\{.*?\})\s*```'
        json_matches = re.findall(json_pattern, output, re.DOTALL)
        if json_matches:
            for match in json_matches:
                try:
                    parsed = json.loads(match)
                    if isinstance(parsed, list):
                        for item in parsed:
                            if isinstance(item, dict) and "name" in item:
                                tool_call = self._json_to_python_call(item)
                                if tool_call:
                                    tool_calls.append(tool_call)
                    elif isinstance(parsed, dict) and "name" in parsed:
                        tool_call = self._json_to_python_call(parsed)
                        if tool_call:
                            tool_calls.append(tool_call)
                except json.JSONDecodeError:
                    pass
        
        if tool_calls:
            return tool_calls
        
        # Try to extract JSON without code blocks (handle nested JSON properly)
        # Find positions where JSON objects might start
        json_starts = [i for i, char in enumerate(output) if char == '{']
        for start_pos in json_starts:
            # Try to find a valid JSON object starting from this position
            depth = 0
            for end_pos in range(start_pos, len(output)):
                if output[end_pos] == '{':
                    depth += 1
                elif output[end_pos] == '}':
                    depth -= 1
                    if depth == 0:
                        # Found a complete JSON object
                        candidate = output[start_pos:end_pos+1]
                        # Check if it looks like a tool call (has "name" and "arguments")
                        if '"name"' in candidate and '"arguments"' in candidate:
                            try:
                                parsed = json.loads(candidate)
                                if isinstance(parsed, dict) and "name" in parsed:
                                    tool_call = self._json_to_python_call(parsed)
                                    if tool_call and tool_call not in tool_calls:
                                        tool_calls.append(tool_call)
                            except json.JSONDecodeError:
                                pass
                        break
        
        if tool_calls:
            return tool_calls
        
        # Try to extract Python-style list: [call1, call2, ...]
        list_pattern = r'\[([^\[\]]*\([^\)]*\)[^\[\]]*)\]'
        list_matches = re.findall(list_pattern, output)
        if list_matches:
            for list_content in list_matches:
                calls = self._split_function_calls(list_content)
                if calls:
                    tool_calls.extend(calls)
        
        if tool_calls:
            return tool_calls
        
        # Try to extract single function calls: func_name(arg1=val1, ...)
        # Be more specific: require at least one character before parenthesis
        single_call_pattern = r'\b([a-zA-Z_][a-zA-Z0-9_]*)\s*\(([^)]*)\)'
        single_matches = re.findall(single_call_pattern, output)
        if single_matches:
            for func_name, args in single_matches:
                # Skip common words that aren't function calls
                if func_name.lower() in ['if', 'while', 'for', 'print', 'return', 'import']:
                    continue
                # Reconstruct the call
                tool_calls.append(f"{func_name}({args})")
        
        return tool_calls
    
    def _json_to_python_call(self, json_obj: Dict[str, Any]) -> Optional[str]:
        """Convert JSON tool call to Python-style string."""
        if "name" not in json_obj:
            return None
        
        func_name = json_obj["name"]
        args = json_obj.get("arguments", {})
        
        if not args:
            return f"{func_name}()"
        
        # Format arguments
        arg_strs = []
        for key, value in args.items():
            if isinstance(value, str):
                arg_strs.append(f"{key}='{value}'")
            else:
                arg_strs.append(f"{key}={value}")
        
        return f"{func_name}({', '.join(arg_strs)})"
    
    def _split_function_calls(self, list_content: str) -> List[str]:
        """Split comma-separated function calls respecting nested parentheses."""
        calls = []
        current = ""
        depth = 0
        
        for char in list_content:
            if char == '(':
                depth += 1
                current += char
            elif char == ')':
                depth -= 1
                current += char
            elif char == ',' and depth == 0:
                if current.strip():
                    calls.append(current.strip())
                current = ""
            else:
                current += char
        
        if current.strip():
            calls.append(current.strip())
        
        return calls


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


def convert_function_call_to_glaive(func_call: str, tool_docs: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """
    Convert a Python-style function call to Glaive JSON format.
    
    Args:
        func_call: Function call string, e.g., "cd(folder='temp')"
        tool_docs: Optional tool documentation to resolve positional argument names
        
    Returns:
        Dictionary with "name" and "arguments" keys in Glaive format
    """
    if '(' not in func_call:
        return {"name": func_call.strip(), "arguments": {}}
    
    # Extract function name
    func_name = func_call[:func_call.index('(')].strip()
    params_str = func_call[func_call.index('(') + 1:func_call.rindex(')')].strip()
    
    # Parse parameters into a dictionary
    arguments = {}
    if params_str:
        # Parse all parameters (both keyword and positional)
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
        
        # Resolve positional arguments to parameter names
        positional_param_names = None
        if any(key is None for key, _ in parsed_params):
            if tool_docs:
                tool_doc = tool_docs.get(func_name)
                if tool_doc and 'parameters' in tool_doc:
                    params_schema = tool_doc['parameters']
                    if 'properties' in params_schema:
                        required_params = params_schema.get('required', [])
                        all_params = list(params_schema['properties'].keys())
                        positional_param_names = required_params + [p for p in all_params if p not in required_params]
        
        # Build final arguments dictionary
        positional_index = 0
        for key, value in parsed_params:
            if key is not None:
                arguments[key] = parse_parameter_value(value)
            else:
                if positional_param_names and positional_index < len(positional_param_names):
                    param_name = positional_param_names[positional_index]
                    arguments[param_name] = parse_parameter_value(value)
                    positional_index += 1
                else:
                    arguments[f"arg{positional_index}"] = parse_parameter_value(value)
                    positional_index += 1
    
    return {"name": func_name, "arguments": arguments}


def parse_parameter_value(value: str) -> Any:
    """
    Parse a parameter value string and convert to appropriate Python type.
    
    Args:
        value: String representation of parameter value
        
    Returns:
        Parsed value with appropriate type
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


class TrajectoryBuilder:
    """
    Orchestrates trajectory generation with environment interaction.
    
    Builds Glaive-format conversation records by:
    1. Extracting exact prompts from BFCL data
    2. Generating tool calls using ModelGenerator
    3. Executing tool calls in ToolCallingEnvironment
    4. Capturing observations and building conversation array
    """
    
    def __init__(
        self,
        model_generator: ModelGenerator,
        tool_docs: Dict[str, Any],
        func_doc_dir: Optional[Path] = None
    ):
        """
        Initialize trajectory builder.
        
        Args:
            model_generator: ModelGenerator instance for generating tool calls
            tool_docs: Tool documentation dictionary
            func_doc_dir: Optional path to func_doc directory for environment
        """
        self.model_generator = model_generator
        self.tool_docs = tool_docs
        self.func_doc_dir = func_doc_dir
        self.env = None
    
    def build_trajectory(
        self,
        bfcl_obj: Dict[str, Any],
        model_tier: str
    ) -> Optional[Dict[str, Any]]:
        """
        Build a complete trajectory for a BFCL object.
        
        Args:
            bfcl_obj: BFCL object dictionary
            model_tier: Model tier identifier (expert/intermediate/weak)
            
        Returns:
            Glaive-format record dictionary or None if failed
        """
        obj_id = bfcl_obj.get("id")
        questions = bfcl_obj.get("question", [])
        initial_config = bfcl_obj.get("initial_config", {})
        tool_paths = bfcl_obj.get("path", [])
        
        if not questions:
            logger.warning(f"Skipping {obj_id}: No questions found")
            return None
        
        # Format tools as JSON string
        tools_json = format_tools_as_json_string(tool_paths, self.tool_docs)
        
        # Initialize environment
        try:
            self.env = ToolCallingEnvironment(func_doc_dir=self.func_doc_dir)
            total_turns = len(questions)
            self.env.reset(initial_config, episode_id=f"{obj_id}_{model_tier}", max_turns=total_turns * 10)
        except Exception as e:
            logger.error(f"Failed to initialize environment for {obj_id}: {e}")
            return None
        
        # Build conversations array
        conversations = []
        
        # Add initial environment context
        if initial_config:
            env_description = format_initial_config(initial_config)
            if env_description and env_description != "No initial configuration provided.":
                env_context = f"Environment setup:\n{env_description}"
                conversations.append({
                    "from": "human",
                    "value": env_context
                })
                conversations.append({
                    "from": "gpt",
                    "value": "I understand the environment setup. I'm ready to help you with your tasks."
                })
        
        # Process each turn
        try:
            for turn_idx in range(total_turns):
                # Get user request for this turn
                user_request = extract_user_request_for_turn(questions, turn_idx)
                
                # Add human message
                conversations.append({
                    "from": "human",
                    "value": user_request
                })
                
                # Convert Glaive conversations to messages format for chat template
                messages = self._convert_conversations_to_messages(conversations)
                
                # Get tools JSON string
                tools_json = format_tools_as_json_string(tool_paths, self.tool_docs)
                
                # Generate tool calls using chat template
                generated_text = self.model_generator.generate(messages, tools=tools_json)
                tool_calls = self.model_generator.parse_tool_calls(generated_text)
                
                if not tool_calls:
                    logger.warning(f"No tool calls generated for {obj_id} turn {turn_idx + 1}")
                    logger.debug(f"Generated text: {generated_text[:512]}...")  # Log first 200 chars
                    # Add empty response
                    conversations.append({
                        "from": "gpt",
                        "value": "I don't need to take any actions for this step."
                    })
                    continue
                
                # Execute each tool call
                for func_call in tool_calls:
                    # Convert to Glaive format
                    glaive_func_call = convert_function_call_to_glaive(func_call, self.tool_docs)
                    
                    # Add function_call message
                    conversations.append({
                        "from": "function_call",
                        "value": json.dumps(glaive_func_call)
                    })
                    
                    # Execute in environment
                    try:
                        obs, done, info = self.env.step(func_call)
                        conversations.append({
                            "from": "observation",
                            "value": obs
                        })
                    except Exception as e:
                        logger.warning(f"Error executing {func_call} in {obj_id}: {e}")
                        conversations.append({
                            "from": "observation",
                            "value": json.dumps({"error": str(e)})
                        })
                
                # Add assistant response after all function calls for this turn
                if turn_idx < total_turns - 1:
                    conversations.append({
                        "from": "gpt",
                        "value": "I've completed the requested actions. What would you like me to do next?"
                    })
                else:
                    conversations.append({
                        "from": "gpt",
                        "value": "I've completed all the requested actions."
                    })
        
        except Exception as e:
            logger.error(f"Error building trajectory for {obj_id}: {e}")
            return None
        finally:
            if self.env:
                self.env.close()
        
        # Create Glaive record
        glaive_record = {
            "conversations": conversations,
            "tools": tools_json
        }
        
        return glaive_record
    
    def _convert_conversations_to_messages(self, conversations: List[Dict[str, str]]) -> List[Dict[str, str]]:
        """
        Convert Glaive conversations format to messages format for chat template.
        
        Glaive format uses "from" field with values: human, gpt, function_call, observation
        Chat template expects "role" and "content" fields.
        
        Args:
            conversations: Glaive format conversations
            
        Returns:
            Messages format for chat template
        """
        messages = []
        
        for conv in conversations:
            role_from = conv.get("from")
            value = conv.get("value")
            
            # Map Glaive roles to standard chat roles
            if role_from == "human":
                messages.append({"role": "user", "content": value})
            elif role_from == "gpt":
                messages.append({"role": "assistant", "content": value})
            elif role_from == "function_call":
                # Function calls - some templates have special handling
                messages.append({"role": "assistant", "content": value})
            elif role_from == "observation":
                # Observations - treated as tool/function responses
                # Some templates support "tool" role, fallback to "user" or "system"
                messages.append({"role": "user", "content": value})
        
        return messages
    
    def _build_prompt(self, conversations: List[Dict[str, str]]) -> str:
        """
        Build prompt for model from conversation history.
        
        CRITICAL: Use exact in-distribution format - no templates, no system prompts, no labels.
        Include full conversation history including observations for context.
        Use exact text from BFCL data with minimal formatting.
        
        Args:
            conversations: Current conversation history (up to current user request)
            
        Returns:
            Plain text prompt with full conversation context (exact BFCL text)
        """
        # Build prompt from full conversation history
        # Use exact text from conversations - no formatting, no templates, no labels
        prompt_parts = []
        
        # Include all conversation turns (exact text)
        # For in-distribution: use exact user messages from BFCL, include observations for context
        for conv in conversations:
            role = conv["from"]
            value = conv["value"]
            
            if role == "human":
                # Use exact user message text (from BFCL question field)
                prompt_parts.append(value)
            elif role == "observation":
                # Include observations for multi-turn context
                # Use the raw observation value (JSON string) - model can parse if needed
                # Or extract key info for readability
                try:
                    obs_data = json.loads(value)
                    # Prefer context field if available (more readable)
                    context = obs_data.get("context", "")
                    result = obs_data.get("result", "")
                    if context:
                        prompt_parts.append(f"Result: {context}")
                    elif result:
                        prompt_parts.append(f"Result: {result}")
                    else:
                        # Fallback to full JSON
                        prompt_parts.append(f"Result: {value}")
                except:
                    # Not JSON, use as-is
                    prompt_parts.append(f"Result: {value}")
            # Skip function_call and gpt responses in prompt
            # Model should generate based on user requests and observations only
        
        # Add instruction about output format at the end
        prompt_parts.append("\nGenerate the tool calls needed as a Python list: [tool_name(arg1=value1, arg2=value2), ...]")
        
        # Join with double newlines to separate turns (minimal formatting)
        # This maintains in-distribution by using exact BFCL text
        return "\n\n".join(prompt_parts)


def load_bfcl_files(input_dir: Path) -> List[Tuple[Path, Dict[str, Any]]]:
    """
    Load all BFCL JSON files from input directory.
    
    Args:
        input_dir: Directory containing BFCL_v4_multi_turn_*.json files
        
    Returns:
        List of (file_path, bfcl_object) tuples
    """
    bfcl_files = list(input_dir.rglob("BFCL_v4_multi_turn_*.json"))
    
    # Exclude possible_answer directory and unused_datasets directory
    bfcl_files = [f for f in bfcl_files if "possible_answer" not in str(f) and "unused_datasets" not in str(f)]
    
    all_objects = []
    
    for file_path in sorted(bfcl_files):
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                for line_num, line in enumerate(f, 1):
                    line = line.strip()
                    if not line:
                        continue
                    
                    try:
                        bfcl_obj = json.loads(line)
                        all_objects.append((file_path, bfcl_obj))
                    except json.JSONDecodeError as e:
                        logger.error(f"JSON decode error in {file_path.name} line {line_num}: {e}")
        except Exception as e:
            logger.error(f"Error reading file {file_path}: {e}")
    
    return all_objects


def filter_for_test_mode(
    bfcl_objects: List[Tuple[Path, Dict[str, Any]]],
    target_api: Optional[str],
    samples_per_api_file: int
) -> List[Tuple[Path, Dict[str, Any]]]:
    """
    Filter BFCL objects for test mode.
    
    If target_api is specified: selects samples for that specific API only.
    If target_api is None: selects samples for ALL APIs found in the dataset.
    
    Final size = num_apis × samples_per_api_file × 4 files
    
    Args:
        bfcl_objects: List of (file_path, bfcl_object) tuples
        target_api: Optional API class to filter for (e.g., "GorillaFileSystem"). 
                   If None, includes all APIs.
        samples_per_api_file: Number of samples to select per API per file
        
    Returns:
        Filtered list of (file_path, bfcl_object) tuples
    """
    # The 4 main BFCL files
    main_files = [
        "BFCL_v4_multi_turn_base.json",
        "BFCL_v4_multi_turn_miss_param.json",
        "BFCL_v4_multi_turn_miss_func.json",
        "BFCL_v4_multi_turn_long_context.json"
    ]
    
    # Group objects by file
    objects_by_file = {}
    for file_path, obj in bfcl_objects:
        file_name = file_path.name
        if file_name not in objects_by_file:
            objects_by_file[file_name] = []
        objects_by_file[file_name].append((file_path, obj))
    
    selected = []
    
    if target_api:
        # Single API mode: select samples for specified API only
        for main_file in main_files:
            if main_file not in objects_by_file:
                logger.warning(f"Test mode: file {main_file} not found")
                continue
            
            # Filter objects that use the target API
            api_objects = []
            for file_path, obj in objects_by_file[main_file]:
                involved_classes = obj.get("involved_classes", [])
                if target_api in involved_classes:
                    api_objects.append((file_path, obj))
            
            # Take first N samples
            if len(api_objects) >= samples_per_api_file:
                selected.extend(api_objects[:samples_per_api_file])
                logger.info(f"Test mode: selected {samples_per_api_file} samples from {main_file} with API {target_api}")
            elif len(api_objects) > 0:
                selected.extend(api_objects)
                logger.warning(f"Test mode: only {len(api_objects)} sample(s) found in {main_file} with API {target_api}")
            else:
                logger.warning(f"Test mode: no samples found in {main_file} with API {target_api}")
    else:
        # All APIs mode: collect samples for each unique API
        # First, discover all unique APIs across all files
        all_apis = set()
        for main_file in main_files:
            if main_file not in objects_by_file:
                continue
            for _, obj in objects_by_file[main_file]:
                involved_classes = obj.get("involved_classes", [])
                all_apis.update(involved_classes)
        
        all_apis = sorted(list(all_apis))
        logger.info(f"Test mode: found {len(all_apis)} unique APIs: {', '.join(all_apis)}")
        
        # Process file by file (not API by API) for better organization
        for main_file in main_files:
            if main_file not in objects_by_file:
                continue
            
            logger.info(f"Test mode: processing {main_file}")
            
            # For each API, select samples from this file
            for api in all_apis:
                api_objects = []
                for file_path, obj in objects_by_file[main_file]:
                    involved_classes = obj.get("involved_classes", [])
                    if api in involved_classes:
                        api_objects.append((file_path, obj))
                
                # Take first N samples for this API from this file
                if len(api_objects) >= samples_per_api_file:
                    selected.extend(api_objects[:samples_per_api_file])
                elif len(api_objects) > 0:
                    selected.extend(api_objects)
        
        logger.info(f"Test mode: selected {samples_per_api_file} samples per API per file")
        logger.info(f"Test mode: total = {len(all_apis)} APIs × {samples_per_api_file} samples × 4 files = {len(selected)} samples (approx)")
    
    return selected


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Generate RL trajectories using multiple model tiers on BFCL data"
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
        help="Output path for trajectory dataset JSON file"
    )
    parser.add_argument(
        "--expert_model_path",
        type=str,
        required=True,
        help="HuggingFace model path for expert tier (70B+)"
    )
    parser.add_argument(
        "--intermediate_model_path",
        type=str,
        required=True,
        help="HuggingFace model path for intermediate tier (7B-8B)"
    )
    parser.add_argument(
        "--weak_model_path",
        type=str,
        required=True,
        help="HuggingFace model path for weak tier (1B)"
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda:0",
        help="Device to use for inference (cuda/cpu/cuda:0/cuda:1/etc). Use 'cuda' for multi-GPU with device_map='auto'"
    )
    parser.add_argument(
        "--no_auto_device_map",
        action="store_true",
        help="Disable automatic device mapping (loads all models to single device specified by --device)"
    )
    parser.add_argument(
        "--quantization",
        type=str,
        default="none",
        choices=["none", "4bit", "8bit"],
        help="Quantization mode for model loading"
    )
    parser.add_argument(
        "--max_samples",
        type=int,
        default=None,
        help="Limit number of samples to process (default: all)"
    )
    parser.add_argument(
        "--max_new_tokens",
        type=int,
        default=512,
        help="Maximum tokens to generate per model call"
    )
    parser.add_argument(
        "--test_mode",
        action="store_true",
        help="Generate test dataset with limited samples per API per file"
    )
    parser.add_argument(
        "--test_api",
        type=str,
        default=None,
        help="Specific API class to use for test mode (default: None = all APIs). Examples: GorillaFileSystem, TwitterAPI"
    )
    parser.add_argument(
        "--samples_per_api_file",
        type=int,
        default=2,
        help="[Test mode only] Number of samples per API per file (default: 2). Final size = num_apis × samples_per_api_file × 4 files"
    )
    parser.add_argument(
        "--trajectories_per_sample",
        type=int,
        default=1,
        help="[Full mode only] Number of trajectories to generate per sample (default: 1). Use >1 to generate diverse trajectories for the same input."
    )
    
    args = parser.parse_args()
    
    # Validate argument combinations
    if not args.test_mode:
        # Full mode: --samples_per_api_file and --test_api should not be used
        if args.samples_per_api_file != 2:  # 2 is the default
            logger.warning("--samples_per_api_file is only used in test mode (--test_mode). Ignoring.")
        if args.test_api is not None:
            logger.warning("--test_api is only used in test mode (--test_mode). Ignoring.")
    else:
        # Test mode: --trajectories_per_sample should not be used
        if args.trajectories_per_sample != 1:  # 1 is the default
            logger.warning("--trajectories_per_sample is only used in full mode (without --test_mode). Ignoring.")
            args.trajectories_per_sample = 1  # Reset to default for test mode
    
    # Determine output mode: test_mode uses verbose logging, full mode uses progress bar
    use_progress_bar = not args.test_mode
    
    input_dir = Path(args.input_dir)
    output_path = Path(args.output_path)
    
    # Validate input directory
    if not input_dir.exists():
        logger.error(f"Input directory does not exist: {input_dir}")
        return 1
    
    # Find tool documentation directory
    func_doc_dir = input_dir / "multi_turn_func_doc"
    if not func_doc_dir.exists():
        # Try relative to script location
        script_dir = Path(__file__).parent.parent
        func_doc_dir = script_dir / "berkeley-function-call-leaderboard/bfcl_eval/data/multi_turn_func_doc"
    
    if not func_doc_dir.exists():
        logger.error(f"Tool documentation directory not found: {func_doc_dir}")
        return 1
    
    # Load BFCL files
    logger.info("Loading BFCL files...")
    bfcl_objects = load_bfcl_files(input_dir)
    
    # Apply test mode filtering if enabled
    if args.test_mode:
        if args.test_api:
            logger.info(f"Test mode enabled: selecting {args.samples_per_api_file} samples per file for API '{args.test_api}'")
        else:
            logger.info(f"Test mode enabled: selecting {args.samples_per_api_file} samples per API per file for ALL APIs")
        bfcl_objects = filter_for_test_mode(bfcl_objects, args.test_api, args.samples_per_api_file)
        logger.info(f"Test mode: selected {len(bfcl_objects)} samples")
    elif args.max_samples:
        bfcl_objects = bfcl_objects[:args.max_samples]
    
    logger.info(f"Found {len(bfcl_objects)} BFCL objects to process")
    
    # Prepare model configurations (lazy loading)
    logger.info("Preparing model configurations...")
    logger.info(f"Auto device mapping: {'disabled' if args.no_auto_device_map else 'enabled'}")
    
    use_auto_map = not args.no_auto_device_map
    
    # Model configurations for lazy loading
    model_configs = {
        "expert": {
            "path": args.expert_model_path,
            "device": args.device,
            "quantization": args.quantization if args.quantization != "none" else None,
            "max_new_tokens": args.max_new_tokens,
            "use_auto_device_map": use_auto_map
        },
        "intermediate": {
            "path": args.intermediate_model_path,
            "device": args.device,
            "quantization": args.quantization if args.quantization != "none" else None,
            "max_new_tokens": args.max_new_tokens,
            "use_auto_device_map": use_auto_map
        },
        "weak": {
            "path": args.weak_model_path,
            "device": args.device,
            "quantization": args.quantization if args.quantization != "none" else None,
            "max_new_tokens": args.max_new_tokens,
            "use_auto_device_map": use_auto_map
        }
    }
    
    logger.info("Models will be loaded on-demand to optimize GPU memory usage")
    
    # Statistics
    total_iterations_per_tier = len(bfcl_objects) * args.trajectories_per_sample
    stats = {
        'total_samples': len(bfcl_objects),
        'trajectories_per_sample': args.trajectories_per_sample,
        'total_iterations_per_tier': total_iterations_per_tier,
        'expert_success': 0,
        'expert_failed': 0,
        'intermediate_success': 0,
        'intermediate_failed': 0,
        'weak_success': 0,
        'weak_failed': 0,
    }
    
    if args.trajectories_per_sample > 1:
        logger.info(f"Generating {args.trajectories_per_sample} trajectories per sample")
        logger.info(f"Total iterations per model tier: {total_iterations_per_tier}")
    
    # Process all objects by model tier (load one model at a time)
    all_trajectories = []
    
    # Process each model tier separately to avoid GPU memory conflicts
    model_tiers = ["expert", "intermediate", "weak"]
    
    for model_tier in model_tiers:
        logger.info("=" * 60)
        logger.info(f"Processing all samples with {model_tier.upper()} model")
        logger.info("=" * 60)
        
        # Load model for this tier
        config = model_configs[model_tier]
        try:
            logger.info(f"Loading {model_tier} model from {config['path']}...")
            generator = ModelGenerator(
                model_path=config["path"],
                device=config["device"],
                quantization=config["quantization"],
                max_new_tokens=config["max_new_tokens"],
                use_auto_device_map=config["use_auto_device_map"]
            )
            logger.info(f"{model_tier.capitalize()} model loaded successfully!")
        except Exception as e:
            logger.error(f"Failed to load {model_tier} model: {e}")
            stats[f"{model_tier}_failed"] = len(bfcl_objects) * args.trajectories_per_sample
            continue
        
        # Process all samples with this model
        # Calculate total iterations (samples × trajectories_per_sample)
        total_iterations = len(bfcl_objects) * args.trajectories_per_sample
        
        # Use tqdm progress bar for full runs, verbose logging for test mode
        if use_progress_bar:
            # Create a flat iterator over (sample_idx, trajectory_idx, bfcl_obj)
            def iteration_generator():
                iter_num = 0
                for sample_idx, (file_path, bfcl_obj) in enumerate(bfcl_objects, 1):
                    for traj_idx in range(1, args.trajectories_per_sample + 1):
                        iter_num += 1
                        yield iter_num, sample_idx, traj_idx, file_path, bfcl_obj
            
            sample_iterator = tqdm(
                iteration_generator(),
                total=total_iterations,
                desc=f"[{model_tier}]",
                unit="traj"
            )
        else:
            def iteration_generator():
                iter_num = 0
                for sample_idx, (file_path, bfcl_obj) in enumerate(bfcl_objects, 1):
                    for traj_idx in range(1, args.trajectories_per_sample + 1):
                        iter_num += 1
                        yield iter_num, sample_idx, traj_idx, file_path, bfcl_obj
            sample_iterator = iteration_generator()
        
        for iter_num, sample_idx, traj_idx, file_path, bfcl_obj in sample_iterator:
            obj_id = bfcl_obj.get("id", "unknown")
            involved_classes = bfcl_obj.get("involved_classes", [])
            tool_paths = bfcl_obj.get("path", [])
            
            # Load tool docs for this object
            tool_docs = load_tool_docs(involved_classes, func_doc_dir)
            
            # Only log verbose output in test mode (not using progress bar)
            if not use_progress_bar:
                if args.trajectories_per_sample > 1:
                    logger.info(f"[{model_tier}] Processing {obj_id} traj {traj_idx}/{args.trajectories_per_sample} ({sample_idx}/{len(bfcl_objects)})...")
                else:
                    logger.info(f"[{model_tier}] Processing {obj_id} ({sample_idx}/{len(bfcl_objects)})...")
            
            try:
                builder = TrajectoryBuilder(
                    model_generator=generator,
                    tool_docs=tool_docs,
                    func_doc_dir=func_doc_dir
                )
                
                # Use unique model_tier identifier when generating multiple trajectories
                tier_id = f"{model_tier}_t{traj_idx}" if args.trajectories_per_sample > 1 else model_tier
                trajectory = builder.build_trajectory(bfcl_obj, tier_id)
                
                if trajectory:
                    all_trajectories.append(trajectory)
                    stats[f"{model_tier}_success"] += 1
                else:
                    stats[f"{model_tier}_failed"] += 1
                    if not use_progress_bar:
                        logger.warning(f"Failed to generate trajectory for {obj_id} with {model_tier} model")
            
            except Exception as e:
                stats[f"{model_tier}_failed"] += 1
                if not use_progress_bar:
                    logger.error(f"Error processing {obj_id} with {model_tier} model: {e}")
        
        # Unload model to free GPU memory
        logger.info(f"Unloading {model_tier} model to free GPU memory...")
        del generator
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        logger.info(f"GPU memory cleared")
    
    # Write output
    output_path.parent.mkdir(parents=True, exist_ok=True)
    logger.info(f"Writing {len(all_trajectories)} trajectories to {output_path}...")
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(all_trajectories, f, ensure_ascii=False, indent=2)
    
    # Log final statistics
    logger.info("=" * 60)
    logger.info("Trajectory generation complete!")
    logger.info(f"Total samples: {stats['total_samples']}")
    if stats['trajectories_per_sample'] > 1:
        logger.info(f"Trajectories per sample: {stats['trajectories_per_sample']}")
        logger.info(f"Total iterations per tier: {stats['total_iterations_per_tier']}")
    logger.info(f"Expert model: {stats['expert_success']} success, {stats['expert_failed']} failed")
    logger.info(f"Intermediate model: {stats['intermediate_success']} success, {stats['intermediate_failed']} failed")
    logger.info(f"Weak model: {stats['weak_success']} success, {stats['weak_failed']} failed")
    logger.info(f"Total trajectories generated: {len(all_trajectories)}")
    logger.info(f"Output written to: {output_path}")
    logger.info("=" * 60)
    
    return 0


if __name__ == "__main__":
    exit(main())

