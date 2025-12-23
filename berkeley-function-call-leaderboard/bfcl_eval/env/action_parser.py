"""
Action Parser for Tool Calling Environment

Parses various action formats from instruct models into standardized format.
Supports Python-style, JSON, and list formats.
"""

import ast
import json
import re
from typing import Any, Dict, List, Tuple, Union


class ActionParser:
    """Parse tool call strings in various formats."""

    @staticmethod
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
                return ast.literal_eval(value)
            except:
                return value
        
        # Handle dicts
        if value.startswith('{') and value.endswith('}'):
            try:
                return ast.literal_eval(value)
            except:
                return value
        
        # Default: return as string
        return value

    @staticmethod
    def parse_python_style(func_call: str) -> Dict[str, Any]:
        """
        Parse Python-style function call to dict format.
        
        Args:
            func_call: Function call string (e.g., "cd(folder='document')")
            
        Returns:
            Dictionary with "name" and "arguments" keys
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
            # Handle parameter parsing with proper type conversion
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
                    if '=' in current_param:
                        key, value = current_param.split('=', 1)
                        key = key.strip()
                        value = value.strip()
                        
                        # Convert value to appropriate type
                        arguments[key] = ActionParser.parse_parameter_value(value)
                    current_param = ""
                else:
                    current_param += char
        
        return {"name": func_name, "arguments": arguments}

    @staticmethod
    def parse_json_format(func_call_dict: Dict[str, Any]) -> Dict[str, Any]:
        """
        Parse JSON format function call (already a dict).
        
        Args:
            func_call_dict: Dictionary with "name" and "arguments" keys
            
        Returns:
            Standardized dictionary format
        """
        return {
            "name": func_call_dict.get("name", ""),
            "arguments": func_call_dict.get("arguments", {})
        }

    @staticmethod
    def parse_list_format(action_list: Union[str, List]) -> List[Dict[str, Any]]:
        """
        Parse list format actions (e.g., "[cd(folder='doc'), mkdir(dir_name='temp')]").
        
        Args:
            action_list: String representation of list or actual list
            
        Returns:
            List of parsed action dictionaries
        """
        if isinstance(action_list, str):
            # Remove outer brackets and split by comma (respecting nested structures)
            action_list = action_list.strip()
            if action_list.startswith('[') and action_list.endswith(']'):
                action_list = action_list[1:-1]
            
            # Split by commas that are not inside parentheses or quotes
            actions = []
            current_action = ""
            paren_depth = 0
            in_quotes = False
            quote_char = None
            
            for char in action_list:
                if char in ('"', "'") and (not in_quotes or char == quote_char):
                    if not in_quotes:
                        in_quotes = True
                        quote_char = char
                    else:
                        in_quotes = False
                        quote_char = None
                    current_action += char
                elif char == '(' and not in_quotes:
                    paren_depth += 1
                    current_action += char
                elif char == ')' and not in_quotes:
                    paren_depth -= 1
                    current_action += char
                elif char == ',' and paren_depth == 0 and not in_quotes:
                    if current_action.strip():
                        actions.append(current_action.strip())
                    current_action = ""
                else:
                    current_action += char
            
            # Add last action
            if current_action.strip():
                actions.append(current_action.strip())
        else:
            actions = action_list
        
        # Parse each action
        parsed_actions = []
        for action in actions:
            if isinstance(action, str):
                parsed_actions.append(ActionParser.parse_python_style(action))
            elif isinstance(action, dict):
                parsed_actions.append(ActionParser.parse_json_format(action))
        
        return parsed_actions

    @staticmethod
    def parse(action: Union[str, Dict, List]) -> List[Dict[str, Any]]:
        """
        Parse action in any supported format.
        
        Args:
            action: Action in Python-style, JSON, or list format
            
        Returns:
            List of parsed action dictionaries (always returns a list)
        """
        # If it's already a dict, return as single-item list
        if isinstance(action, dict):
            return [ActionParser.parse_json_format(action)]
        
        # If it's a list, parse as list format
        if isinstance(action, list):
            return ActionParser.parse_list_format(action)
        
        # If it's a string
        if isinstance(action, str):
            action = action.strip()
            
            # Try to parse as JSON first
            if action.startswith('{'):
                try:
                    json_obj = json.loads(action)
                    return [ActionParser.parse_json_format(json_obj)]
                except json.JSONDecodeError:
                    pass
            
            # Try to parse as list
            if action.startswith('['):
                return ActionParser.parse_list_format(action)
            
            # Otherwise, parse as Python-style single function call
            return [ActionParser.parse_python_style(action)]
        
        raise ValueError(f"Unsupported action format: {type(action)}")

    @staticmethod
    def format_action(name: str, arguments: Dict[str, Any]) -> str:
        """
        Format action back to Python-style string.
        
        Args:
            name: Function name
            arguments: Function arguments dict
            
        Returns:
            Python-style function call string
        """
        if not arguments:
            return f"{name}()"
        
        arg_strs = []
        for key, value in arguments.items():
            if isinstance(value, str):
                arg_strs.append(f"{key}='{value}'")
            elif isinstance(value, bool):
                arg_strs.append(f"{key}={value}")
            elif isinstance(value, (int, float)):
                arg_strs.append(f"{key}={value}")
            elif isinstance(value, (list, dict)):
                arg_strs.append(f"{key}={value}")
            elif value is None:
                arg_strs.append(f"{key}=None")
            else:
                arg_strs.append(f"{key}={repr(value)}")
        
        return f"{name}({', '.join(arg_strs)})"

