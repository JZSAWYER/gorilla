"""
API Manager for Tool Calling Environment

Loads and manages all API instances, routing tool calls to appropriate methods.
"""

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

# Import APIs with graceful fallback
try:
    from bfcl_eval.eval_checker.multi_turn_eval.func_source_code.gorilla_file_system import GorillaFileSystem
except ImportError:
    GorillaFileSystem = None

try:
    from bfcl_eval.eval_checker.multi_turn_eval.func_source_code.posting_api import TwitterAPI
except ImportError:
    TwitterAPI = None

try:
    from bfcl_eval.eval_checker.multi_turn_eval.func_source_code.message_api import MessageAPI
except ImportError:
    MessageAPI = None

try:
    from bfcl_eval.eval_checker.multi_turn_eval.func_source_code.ticket_api import TicketAPI
except ImportError:
    TicketAPI = None

try:
    from bfcl_eval.eval_checker.multi_turn_eval.func_source_code.trading_bot import TradingBot
except ImportError:
    TradingBot = None

try:
    from bfcl_eval.eval_checker.multi_turn_eval.func_source_code.travel_booking import TravelAPI
except ImportError:
    TravelAPI = None

try:
    from bfcl_eval.eval_checker.multi_turn_eval.func_source_code.vehicle_control import VehicleControlAPI
except ImportError:
    VehicleControlAPI = None

try:
    from bfcl_eval.eval_checker.multi_turn_eval.func_source_code.web_search import WebSearchAPI
except ImportError:
    WebSearchAPI = None

try:
    from bfcl_eval.eval_checker.multi_turn_eval.func_source_code.math_api import MathAPI
except ImportError:
    MathAPI = None

try:
    from bfcl_eval.eval_checker.multi_turn_eval.func_source_code.memory_kv import MemoryKeyValueAPI
except ImportError:
    MemoryKeyValueAPI = None

try:
    from bfcl_eval.eval_checker.multi_turn_eval.func_source_code.memory_vector import MemoryVectorAPI
except ImportError:
    MemoryVectorAPI = None

try:
    from bfcl_eval.eval_checker.multi_turn_eval.func_source_code.memory_rec_sum import MemoryRecentSummaryAPI
except ImportError:
    MemoryRecentSummaryAPI = None


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

# Class mapping (filter out None values for APIs that failed to import)
API_CLASSES = {
    k: v for k, v in {
        "GorillaFileSystem": GorillaFileSystem,
        "TwitterAPI": TwitterAPI,
        "MessageAPI": MessageAPI,
        "TicketAPI": TicketAPI,
        "TradingBot": TradingBot,
        "TravelAPI": TravelAPI,
        "VehicleControlAPI": VehicleControlAPI,
        "WebSearchAPI": WebSearchAPI,
        "MathAPI": MathAPI,
        "MemoryAPI_kv": MemoryKeyValueAPI,
        "MemoryAPI_vector": MemoryVectorAPI,
        "MemoryAPI_rec_sum": MemoryRecentSummaryAPI,
    }.items() if v is not None
}


class APIManager:
    """Manage API instances and route tool calls."""

    def __init__(self, func_doc_dir: Optional[Path] = None):
        """
        Initialize API manager.
        
        Args:
            func_doc_dir: Path to multi_turn_func_doc directory
        """
        self.api_instances: Dict[str, Any] = {}
        self.tool_docs: Dict[str, Dict[str, Any]] = {}
        
        # Set default func_doc_dir if not provided
        if func_doc_dir is None:
            current_file = Path(__file__)
            func_doc_dir = current_file.parent.parent / "data" / "multi_turn_func_doc"
        
        self.func_doc_dir = func_doc_dir
        
    def load_tool_docs(self, involved_classes: List[str]) -> None:
        """
        Load tool documentation for involved classes.
        
        Args:
            involved_classes: List of class names to load documentation for
        """
        self.tool_docs = {}
        
        for class_name in involved_classes:
            # Handle special case for MemoryAPI variants
            doc_file = CLASS_TO_DOC_FILE.get(class_name)
            if not doc_file and class_name.startswith("MemoryAPI"):
                # Try generic MemoryAPI
                doc_file = CLASS_TO_DOC_FILE.get(f"MemoryAPI_{class_name.split('_')[-1]}")
            
            if not doc_file:
                continue
                
            doc_path = self.func_doc_dir / doc_file
            if not doc_path.exists():
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
                            self.tool_docs[tool_name] = tool_def
                            self.tool_docs[f"{class_name}.{tool_name}"] = tool_def
            except Exception as e:
                print(f"Error loading tool docs from {doc_path}: {e}")
    
    def initialize_apis(self, initial_config: Dict[str, Any]) -> None:
        """
        Initialize API instances based on initial configuration.
        
        Args:
            initial_config: Dictionary mapping class names to their configurations
        """
        self.api_instances = {}
        
        for class_name, config in initial_config.items():
            # Get the API class
            api_class = API_CLASSES.get(class_name)
            if api_class is None:
                # Try MemoryAPI variants
                if class_name.startswith("MemoryAPI"):
                    variant = class_name.split("_")[-1] if "_" in class_name else "kv"
                    api_class = API_CLASSES.get(f"MemoryAPI_{variant}")
            
            if api_class is None:
                print(f"Warning: Unknown API class {class_name}")
                continue
            
            # Instantiate the API
            instance = api_class()
            
            # Load scenario if the API has this method
            if hasattr(instance, '_load_scenario'):
                instance._load_scenario(config)
            
            self.api_instances[class_name] = instance
    
    def execute_tool_call(self, tool_name: str, arguments: Dict[str, Any]) -> tuple:
        """
        Execute a tool call by routing to the appropriate API method.
        
        Args:
            tool_name: Name of the tool to execute
            arguments: Dictionary of arguments
            
        Returns:
            Tuple of (result, api_instance) where:
            - result: Result of the tool call execution
            - api_instance: The API instance that executed the tool (or None on error)
        """
        # RC-GRPO: Validate tool_name is a string (prevents 'attribute name must be string' error)
        if not isinstance(tool_name, str):
            return (
                {
                    "error": f"Tool name must be a string, got {type(tool_name).__name__}",
                    "received": str(tool_name)[:100]  # Truncate for safety
                },
                None
            )
        
        # Find which API this tool belongs to
        api_instance = None
        method_name = tool_name
        
        # Check if tool_name includes class prefix (e.g., "GorillaFileSystem.cd")
        if '.' in tool_name:
            class_name, method_name = tool_name.rsplit('.', 1)
            api_instance = self.api_instances.get(class_name)
        else:
            # Search through all instances for a method with this name
            for instance in self.api_instances.values():
                if hasattr(instance, method_name):
                    api_instance = instance
                    break
        
        if api_instance is None:
            return (
                {
                    "error": f"Tool '{tool_name}' not found in available APIs",
                    "available_apis": list(self.api_instances.keys())
                },
                None
            )
        
        if not hasattr(api_instance, method_name):
            return (
                {
                    "error": f"Method '{method_name}' not found in API",
                    "api_type": type(api_instance).__name__
                },
                None
            )
        
        # Get the method
        method = getattr(api_instance, method_name)
        
        # Execute the method with arguments
        try:
            result = method(**arguments)
            return result, api_instance
        except TypeError as e:
            return (
                {
                    "error": f"Invalid arguments for {method_name}: {str(e)}",
                    "provided_arguments": arguments
                },
                None
            )
        except Exception as e:
            return (
                {
                    "error": f"Error executing {method_name}: {str(e)}",
                    "error_type": type(e).__name__
                },
                None
            )
    
    def get_tool_documentation(self, tool_name: str) -> Optional[Dict[str, Any]]:
        """
        Get documentation for a specific tool.
        
        Args:
            tool_name: Name of the tool
            
        Returns:
            Tool documentation dictionary or None
        """
        return self.tool_docs.get(tool_name)
    
    def get_all_tool_docs(self) -> Dict[str, Dict[str, Any]]:
        """
        Get all loaded tool documentation.
        
        Returns:
            Dictionary of all tool documentation
        """
        return self.tool_docs
    
    def get_api_state(self, class_name: str) -> Optional[Dict[str, Any]]:
        """
        Get the current state of a specific API.
        
        Args:
            class_name: Name of the API class
            
        Returns:
            State dictionary or None if API not found
        """
        instance = self.api_instances.get(class_name)
        if instance is None:
            return None
        
        # Extract state from instance attributes
        state = {}
        for attr_name in dir(instance):
            if not attr_name.startswith('_') and not callable(getattr(instance, attr_name)):
                state[attr_name] = getattr(instance, attr_name)
        
        return state
    
    def get_all_states(self) -> Dict[str, Dict[str, Any]]:
        """
        Get states of all initialized APIs.
        
        Returns:
            Dictionary mapping class names to their states
        """
        states = {}
        for class_name in self.api_instances.keys():
            state = self.get_api_state(class_name)
            if state:
                states[class_name] = state
        return states
    
    def reset(self) -> None:
        """Reset all API instances."""
        self.api_instances = {}
        self.tool_docs = {}

