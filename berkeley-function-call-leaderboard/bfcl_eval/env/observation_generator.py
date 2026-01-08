"""
Observation Generator for Tool Calling Environment

Generates detailed observations from API responses in Glaive-compatible format.
"""

import json
from typing import Any, Dict, Optional, Union


def convert_mpf_to_float(obj):
    """
    Recursively convert mpf objects to Python floats for JSON serialization.
    
    Args:
        obj: Object that may contain mpf values
        
    Returns:
        Object with mpf values converted to floats
    """
    if hasattr(obj, '__class__') and obj.__class__.__name__ == 'mpf':
        return float(obj)
    elif isinstance(obj, dict):
        return {k: convert_mpf_to_float(v) for k, v in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return type(obj)(convert_mpf_to_float(item) for item in obj)
    return obj


class ObservationGenerator:
    """Generate detailed observations from API responses."""

    # Define standard observation schemas for each API class
    # These keys will always be present in observations for consistency
    API_SCHEMAS = {
        "GorillaFileSystem": {
            "required_state": ["current_working_directory", "directory_contents"],
            "optional_fields": ["result", "context"]
        },
        "VehicleControlAPI": {
            "required_state": ["fuel_level", "battery_voltage", "engine_state", "door_status",
                               "ac_temperature", "fan_speed", "ac_mode", "headlight_status",
                               "parking_brake_status", "brake_pedal_status",
                               "front_left_tire_pressure", "front_right_tire_pressure",
                               "rear_left_tire_pressure", "rear_right_tire_pressure",
                               "cruise_status", "destination"],
            "optional_fields": ["result", "context"]
        },
        "TwitterAPI": {
            "required_state": ["authenticated", "username", "tweet_count"],
            "optional_fields": ["result", "context"]
        },
        "MessageAPI": {
            "required_state": ["login_status", "message_count_sent", "message_count_received", "total_contacts"],
            "optional_fields": ["result", "context"]
        },
        "TicketAPI": {
            "required_state": ["login_status", "total_tickets", "open_tickets", "resolved_tickets", "closed_tickets"],
            "optional_fields": ["result", "context"]
        },
        "MathAPI": {
            "required_state": [],  # Stateless API
            "optional_fields": ["result", "context"]
        },
        "TradingBot": {
            "required_state": ["authenticated", "account_balance", "account_id", "market_status",
                               "total_orders", "watch_list_count", "transaction_count"],
            "optional_fields": ["result", "context"]
        },
        "TravelAPI": {
            "required_state": ["authenticated", "total_bookings", "total_credit_cards", "budget_limit",
                               "user_first_name", "user_last_name"],
            "optional_fields": ["result", "context"]
        },
        "WebSearchAPI": {
            "required_state": [],  # Stateless API
            "optional_fields": ["result", "context"]
        },
        "MemoryAPI_kv": {
            "required_state": ["core_memory_count", "archival_memory_count"],
            "optional_fields": ["result", "context"]
        },
        "MemoryAPI_vector": {
            "required_state": ["core_memory_count", "archival_memory_count"],
            "optional_fields": ["result", "context"]
        },
        "MemoryAPI_rec_sum": {
            "required_state": ["memory_length", "memory_empty"],
            "optional_fields": ["result", "context"]
        }
    }

    def __init__(self):
        """Initialize observation generator."""
        self.last_observation = None
    
    @staticmethod
    def _get_filesystem_state_info(api_instance) -> Dict[str, Any]:
        """
        Extract current filesystem state for consistent observations.
        
        Always returns ALL possible filesystem state keys for RL consistency.
        
        Args:
            api_instance: The GorillaFileSystem instance
            
        Returns:
            Dictionary with comprehensive filesystem state information (consistent keys)
        """
        # Define ALL possible filesystem state keys for consistency
        state_info = {
            # Core state (always present)
            "current_working_directory": None,
            "directory_contents": [],
            # Operation result and context
            "result": None,
            "context": None
        }
        
        try:
            # Get current working directory
            if api_instance:
                pwd_result = api_instance.pwd()
                if "current_working_directory" in pwd_result:
                    state_info["current_working_directory"] = pwd_result["current_working_directory"]
                
                # Get current directory contents
                ls_result = api_instance.ls()
                if "current_directory_content" in ls_result:
                    state_info["directory_contents"] = ls_result["current_directory_content"]
        except:
            pass  # Silently fail if state query fails, use defaults
        
        return state_info
    
    @staticmethod
    def _get_vehicle_state_info(api_instance) -> Dict[str, Any]:
        """
        Extract current vehicle state for consistent observations.
        
        Always returns ALL possible vehicle state keys for RL consistency.
        
        Args:
            api_instance: The VehicleControlAPI instance
            
        Returns:
            Dictionary with comprehensive vehicle state information (consistent keys)
        """
        # Define ALL possible vehicle state keys for consistency
        state_info = {
            # Core state (always present)
            "fuel_level": None,
            "battery_voltage": None,
            "engine_state": None,
            "door_status": None,
            # Climate control
            "ac_temperature": None,
            "fan_speed": None,
            "ac_mode": None,
            # Lights and brakes
            "headlight_status": None,
            "parking_brake_status": None,
            "brake_pedal_status": None,
            # Tires
            "front_left_tire_pressure": None,
            "front_right_tire_pressure": None,
            "rear_left_tire_pressure": None,
            "rear_right_tire_pressure": None,
            # Other
            "cruise_status": None,
            "destination": None,
            "result": None,
            "context": None
        }
        
        try:
            # Get state from the API instance attributes
            if hasattr(api_instance, 'fuelLevel'):
                state_info["fuel_level"] = api_instance.fuelLevel
            if hasattr(api_instance, 'batteryVoltage'):
                state_info["battery_voltage"] = api_instance.batteryVoltage
            if hasattr(api_instance, 'engine_state'):
                state_info["engine_state"] = api_instance.engine_state
            if hasattr(api_instance, 'doorStatus'):
                state_info["door_status"] = api_instance.doorStatus
            if hasattr(api_instance, 'acTemperature'):
                state_info["ac_temperature"] = api_instance.acTemperature
            if hasattr(api_instance, 'fanSpeed'):
                state_info["fan_speed"] = api_instance.fanSpeed
            if hasattr(api_instance, 'acMode'):
                state_info["ac_mode"] = api_instance.acMode
            if hasattr(api_instance, 'headLightStatus'):
                state_info["headlight_status"] = api_instance.headLightStatus
            if hasattr(api_instance, 'parkingBrakeStatus'):
                state_info["parking_brake_status"] = api_instance.parkingBrakeStatus
            if hasattr(api_instance, 'brakePedalStatus'):
                state_info["brake_pedal_status"] = api_instance.brakePedalStatus
            if hasattr(api_instance, 'frontLeftTirePressure'):
                state_info["front_left_tire_pressure"] = api_instance.frontLeftTirePressure
            if hasattr(api_instance, 'frontRightTirePressure'):
                state_info["front_right_tire_pressure"] = api_instance.frontRightTirePressure
            if hasattr(api_instance, 'rearLeftTirePressure'):
                state_info["rear_left_tire_pressure"] = api_instance.rearLeftTirePressure
            if hasattr(api_instance, 'rearRightTirePressure'):
                state_info["rear_right_tire_pressure"] = api_instance.rearRightTirePressure
            if hasattr(api_instance, 'cruiseStatus'):
                state_info["cruise_status"] = api_instance.cruiseStatus
            if hasattr(api_instance, 'destination'):
                state_info["destination"] = api_instance.destination
        except:
            pass  # Silently fail if state query fails, use defaults
        
        return state_info
    
    @staticmethod
    def _get_twitter_state_info(api_instance) -> Dict[str, Any]:
        """Extract current TwitterAPI state for consistent observations."""
        state_info = {
            "authenticated": False,
            "username": None,
            "tweet_count": 0,
            "result": None,
            "context": None
        }
        
        try:
            if api_instance:
                state_info["authenticated"] = getattr(api_instance, 'authenticated', False)
                state_info["username"] = getattr(api_instance, 'username', None)
                tweets = getattr(api_instance, 'tweets', {})
                state_info["tweet_count"] = len(tweets) if isinstance(tweets, dict) else 0
        except:
            pass
        
        return state_info
    
    @staticmethod
    def _get_message_state_info(api_instance) -> Dict[str, Any]:
        """Extract current MessageAPI state for consistent observations."""
        state_info = {
            "login_status": False,
            "message_count_sent": 0,
            "message_count_received": 0,
            "total_contacts": 0,
            "result": None,
            "context": None
        }
        
        try:
            if api_instance:
                current_user = getattr(api_instance, 'current_user', None)
                state_info["login_status"] = current_user is not None
                
                inbox = getattr(api_instance, 'inbox', [])
                if isinstance(inbox, list):
                    # Count messages sent by current user
                    sent_count = 0
                    received_count = len(inbox)
                    contacts = set()
                    
                    for message in inbox:
                        if isinstance(message, dict):
                            receiver_id = list(message.keys())[0] if message else None
                            if receiver_id:
                                contacts.add(receiver_id)
                    
                    state_info["message_count_sent"] = sent_count
                    state_info["message_count_received"] = received_count
                    state_info["total_contacts"] = len(contacts)
        except:
            pass
        
        return state_info
    
    @staticmethod
    def _get_ticket_state_info(api_instance) -> Dict[str, Any]:
        """Extract current TicketAPI state for consistent observations."""
        state_info = {
            "login_status": False,
            "total_tickets": 0,
            "open_tickets": 0,
            "resolved_tickets": 0,
            "closed_tickets": 0,
            "result": None,
            "context": None
        }
        
        try:
            if api_instance:
                current_user = getattr(api_instance, 'current_user', None)
                state_info["login_status"] = current_user is not None
                
                ticket_queue = getattr(api_instance, 'ticket_queue', [])
                if isinstance(ticket_queue, list):
                    state_info["total_tickets"] = len(ticket_queue)
                    for ticket in ticket_queue:
                        if isinstance(ticket, dict):
                            status = ticket.get('status', '').lower()
                            if status == 'open':
                                state_info["open_tickets"] += 1
                            elif status == 'resolved':
                                state_info["resolved_tickets"] += 1
                            elif status == 'closed':
                                state_info["closed_tickets"] += 1
        except:
            pass
        
        return state_info
    
    @staticmethod
    def _get_trading_state_info(api_instance) -> Dict[str, Any]:
        """Extract current TradingBot state for consistent observations."""
        state_info = {
            "authenticated": False,
            "account_balance": 0.0,
            "account_id": None,
            "market_status": None,
            "total_orders": 0,
            "watch_list_count": 0,
            "transaction_count": 0,
            "result": None,
            "context": None
        }
        
        try:
            if api_instance:
                state_info["authenticated"] = getattr(api_instance, 'authenticated', False)
                account_info = getattr(api_instance, 'account_info', {})
                if isinstance(account_info, dict):
                    state_info["account_balance"] = account_info.get('balance', 0.0)
                    state_info["account_id"] = account_info.get('account_id', None)
                state_info["market_status"] = getattr(api_instance, 'market_status', None)
                
                orders = getattr(api_instance, 'orders', {})
                state_info["total_orders"] = len(orders) if isinstance(orders, dict) else 0
                
                watch_list = getattr(api_instance, 'watch_list', [])
                state_info["watch_list_count"] = len(watch_list) if isinstance(watch_list, list) else 0
                
                transaction_history = getattr(api_instance, 'transaction_history', [])
                state_info["transaction_count"] = len(transaction_history) if isinstance(transaction_history, list) else 0
        except:
            pass
        
        return state_info
    
    @staticmethod
    def _get_travel_state_info(api_instance) -> Dict[str, Any]:
        """Extract current TravelAPI state for consistent observations."""
        state_info = {
            "authenticated": False,
            "total_bookings": 0,
            "total_credit_cards": 0,
            "budget_limit": None,
            "user_first_name": None,
            "user_last_name": None,
            "result": None,
            "context": None
        }
        
        try:
            if api_instance:
                token_expires_in = getattr(api_instance, 'token_expires_in', None)
                state_info["authenticated"] = token_expires_in is not None and token_expires_in > 0
                
                booking_record = getattr(api_instance, 'booking_record', {})
                state_info["total_bookings"] = len(booking_record) if isinstance(booking_record, dict) else 0
                
                credit_card_list = getattr(api_instance, 'credit_card_list', {})
                state_info["total_credit_cards"] = len(credit_card_list) if isinstance(credit_card_list, dict) else 0
                
                state_info["budget_limit"] = getattr(api_instance, 'budget_limit', None)
                state_info["user_first_name"] = getattr(api_instance, 'user_first_name', None)
                state_info["user_last_name"] = getattr(api_instance, 'user_last_name', None)
        except:
            pass
        
        return state_info
    
    @staticmethod
    def _get_web_search_state_info(api_instance) -> Dict[str, Any]:
        """Extract current WebSearchAPI state for consistent observations."""
        # WebSearchAPI is stateless, so we just return minimal consistent schema
        state_info = {
            "result": None,
            "context": None
        }
        return state_info
    
    @staticmethod
    def _get_memory_state_info(api_instance) -> Dict[str, Any]:
        """Extract current MemoryAPI state for consistent observations."""
        # Check which variant we're dealing with
        try:
            if api_instance:
                # Check if it's rec_sum variant (has 'memory' attribute as string)
                if hasattr(api_instance, 'memory') and isinstance(getattr(api_instance, 'memory'), str):
                    memory = getattr(api_instance, 'memory', '')
                    state_info = {
                        "memory_length": len(memory) if isinstance(memory, str) else 0,
                        "memory_empty": not memory or len(memory) == 0,
                        "result": None,
                        "context": None
                    }
                    return state_info
        except:
            pass
        
        # Default to kv/vector variant schema
        state_info = {
            "core_memory_count": 0,
            "archival_memory_count": 0,
            "result": None,
            "context": None
        }
        
        try:
            if api_instance:
                # Check if it's kv variant
                if hasattr(api_instance, 'core_memory') and isinstance(getattr(api_instance, 'core_memory'), dict):
                    core_memory = getattr(api_instance, 'core_memory', {})
                    archival_memory = getattr(api_instance, 'archival_memory', {})
                    state_info["core_memory_count"] = len(core_memory) if isinstance(core_memory, dict) else 0
                    state_info["archival_memory_count"] = len(archival_memory) if isinstance(archival_memory, dict) else 0
                # Check if it's vector variant
                elif hasattr(api_instance, 'core_memory') and hasattr(getattr(api_instance, 'core_memory'), '_store'):
                    core_memory = getattr(api_instance, 'core_memory')
                    archival_memory = getattr(api_instance, 'archival_memory')
                    state_info["core_memory_count"] = len(core_memory._store) if hasattr(core_memory, '_store') else 0
                    state_info["archival_memory_count"] = len(archival_memory._store) if hasattr(archival_memory, '_store') else 0
        except:
            pass
        
        return state_info
    
    @staticmethod
    def format_file_system_observation(
        result: Dict[str, Any], 
        tool_name: str,
        api_instance: Optional[Any] = None,
        arguments: Optional[Dict[str, Any]] = None
    ) -> str:
        """
        Format GorillaFileSystem operation observations with consistent state.
        
        All filesystem observations include the same state keys for RL consistency:
        - current_working_directory
        - directory_contents
        - result (operation-specific message)
        - context (human-readable description)
        
        Args:
            result: Result from file system operation
            tool_name: Name of the tool that was executed
            api_instance: Optional API instance for state queries
            arguments: Optional arguments passed to the tool
            
        Returns:
            Formatted observation string with consistent keys
        """
        if arguments is None:
            arguments = {}
        
        # Start with consistent state schema
        observation_dict = {
            "current_working_directory": None,
            "directory_contents": [],
            "result": None,
            "context": None
        }
        
        # Query current filesystem state if api_instance is available
        if api_instance:
            state_info = ObservationGenerator._get_filesystem_state_info(api_instance)
            observation_dict.update(state_info)
            
        # Handle error cases
        if "error" in result:
            observation_dict["result"] = result.get("error")
            observation_dict["context"] = f"Error in {tool_name}: {result.get('error')}"
            return json.dumps(convert_mpf_to_float(observation_dict))
        
        # Add contextual information based on operation type
        observations = []
        
        if tool_name == "cd":
            if "current_working_directory" in result:
                observation_dict["result"] = f"Changed to directory: {result['current_working_directory']}"
                observations.append(observation_dict["result"])
        
        elif tool_name == "ls":
            if "current_directory_content" in result:
                contents = result["current_directory_content"]
                if contents:
                    observation_dict["result"] = f"Directory contains {len(contents)} items"
                    observations.append(f"Directory contains {len(contents)} items: {', '.join(contents)}")
                else:
                    observation_dict["result"] = "Directory is empty"
                    observations.append("Directory is empty")
        
        elif tool_name == "cat":
            if "file_content" in result:
                content = result["file_content"]
                observation_dict["result"] = f"File content ({len(content)} characters)"
                # Put content in context instead of separate key for consistency
                observations.append(f"File content: {content}")
        
        elif tool_name == "grep":
            if "matching_lines" in result:
                matches = result["matching_lines"]
                if matches:
                    observation_dict["result"] = f"Found {len(matches)} matching lines"
                    # Put matches in context instead of separate key for consistency
                    observations.append(f"Matching lines: {matches}")
                else:
                    observation_dict["result"] = "No matching lines found"
                    observations.append("No matching lines found")
        
        elif tool_name == "find":
            if "matches" in result:
                matches = result["matches"]
                if matches:
                    observation_dict["result"] = f"Found {len(matches)} matches"
                    # Put matches in context instead of separate key for consistency
                    observations.append(f"Matches: {', '.join(matches)}")
                else:
                    observation_dict["result"] = "No matches found"
                    observations.append("No matches found")
        
        elif tool_name in ["mkdir", "touch", "rm", "rmdir", "mv", "cp"]:
            # Handle None results
            if "result" in result and (result["result"] is None or result["result"] == "None" or str(result["result"]) == "None"):
                # Provide meaningful messages for None results
                if tool_name == "mkdir":
                    dir_name = arguments.get("dir_name", "directory")
                    msg = f"Directory '{dir_name}' created successfully"
                    observation_dict["result"] = msg
                    observations.append(msg)
                elif tool_name == "touch":
                    file_name = arguments.get("file_name", "file")
                    msg = f"File '{file_name}' created successfully"
                    observation_dict["result"] = msg
                    observations.append(msg)
                elif tool_name == "rm":
                    file_name = arguments.get("file_name", "file")
                    msg = f"Removed '{file_name}' successfully"
                    observation_dict["result"] = msg
                    observations.append(msg)
                elif tool_name == "rmdir":
                    dir_name = arguments.get("dir_name", "directory")
                    msg = f"Directory '{dir_name}' removed successfully"
                    observation_dict["result"] = msg
                    observations.append(msg)
                else:
                    msg = f"Operation {tool_name} completed successfully"
                    observation_dict["result"] = msg
                    observations.append(msg)
            elif "result" in result:
                observation_dict["result"] = result["result"]
                observations.append(result["result"])
            else:
                msg = f"Operation {tool_name} completed successfully"
                observation_dict["result"] = msg
                observations.append(msg)
        
        elif tool_name == "wc":
            if "count" in result and "type" in result:
                observation_dict["result"] = f"Count: {result['count']} {result['type']}"
                # Put details in context instead of separate keys for consistency
                observations.append(f"Count: {result['count']} {result['type']}")
        
        elif tool_name == "tail":
            if "last_lines" in result:
                lines = result["last_lines"]
                observation_dict["result"] = "Tail operation completed"
                # Put lines in context instead of separate key for consistency
                observations.append(f"Last lines: {lines}")
        
        elif tool_name == "sort":
            if "sorted_content" in result:
                content = result["sorted_content"]
                observation_dict["result"] = "Content sorted successfully"
                # Put content in context instead of separate key for consistency
                observations.append(f"Sorted content: {content}")
        
        elif tool_name == "diff":
            if "diff_lines" in result:
                diff = result["diff_lines"]
                if diff:
                    observation_dict["result"] = "Differences found"
                    # Put diff in context instead of separate key for consistency
                    observations.append(f"Differences: {diff}")
                else:
                    observation_dict["result"] = "Files are identical"
                    observations.append("Files are identical")
        
        elif tool_name == "du":
            if "disk_usage" in result:
                observation_dict["result"] = f"Disk usage: {result['disk_usage']}"
                # Put details in context instead of separate key for consistency
                observations.append(f"Disk usage: {result['disk_usage']}")
        
        elif tool_name == "echo":
            if "terminal_output" in result:
                observation_dict["result"] = f"Output: {result['terminal_output']}"
                # Put output in context instead of separate key for consistency
                observations.append(f"Output: {result['terminal_output']}")
        
        elif tool_name == "pwd":
            if "current_working_directory" in result:
                observation_dict["result"] = f"Current directory: {result['current_working_directory']}"
                observations.append(f"Current directory: {result['current_working_directory']}")
        
        # Add context
        if observations:
            observation_dict["context"] = " | ".join(observations)
        else:
            observation_dict["context"] = f"Operation {tool_name} completed"
        
        # Ensure result is set
        if observation_dict["result"] is None:
            observation_dict["result"] = f"Operation {tool_name} completed"
        
        return json.dumps(convert_mpf_to_float(observation_dict))
    
    @staticmethod
    def format_vehicle_control_observation(
        result: Dict[str, Any],
        tool_name: str,
        api_instance: Optional[Any] = None,
        arguments: Optional[Dict[str, Any]] = None
    ) -> str:
        """
        Format VehicleControlAPI operation observations with consistent state.
        
        All vehicle control observations include the SAME comprehensive state keys for RL consistency.
        No operation-specific keys are added to maintain consistency across turns.
        
        Args:
            result: Result from vehicle control operation
            tool_name: Name of the tool that was executed
            api_instance: Optional API instance for state queries
            arguments: Optional arguments passed to the tool
            
        Returns:
            Formatted observation string with consistent keys
        """
        if arguments is None:
            arguments = {}
        
        # Get comprehensive vehicle state (includes all keys with null defaults)
        if api_instance:
            observation_dict = ObservationGenerator._get_vehicle_state_info(api_instance)
        else:
            # Use empty state if no api_instance
            observation_dict = ObservationGenerator._get_vehicle_state_info(None)
        
        # Handle error cases
        if "error" in result:
            observation_dict["result"] = result.get("error")
            observation_dict["context"] = f"Error in {tool_name}: {result.get('error')}"
            return json.dumps(convert_mpf_to_float(observation_dict))
        
        # Build context message from result (but don't add extra keys)
        observations = []
        for key, value in result.items():
            if isinstance(value, (str, int, float, bool)) and value is not None:
                observations.append(f"{key}: {value}")
        
        # Set result and context messages
        if observations:
            observation_dict["result"] = f"Operation {tool_name} completed"
            observation_dict["context"] = " | ".join(observations[:5])  # Limit to avoid very long strings
        else:
            observation_dict["result"] = f"Operation {tool_name} completed successfully"
            observation_dict["context"] = f"Operation {tool_name} completed successfully"
        
        return json.dumps(convert_mpf_to_float(observation_dict))
    
    @staticmethod
    def format_twitter_observation(
        result: Dict[str, Any], 
        tool_name: str,
        api_instance: Optional[Any] = None
    ) -> str:
        """
        Format TwitterAPI operation observations with consistent state.
        
        All TwitterAPI observations include the same state keys for RL consistency:
        - authenticated
        - username
        - tweet_count
        - result
        - context
        
        Args:
            result: Result from Twitter operation
            tool_name: Name of the tool that was executed
            api_instance: Optional API instance for state queries
            
        Returns:
            Formatted observation string with consistent keys
        """
        # Start with consistent state schema
        if api_instance:
            observation_dict = ObservationGenerator._get_twitter_state_info(api_instance)
        else:
            observation_dict = ObservationGenerator._get_twitter_state_info(None)
        
        # Handle error cases
        if "error" in result:
            observation_dict["result"] = result.get("error")
            observation_dict["context"] = f"Error in {tool_name}: {result.get('error')}"
            return json.dumps(convert_mpf_to_float(observation_dict))
        
        observations = []
        
        if tool_name == "post_tweet":
            if "id" in result:
                tweet_id = result["id"]
                content = result.get("content", "")
                observations.append(f"Tweet posted with ID {tweet_id}")
                if "tags" in result and result["tags"]:
                    observations.append(f"Tags: {', '.join(result['tags'])}")
                if "mentions" in result and result["mentions"]:
                    observations.append(f"Mentions: {', '.join(result['mentions'])}")
        
        elif tool_name == "retweet":
            if "retweet_status" in result:
                observations.append(f"Retweet status: {result['retweet_status']}")
        
        elif tool_name == "comment":
            if "comment_status" in result:
                observations.append(f"Comment status: {result['comment_status']}")
        
        elif tool_name == "authenticate_twitter":
            if "authentication_status" in result:
                status = "successful" if result["authentication_status"] else "failed"
                observations.append(f"Authentication {status}")
        
        elif tool_name == "posting_get_login_status":
            if "login_status" in result:
                status = "logged in" if result["login_status"] else "not logged in"
                observations.append(f"User is {status}")
        
        # RC-GRPO FIX: Add cases for discovery tools that need to return full content
        elif tool_name == "get_tweet":
            if "id" in result:
                observations.append(f"Tweet ID: {result['id']}")
            if "username" in result:
                observations.append(f"Author: {result['username']}")
            if "content" in result:
                # CRITICAL: Include the actual tweet content!
                observations.append(f"Content: {result['content']}")
            if "tags" in result and result["tags"]:
                observations.append(f"Tags: {', '.join(result['tags'])}")
            if "mentions" in result and result["mentions"]:
                observations.append(f"Mentions: {', '.join(result['mentions'])}")
        
        elif tool_name == "get_user_tweets":
            # Result is a list of tweets - may be wrapped in _list_result
            tweets = result.get("_list_result", result if isinstance(result, list) else [])
            if tweets:
                observations.append(f"Found {len(tweets)} tweets")
                for i, tweet in enumerate(tweets[:5]):  # Limit to first 5
                    if isinstance(tweet, dict) and "content" in tweet:
                        observations.append(f"Tweet {i+1}: {tweet['content']}")
        
        elif tool_name == "search_tweets":
            # Result is a list of matching tweets - may be wrapped in _list_result
            tweets = result.get("_list_result", result if isinstance(result, list) else [])
            if tweets:
                observations.append(f"Found {len(tweets)} matching tweets")
                for i, tweet in enumerate(tweets[:5]):  # Limit to first 5
                    if isinstance(tweet, dict) and "content" in tweet:
                        observations.append(f"Match {i+1}: {tweet['content']}")
        
        # Set result and context
        observation_dict["result"] = f"Operation {tool_name} completed"
        if observations:
            observation_dict["context"] = " | ".join(observations)
        else:
            observation_dict["context"] = f"Operation {tool_name} completed successfully"
        
        return json.dumps(convert_mpf_to_float(observation_dict))
    
    @staticmethod
    def format_message_observation(
        result: Dict[str, Any], 
        tool_name: str,
        api_instance: Optional[Any] = None
    ) -> str:
        """
        Format MessageAPI operation observations with consistent state.
        
        All MessageAPI observations include the same state keys for RL consistency:
        - login_status
        - message_count_sent
        - message_count_received
        - total_contacts
        - result
        - context
        
        Args:
            result: Result from message operation
            tool_name: Name of the tool that was executed
            api_instance: Optional API instance for state queries
            
        Returns:
            Formatted observation string with consistent keys
        """
        # Start with consistent state schema
        if api_instance:
            observation_dict = ObservationGenerator._get_message_state_info(api_instance)
        else:
            observation_dict = ObservationGenerator._get_message_state_info(None)
        
        # Handle error cases
        if "error" in result:
            observation_dict["result"] = result.get("error")
            observation_dict["context"] = f"Error in {tool_name}: {result.get('error')}"
            return json.dumps(convert_mpf_to_float(observation_dict))
        
        observations = []
        
        if tool_name == "send_message":
            if "sent_status" in result:
                observations.append(f"Message sent: {result['sent_status']}")
            if "message_id" in result:
                observations.append(f"Message ID: {result['message_id']}")
        
        elif tool_name == "view_messages_sent":
            if "messages" in result:
                messages = result["messages"]
                # RC-GRPO FIX: Include actual message content, not just count
                if isinstance(messages, dict):
                    count = sum(len(msgs) if isinstance(msgs, list) else 1 for msgs in messages.values())
                    observations.append(f"Retrieved {count} sent messages")
                    for receiver_id, msgs in list(messages.items())[:3]:  # Limit
                        if isinstance(msgs, list):
                            for msg in msgs[:2]:  # Limit messages per receiver
                                observations.append(f"To {receiver_id}: {msg}")
                        else:
                            observations.append(f"To {receiver_id}: {msgs}")
                else:
                    count = len(messages)
                    observations.append(f"Retrieved {count} sent messages")
        
        elif tool_name == "view_messages_received":
            if "messages" in result:
                messages = result["messages"]
                # RC-GRPO FIX: Include actual message content
                if isinstance(messages, dict):
                    count = sum(len(msgs) if isinstance(msgs, list) else 1 for msgs in messages.values())
                    observations.append(f"Retrieved {count} received messages")
                    for sender_id, msgs in list(messages.items())[:3]:  # Limit
                        if isinstance(msgs, list):
                            for msg in msgs[:2]:
                                observations.append(f"From {sender_id}: {msg}")
                        else:
                            observations.append(f"From {sender_id}: {msgs}")
                else:
                    count = len(messages)
                    observations.append(f"Retrieved {count} received messages")
        
        # RC-GRPO FIX: Add search_messages handler
        elif tool_name == "search_messages":
            # Result is {"results": [{"receiver_id": ..., "message": ...}, ...]}
            if "results" in result:
                messages = result["results"]
                if isinstance(messages, list):
                    observations.append(f"Found {len(messages)} matching messages")
                    for i, msg in enumerate(messages[:5]):  # Limit to first 5
                        if isinstance(msg, dict):
                            msg_content = msg.get("message", "")
                            receiver = msg.get("receiver_id", "")
                            observations.append(f"Match {i+1} (to {receiver}): {msg_content}")
                        else:
                            observations.append(f"Match {i+1}: {msg}")
        
        elif tool_name == "delete_message":
            if "deleted_status" in result:
                observations.append(f"Deletion: {result['deleted_status']}")
        
        elif tool_name == "message_login":
            if "login_status" in result:
                status = "successful" if result["login_status"] else "failed"
                observations.append(f"Login {status}")
        
        # Set result and context
        observation_dict["result"] = f"Operation {tool_name} completed"
        if observations:
            observation_dict["context"] = " | ".join(observations)
        else:
            observation_dict["context"] = f"Operation {tool_name} completed successfully"
        
        return json.dumps(convert_mpf_to_float(observation_dict))
    
    @staticmethod
    def format_ticket_observation(
        result: Dict[str, Any], 
        tool_name: str,
        api_instance: Optional[Any] = None
    ) -> str:
        """
        Format TicketAPI operation observations with consistent state.
        
        All TicketAPI observations include the same state keys for RL consistency:
        - login_status
        - total_tickets
        - open_tickets
        - resolved_tickets
        - closed_tickets
        - result
        - context
        
        Args:
            result: Result from ticket operation
            tool_name: Name of the tool that was executed
            api_instance: Optional API instance for state queries
            
        Returns:
            Formatted observation string with consistent keys
        """
        # Start with consistent state schema
        if api_instance:
            observation_dict = ObservationGenerator._get_ticket_state_info(api_instance)
        else:
            observation_dict = ObservationGenerator._get_ticket_state_info(None)
        
        # Handle error cases
        if "error" in result:
            observation_dict["result"] = result.get("error")
            observation_dict["context"] = f"Error in {tool_name}: {result.get('error')}"
            return json.dumps(convert_mpf_to_float(observation_dict))
        
        observations = []
        
        if tool_name == "create_ticket":
            if "id" in result:
                observations.append(f"Ticket created with ID {result['id']}")
            if "status" in result:
                observations.append(f"Status: {result['status']}")
        
        elif tool_name == "get_ticket":
            if "id" in result:
                observations.append(f"Ticket ID: {result['id']}")
            if "title" in result:
                observations.append(f"Title: {result['title']}")
            # RC-GRPO FIX: Include the description - critical for discovery
            if "description" in result:
                observations.append(f"Description: {result['description']}")
            if "status" in result:
                observations.append(f"Status: {result['status']}")
            if "priority" in result:
                observations.append(f"Priority: {result['priority']}")
            if "created_by" in result:
                observations.append(f"Created by: {result['created_by']}")
        
        elif tool_name == "resolve_ticket":
            if "status" in result:
                observations.append(f"Resolution: {result['status']}")
        
        elif tool_name == "close_ticket":
            if "status" in result:
                observations.append(f"Closure: {result['status']}")
        
        elif tool_name == "edit_ticket":
            if "status" in result:
                observations.append(f"Edit: {result['status']}")
        
        elif tool_name == "ticket_login":
            if "success" in result:
                status = "successful" if result["success"] else "failed"
                observations.append(f"Login {status}")
        
        # RC-GRPO FIX: Add get_user_tickets handler
        elif tool_name == "get_user_tickets":
            # Handle list result - may be wrapped in _list_result
            tickets = result.get("_list_result", result if isinstance(result, list) else [])
            if tickets:
                observations.append(f"Found {len(tickets)} tickets")
                for i, ticket in enumerate(tickets[:5]):  # Limit to first 5
                    if isinstance(ticket, dict):
                        ticket_info = []
                        if "id" in ticket:
                            ticket_info.append(f"ID: {ticket['id']}")
                        if "title" in ticket:
                            ticket_info.append(f"Title: {ticket['title']}")
                        if "description" in ticket:
                            ticket_info.append(f"Desc: {ticket['description']}")
                        if "status" in ticket:
                            ticket_info.append(f"Status: {ticket['status']}")
                        if ticket_info:
                            observations.append(f"Ticket {i+1}: {' | '.join(ticket_info)}")
        
        # Set result and context
        observation_dict["result"] = f"Operation {tool_name} completed"
        if observations:
            observation_dict["context"] = " | ".join(observations)
        else:
            observation_dict["context"] = f"Operation {tool_name} completed successfully"
        
        return json.dumps(convert_mpf_to_float(observation_dict))
    
    @staticmethod
    def format_math_observation(result: Dict[str, Any], tool_name: str) -> str:
        """
        Format MathAPI operation observations with consistent schema.
        
        MathAPI is stateless, so all observations include:
        - result (the calculation result)
        - context (human-readable description)
        
        Args:
            result: Result from math operation
            tool_name: Name of the tool that was executed
            
        Returns:
            Formatted observation string with consistent keys
        """
        # Start with consistent schema (stateless API)
        observation_dict = {
            "result": None,
            "context": None
        }
        
        # Handle error cases
        if "error" in result:
            observation_dict["result"] = result.get("error")
            observation_dict["context"] = f"Error in {tool_name}: {result.get('error')}"
            return json.dumps(convert_mpf_to_float(observation_dict))
        
        # Extract result value
        if "result" in result:
            observation_dict["result"] = result["result"]
        elif "mean" in result:
            observation_dict["result"] = result["mean"]
        elif "std_dev" in result:
            observation_dict["result"] = result["std_dev"]
        elif "log_value" in result:
            observation_dict["result"] = result["log_value"]
        
        # Build context
        observations = []
        if tool_name == "mean":
            observations.append(f"Mean calculated: {observation_dict['result']}")
        elif tool_name == "standard_deviation":
            observations.append(f"Standard deviation: {observation_dict['result']}")
        elif tool_name == "logarithm":
            observations.append(f"Logarithm calculated: {observation_dict['result']}")
        else:
            observations.append(f"Calculation result: {observation_dict['result']}")
        
        observation_dict["context"] = " | ".join(observations) if observations else f"Operation {tool_name} completed"
        
        return json.dumps(convert_mpf_to_float(observation_dict))
    
    @staticmethod
    def format_trading_observation(
        result: Dict[str, Any],
        tool_name: str,
        api_instance: Optional[Any] = None,
        arguments: Optional[Dict[str, Any]] = None
    ) -> str:
        """
        Format TradingBot operation observations with consistent state.
        
        All TradingBot observations include the same state keys for RL consistency:
        - authenticated
        - account_balance
        - account_id
        - market_status
        - total_orders
        - watch_list_count
        - transaction_count
        - result
        - context
        
        Args:
            result: Result from trading operation
            tool_name: Name of the tool that was executed
            api_instance: Optional API instance for state queries
            arguments: Optional arguments passed to the tool
            
        Returns:
            Formatted observation string with consistent keys
        """
        if arguments is None:
            arguments = {}
        
        # Start with consistent state schema
        if api_instance:
            observation_dict = ObservationGenerator._get_trading_state_info(api_instance)
        else:
            observation_dict = ObservationGenerator._get_trading_state_info(None)
        
        # Handle error cases
        if "error" in result:
            observation_dict["result"] = result.get("error")
            observation_dict["context"] = f"Error in {tool_name}: {result.get('error')}"
            return json.dumps(convert_mpf_to_float(observation_dict))
        
        # RC-GRPO FIX: Add specific handlers for discovery tools
        observations = []
        
        if tool_name == "get_stock_info":
            # Include all stock details - these are discoverable
            if "price" in result:
                observations.append(f"Price: {result['price']}")
            if "percent_change" in result:
                observations.append(f"Change: {result['percent_change']}%")
            if "volume" in result:
                observations.append(f"Volume: {result['volume']}")
            if "symbol" in result:
                observations.append(f"Symbol: {result['symbol']}")
        
        elif tool_name == "get_transaction_history":
            # Result is typically a list of transactions
            if "transaction_history" in result:
                transactions = result["transaction_history"]
            elif isinstance(result, list):
                transactions = result
            else:
                transactions = []
            
            if transactions:
                observations.append(f"Found {len(transactions)} transactions")
                for i, tx in enumerate(transactions[:5]):  # Limit to first 5
                    if isinstance(tx, dict):
                        tx_info = []
                        if "type" in tx:
                            tx_info.append(f"Type: {tx['type']}")
                        if "symbol" in tx:
                            tx_info.append(f"Symbol: {tx['symbol']}")
                        if "amount" in tx:
                            tx_info.append(f"Amount: {tx['amount']}")
                        if "price" in tx:
                            tx_info.append(f"Price: {tx['price']}")
                        if tx_info:
                            observations.append(f"TX {i+1}: {' | '.join(tx_info)}")
        
        elif tool_name == "get_account_info":
            # Include balance and other account details
            if "balance" in result:
                observations.append(f"Balance: {result['balance']}")
            if "account_id" in result:
                observations.append(f"Account ID: {result['account_id']}")
            if "buying_power" in result:
                observations.append(f"Buying Power: {result['buying_power']}")
        
        elif tool_name == "get_order_history":
            if "order_history" in result:
                orders = result["order_history"]
            elif isinstance(result, list):
                orders = result
            else:
                orders = []
            
            if orders:
                observations.append(f"Found {len(orders)} orders")
                for i, order in enumerate(orders[:5]):
                    if isinstance(order, dict):
                        order_info = []
                        if "order_id" in order:
                            order_info.append(f"ID: {order['order_id']}")
                        if "symbol" in order:
                            order_info.append(f"Symbol: {order['symbol']}")
                        if "status" in order:
                            order_info.append(f"Status: {order['status']}")
                        if order_info:
                            observations.append(f"Order {i+1}: {' | '.join(order_info)}")
        
        else:
            # Generic handler for other trading tools - include primitive values
            for key, value in result.items():
                if isinstance(value, (str, int, float, bool)) and value is not None and key != "error":
                    observations.append(f"{key}: {value}")
        
        # Set result and context
        observation_dict["result"] = f"Operation {tool_name} completed"
        if observations:
            observation_dict["context"] = " | ".join(observations[:10])  # Increase limit for lists
        else:
            observation_dict["context"] = f"Operation {tool_name} completed successfully"
        
        return json.dumps(convert_mpf_to_float(observation_dict))
    
    @staticmethod
    def format_travel_observation(
        result: Dict[str, Any],
        tool_name: str,
        api_instance: Optional[Any] = None,
        arguments: Optional[Dict[str, Any]] = None
    ) -> str:
        """
        Format TravelAPI operation observations with consistent state.
        
        All TravelAPI observations include the same state keys for RL consistency:
        - authenticated
        - total_bookings
        - total_credit_cards
        - budget_limit
        - user_first_name
        - user_last_name
        - result
        - context
        
        Args:
            result: Result from travel operation
            tool_name: Name of the tool that was executed
            api_instance: Optional API instance for state queries
            arguments: Optional arguments passed to the tool
            
        Returns:
            Formatted observation string with consistent keys
        """
        if arguments is None:
            arguments = {}
        
        # Start with consistent state schema
        if api_instance:
            observation_dict = ObservationGenerator._get_travel_state_info(api_instance)
        else:
            observation_dict = ObservationGenerator._get_travel_state_info(None)
        
        # Handle error cases
        if "error" in result:
            observation_dict["result"] = result.get("error")
            observation_dict["context"] = f"Error in {tool_name}: {result.get('error')}"
            return json.dumps(convert_mpf_to_float(observation_dict))
        
        # RC-GRPO FIX: Add specific handlers for discovery tools
        observations = []
        
        if tool_name == "get_credit_card_balance":
            # Include balance - critical for discovery
            if "card_balance" in result:
                observations.append(f"Balance: {result['card_balance']}")
            if "balance" in result:
                observations.append(f"Balance: {result['balance']}")
        
        elif tool_name == "get_booking_history":
            # Handle booking history - can be dict (keyed by booking_id) or list
            if "booking_history" in result:
                bookings_data = result["booking_history"]
            elif isinstance(result, (list, dict)):
                bookings_data = result
            else:
                bookings_data = {}
            
            # Convert dict to list format for processing
            if isinstance(bookings_data, dict):
                bookings = [(bid, bdata) for bid, bdata in list(bookings_data.items())[:5]]
                observations.append(f"Found {len(bookings_data)} bookings")
                for booking_id, booking in bookings:
                    if isinstance(booking, dict):
                        booking_info = [f"ID: {booking_id}"]
                        if "flight_number" in booking:
                            booking_info.append(f"Flight: {booking['flight_number']}")
                        if "travel_cost" in booking:
                            booking_info.append(f"Cost: {booking['travel_cost']}")
                        if "travel_from" in booking:
                            booking_info.append(f"From: {booking['travel_from']}")
                        if "travel_to" in booking:
                            booking_info.append(f"To: {booking['travel_to']}")
                        observations.append(f"Booking: {' | '.join(booking_info)}")
            elif isinstance(bookings_data, list) and bookings_data:
                observations.append(f"Found {len(bookings_data)} bookings")
                for i, booking in enumerate(bookings_data[:5]):
                    if isinstance(booking, dict):
                        booking_info = []
                        if "booking_id" in booking:
                            booking_info.append(f"ID: {booking['booking_id']}")
                        if "flight_number" in booking:
                            booking_info.append(f"Flight: {booking['flight_number']}")
                        if "cost" in booking:
                            booking_info.append(f"Cost: {booking['cost']}")
                        if "travel_cost" in booking:
                            booking_info.append(f"Cost: {booking['travel_cost']}")
                        if booking_info:
                            observations.append(f"Booking {i+1}: {' | '.join(booking_info)}")
        
        elif tool_name == "retrieve_invoice":
            # Result may have nested "invoice" key
            invoice = result.get("invoice", result)
            if isinstance(invoice, dict):
                if "booking_id" in invoice:
                    observations.append(f"Booking ID: {invoice['booking_id']}")
                if "travel_cost" in invoice:
                    observations.append(f"Cost: {invoice['travel_cost']}")
                if "travel_from" in invoice:
                    observations.append(f"From: {invoice['travel_from']}")
                if "travel_to" in invoice:
                    observations.append(f"To: {invoice['travel_to']}")
                if "travel_class" in invoice:
                    observations.append(f"Class: {invoice['travel_class']}")
                if "transaction_id" in invoice:
                    observations.append(f"Transaction: {invoice['transaction_id']}")
        
        else:
            # Generic handler for other travel tools - include primitive values
            for key, value in result.items():
                if isinstance(value, (str, int, float, bool)) and value is not None and key != "error":
                    observations.append(f"{key}: {value}")
        
        # Set result and context
        observation_dict["result"] = f"Operation {tool_name} completed"
        if observations:
            observation_dict["context"] = " | ".join(observations[:10])  # Increase limit
        else:
            observation_dict["context"] = f"Operation {tool_name} completed successfully"
        
        return json.dumps(convert_mpf_to_float(observation_dict))
    
    @staticmethod
    def format_web_search_observation(
        result: Dict[str, Any],
        tool_name: str,
        api_instance: Optional[Any] = None,
        arguments: Optional[Dict[str, Any]] = None
    ) -> str:
        """
        Format WebSearchAPI operation observations with consistent schema.
        
        WebSearchAPI is stateless, so all observations include:
        - result (search results or content)
        - context (human-readable description)
        
        Args:
            result: Result from web search operation
            tool_name: Name of the tool that was executed
            api_instance: Optional API instance for state queries
            arguments: Optional arguments passed to the tool
            
        Returns:
            Formatted observation string with consistent keys
        """
        if arguments is None:
            arguments = {}
        
        # Start with consistent schema (stateless API)
        observation_dict = {
            "result": None,
            "context": None
        }
        
        # Handle error cases
        if "error" in result:
            observation_dict["result"] = result.get("error")
            observation_dict["context"] = f"Error in {tool_name}: {result.get('error')}"
            return json.dumps(convert_mpf_to_float(observation_dict))
        
        # Build context
        observations = []
        
        if tool_name == "search_engine_query":
            if isinstance(result, list):
                count = len(result)
                observation_dict["result"] = f"Found {count} search results"
                observations.append(f"Retrieved {count} search results")
            elif isinstance(result, dict) and "error" not in result:
                observation_dict["result"] = "Search completed"
                observations.append("Search query executed")
        
        elif tool_name == "fetch_url_content":
            if "content" in result:
                content = result["content"]
                content_length = len(content) if isinstance(content, str) else 0
                observation_dict["result"] = f"Content retrieved ({content_length} characters)"
                observations.append(f"Fetched content: {content_length} characters")
        
        observation_dict["context"] = " | ".join(observations) if observations else f"Operation {tool_name} completed"
        
        return json.dumps(convert_mpf_to_float(observation_dict))
    
    @staticmethod
    def format_memory_observation(
        result: Dict[str, Any],
        tool_name: str,
        api_instance: Optional[Any] = None,
        arguments: Optional[Dict[str, Any]] = None
    ) -> str:
        """
        Format MemoryAPI operation observations with consistent state.
        
        All MemoryAPI observations include the same state keys for RL consistency.
        The exact keys depend on the variant (kv, vector, rec_sum).
        
        Args:
            result: Result from memory operation
            tool_name: Name of the tool that was executed
            api_instance: Optional API instance for state queries
            arguments: Optional arguments passed to the tool
            
        Returns:
            Formatted observation string with consistent keys
        """
        if arguments is None:
            arguments = {}
        
        # Start with consistent state schema
        if api_instance:
            observation_dict = ObservationGenerator._get_memory_state_info(api_instance)
        else:
            observation_dict = ObservationGenerator._get_memory_state_info(None)
        
        # Handle error cases
        if "error" in result:
            observation_dict["result"] = result.get("error")
            observation_dict["context"] = f"Error in {tool_name}: {result.get('error')}"
            return json.dumps(convert_mpf_to_float(observation_dict))
        
        # Build context from result
        observations = []
        for key, value in result.items():
            if isinstance(value, (str, int, float, bool)) and value is not None and key != "error":
                observations.append(f"{key}: {value}")
            elif isinstance(value, (list, dict)) and key != "error":
                # Summarize complex structures
                if isinstance(value, list):
                    observations.append(f"{key}: {len(value)} items")
                elif isinstance(value, dict):
                    observations.append(f"{key}: {len(value)} entries")
        
        # Set result and context
        observation_dict["result"] = f"Operation {tool_name} completed"
        if observations:
            observation_dict["context"] = " | ".join(observations[:5])  # Limit to avoid very long strings
        else:
            observation_dict["context"] = f"Operation {tool_name} completed successfully"
        
        return json.dumps(convert_mpf_to_float(observation_dict))
    
    @staticmethod
    def format_generic_observation(result: Dict[str, Any], tool_name: str) -> str:
        """
        Format generic API operation observations.
        
        Args:
            result: Result from API operation
            tool_name: Name of the tool that was executed
            
        Returns:
            Formatted observation string
        """
        # Just return the result as JSON
        return json.dumps(convert_mpf_to_float(result))
    
    def generate_observation(
        self,
        result: Union[Dict[str, Any], str],
        tool_name: str,
        api_class: Optional[str] = None,
        api_instance: Optional[Any] = None,
        arguments: Optional[Dict[str, Any]] = None
    ) -> str:
        """
        Generate observation from API result.
        
        Args:
            result: Result from API call
            tool_name: Name of the tool that was executed
            api_class: Optional API class name for routing
            api_instance: Optional API instance for state queries
            arguments: Optional arguments passed to the tool
            
        Returns:
            Formatted observation string (JSON)
        """
        # If result is already a string, return it
        if isinstance(result, str):
            self.last_observation = result
            return result
        
        # RC-GRPO FIX: Handle list results properly (e.g., get_user_tweets, get_user_tickets)
        # Don't stringify - wrap in a dict for the formatter to handle
        if isinstance(result, list):
            result = {"_list_result": result}
        
        # If result is not a dict (and not handled above), convert it
        if not isinstance(result, dict):
            result = {"result": str(result)}
        
        # Route to appropriate formatter based on API class or tool name
        if api_class:
            if "FileSystem" in api_class:
                observation = self.format_file_system_observation(result, tool_name, api_instance, arguments)
            elif "VehicleControl" in api_class:
                observation = self.format_vehicle_control_observation(result, tool_name, api_instance, arguments)
            elif "Twitter" in api_class:
                observation = self.format_twitter_observation(result, tool_name, api_instance)
            elif "Message" in api_class:
                observation = self.format_message_observation(result, tool_name, api_instance)
            elif "Ticket" in api_class:
                observation = self.format_ticket_observation(result, tool_name, api_instance)
            elif "Math" in api_class:
                observation = self.format_math_observation(result, tool_name)
            elif "TradingBot" in api_class or "Trading" in api_class:
                observation = self.format_trading_observation(result, tool_name, api_instance, arguments)
            elif "Travel" in api_class:
                observation = self.format_travel_observation(result, tool_name, api_instance, arguments)
            elif "WebSearch" in api_class:
                observation = self.format_web_search_observation(result, tool_name, api_instance, arguments)
            elif "Memory" in api_class:
                observation = self.format_memory_observation(result, tool_name, api_instance, arguments)
            else:
                observation = self.format_generic_observation(result, tool_name)
        else:
            # Try to infer from tool name
            fs_tools = ["cd", "ls", "cat", "grep", "find", "mkdir", "touch", "rm", 
                       "rmdir", "mv", "cp", "wc", "tail", "sort", "diff", "du", "echo", "pwd"]
            vehicle_tools = ["activateParkingBrake", "adjustClimateControl", "check_tire_pressure", 
                            "displayCarStatus", "display_log", "estimate_distance", 
                            "estimate_drive_feasibility_by_mileage", "fillFuelTank", 
                            "find_nearest_tire_shop", "gallon_to_liter", "get_current_speed", 
                            "get_outside_temperature_from_google", "get_outside_temperature_from_weather_com",
                            "get_zipcode_based_on_city", "liter_to_gallon", "lockDoors", 
                            "pressBrakePedal", "releaseBrakePedal", "setCruiseControl", 
                            "setHeadlights", "set_navigation", "startEngine"]
            twitter_tools = ["post_tweet", "retweet", "comment", "authenticate_twitter", 
                            "posting_get_login_status", "get_tweet", "get_user_tweets", 
                            "search_tweets", "get_tweet_comments", "get_user_stats", 
                            "follow_user", "unfollow_user", "list_all_following", "mention"]
            message_tools = ["send_message", "view_messages_sent", "view_messages_received", 
                            "delete_message", "message_login", "message_get_login_status",
                            "list_users", "get_user_id", "add_contact", "search_messages",
                            "get_message_stats"]
            ticket_tools = ["create_ticket", "get_ticket", "resolve_ticket", "close_ticket", 
                           "edit_ticket", "ticket_login", "ticket_get_login_status", "logout",
                           "get_user_tickets"]
            math_tools = ["mean", "standard_deviation", "logarithm", "si_unit_conversion",
                         "imperial_si_conversion", "add", "subtract", "multiply", "divide",
                         "power", "square_root", "absolute_value", "round_number", "percentage",
                         "min_value", "max_value", "sum_values"]
            trading_tools = ["get_current_time", "get_symbol_by_name", "get_stock_info",
                           "get_order_details", "cancel_order", "place_order", "withdraw_funds",
                           "get_account_info", "trading_login", "trading_get_login_status",
                           "trading_logout", "fund_account", "remove_stock_from_watchlist",
                           "get_watchlist", "get_order_history", "get_transaction_history",
                           "get_available_stocks", "filter_stocks_by_price", "add_to_watchlist",
                           "notify_price_change"]
            travel_tools = ["authenticate_travel", "travel_get_login_status", "get_budget_fiscal_year",
                          "register_credit_card", "get_flight_cost", "get_credit_card_balance",
                          "book_flight", "retrieve_invoice", "get_booking_history", "list_all_airports",
                          "cancel_booking", "compute_exchange_rate", "verify_traveler_information",
                          "set_budget_limit", "get_nearest_airport_by_city", "purchase_insurance",
                          "contact_customer_support", "get_all_credit_cards"]
            web_search_tools = ["search_engine_query", "fetch_url_content"]
            memory_tools = ["core_memory_add", "core_memory_remove", "core_memory_replace",
                          "core_memory_clear", "core_memory_retrieve", "core_memory_list_keys",
                          "core_memory_key_search", "core_memory_retrieve_all",
                          "archival_memory_add", "archival_memory_remove", "archival_memory_replace",
                          "archival_memory_clear", "archival_memory_retrieve", "archival_memory_list_keys",
                          "archival_memory_key_search", "core_memory_update", "archival_memory_update",
                          "core_memory_retrieve", "archival_memory_retrieve", "memory_append",
                          "memory_update", "memory_clear", "memory_replace", "memory_retrieve"]
            
            if tool_name in fs_tools:
                observation = self.format_file_system_observation(result, tool_name, api_instance, arguments)
            elif tool_name in vehicle_tools:
                observation = self.format_vehicle_control_observation(result, tool_name, api_instance, arguments)
            elif tool_name in twitter_tools:
                observation = self.format_twitter_observation(result, tool_name, api_instance)
            elif tool_name in message_tools:
                observation = self.format_message_observation(result, tool_name, api_instance)
            elif tool_name in ticket_tools:
                observation = self.format_ticket_observation(result, tool_name, api_instance)
            elif tool_name in math_tools:
                observation = self.format_math_observation(result, tool_name)
            elif tool_name in trading_tools:
                observation = self.format_trading_observation(result, tool_name, api_instance, arguments)
            elif tool_name in travel_tools:
                observation = self.format_travel_observation(result, tool_name, api_instance, arguments)
            elif tool_name in web_search_tools:
                observation = self.format_web_search_observation(result, tool_name, api_instance, arguments)
            elif tool_name in memory_tools:
                observation = self.format_memory_observation(result, tool_name, api_instance, arguments)
            else:
                observation = self.format_generic_observation(result, tool_name)
        
        self.last_observation = observation
        return observation
    
    def get_last_observation(self) -> Optional[str]:
        """Get the last generated observation."""
        return self.last_observation
