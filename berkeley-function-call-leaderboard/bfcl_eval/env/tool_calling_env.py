"""
Tool Calling Environment

Main environment class that combines all components to provide a verl-compatible
interface for tool calling with observation generation.
"""

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

# RC-GRPO: Try package imports first (for bfcl_eval.env), fallback to direct imports
try:
    from bfcl_eval.env.action_parser import ActionParser
    from bfcl_eval.env.api_manager import APIManager
    from bfcl_eval.env.observation_generator import ObservationGenerator
    from bfcl_eval.env.episode_manager import EpisodeManager
except ImportError:
    # Fallback for direct imports (via sys.path)
    from action_parser import ActionParser
    from api_manager import APIManager
    from observation_generator import ObservationGenerator
    from episode_manager import EpisodeManager


class ToolCallingEnvironment:
    """
    Virtual interactive environment for tool calling.
    
    This environment provides a verl-compatible interface for executing tool calls
    and generating observations. It maintains state across turns within an episode
    and resets between episodes.
    """

    def __init__(self, func_doc_dir: Optional[Path] = None):
        """
        Initialize the tool calling environment.
        
        Args:
            func_doc_dir: Optional path to multi_turn_func_doc directory
        """
        self.action_parser = ActionParser()
        self.api_manager = APIManager(func_doc_dir=func_doc_dir)
        self.observation_generator = ObservationGenerator()
        self.episode_manager = EpisodeManager()
        
        self.current_observation = None
        self.episode_count = 0
    
    def reset(
        self,
        initial_config: Dict[str, Any],
        episode_id: Optional[str] = None,
        max_turns: Optional[int] = None
    ) -> str:
        """
        Reset environment for a new episode.
        
        Args:
            initial_config: Initial configuration mapping class names to configs
            episode_id: Optional episode identifier
            max_turns: Optional maximum number of turns
            
        Returns:
            Initial observation string
        """
        # Generate episode ID if not provided
        if episode_id is None:
            self.episode_count += 1
            episode_id = f"episode_{self.episode_count}"
        
        # Extract involved classes from initial_config
        involved_classes = list(initial_config.keys())
        
        # Load tool documentation
        self.api_manager.load_tool_docs(involved_classes)
        
        # Initialize API instances with configuration
        self.api_manager.initialize_apis(initial_config)
        
        # Start episode
        self.episode_manager.start_episode(
            episode_id=episode_id,
            initial_config=initial_config,
            max_turns=max_turns
        )
        
        # Generate initial observation
        initial_obs = {
            "status": "ready",
            "episode_id": episode_id,
            "turn": 0,
            "available_apis": involved_classes,
            "message": "Environment initialized and ready for actions"
        }
        
        self.current_observation = self.observation_generator.generate_observation(
            result=initial_obs,
            tool_name="reset",
            api_class=None
        )
        
        return self.current_observation
    
    def step(
        self,
        action: Union[str, Dict, List]
    ) -> Tuple[str, bool, Dict[str, Any]]:
        """
        Execute an action and return observation.
        
        This is the verl-compatible step function.
        
        Args:
            action: Action in any supported format (Python-style, JSON, or list)
            
        Returns:
            Tuple of (observation, done, info):
            - observation: String observation of the result
            - done: Boolean indicating if episode is complete
            - info: Dictionary with additional information
        """
        if not self.episode_manager.is_active():
            return (
                '{"error": "No active episode. Call reset() first."}',
                True,
                {"error": "no_active_episode"}
            )
        
        # Parse action(s)
        try:
            parsed_actions = self.action_parser.parse(action)
        except Exception as e:
            error_obs = self.observation_generator.generate_observation(
                result={"error": f"Failed to parse action: {str(e)}"},
                tool_name="parse_error"
            )
            return error_obs, False, {"error": "parse_error", "details": str(e)}
        
        # Execute all actions and collect observations
        observations = []
        all_results = []
        
        for parsed_action in parsed_actions:
            tool_name = parsed_action["name"]
            arguments = parsed_action["arguments"]
            
            # Execute tool call (returns tuple of result and api_instance)
            result, api_instance = self.api_manager.execute_tool_call(tool_name, arguments)
            all_results.append(result)
            
            # Generate observation
            # Try to infer API class from tool name
            api_class = None
            for class_name in self.api_manager.api_instances.keys():
                if hasattr(self.api_manager.api_instances[class_name], tool_name):
                    api_class = class_name
                    break
            
            observation = self.observation_generator.generate_observation(
                result=result,
                tool_name=tool_name,
                api_class=api_class,
                api_instance=api_instance,
                arguments=arguments
            )
            observations.append(observation)
        
        # Combine observations if multiple actions
        if len(observations) == 1:
            combined_observation = observations[0]
        else:
            # Create a combined observation for multiple actions
            import json
            combined_result = {
                "multi_action": True,
                "action_count": len(observations),
                "observations": [json.loads(obs) for obs in observations]
            }
            combined_observation = json.dumps(combined_result)
        
        self.current_observation = combined_observation
        
        # Advance turn
        turn_info = self.episode_manager.advance_turn(
            action=str(action),
            observation=combined_observation,
            reward=None  # Reward computed by separate reward manager in verl
        )
        
        # Check if episode is done
        done = turn_info["done"]
        
        # Prepare info dict
        info = {
            "turn": turn_info["turn"],
            "total_turns": turn_info["total_turns"],
            "episode_id": self.episode_manager.episode_id,
            "action_count": len(parsed_actions),
            "results": all_results
        }
        
        return combined_observation, done, info
    
    def get_state(self) -> Dict[str, Any]:
        """
        Get current environment state.
        
        Returns:
            Dictionary containing current state of all APIs
        """
        return {
            "episode_info": self.episode_manager.get_episode_info(),
            "api_states": self.api_manager.get_all_states(),
            "last_observation": self.current_observation
        }
    
    def get_episode_info(self) -> Dict[str, Any]:
        """Get information about current episode."""
        return self.episode_manager.get_episode_info()
    
    def get_turn_history(self) -> List[Dict[str, Any]]:
        """Get history of all turns in current episode."""
        return self.episode_manager.get_turn_history()
    
    def get_available_tools(self) -> Dict[str, Dict[str, Any]]:
        """Get all available tool documentation."""
        return self.api_manager.get_all_tool_docs()
    
    def is_episode_active(self) -> bool:
        """Check if an episode is currently active."""
        return self.episode_manager.is_active()
    
    def end_episode(self) -> Dict[str, Any]:
        """
        Manually end the current episode.
        
        Returns:
            Episode summary
        """
        return self.episode_manager.end_episode()
    
    def close(self) -> None:
        """Clean up resources and end current episode."""
        if self.episode_manager.is_active():
            self.episode_manager.end_episode()
        self.api_manager.reset()
        self.episode_manager.reset()
