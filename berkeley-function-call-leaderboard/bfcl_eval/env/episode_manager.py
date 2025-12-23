"""
Episode Manager for Tool Calling Environment

Manages episode lifecycle: tracks turns, maintains state within episodes,
and handles resets between episodes.
"""

from typing import Any, Dict, List, Optional


class EpisodeManager:
    """Manage episode lifecycle and state."""

    def __init__(self):
        """Initialize episode manager."""
        self.current_turn = 0
        self.max_turns = None
        self.episode_id = None
        self.initial_config = None
        self.episode_active = False
        self.turn_history = []
        
    def start_episode(
        self,
        episode_id: str,
        initial_config: Dict[str, Any],
        max_turns: Optional[int] = None
    ) -> Dict[str, Any]:
        """
        Start a new episode.
        
        Args:
            episode_id: Unique identifier for this episode
            initial_config: Initial configuration for APIs
            max_turns: Maximum number of turns for this episode
            
        Returns:
            Episode info dictionary
        """
        self.episode_id = episode_id
        self.initial_config = initial_config
        self.max_turns = max_turns
        self.current_turn = 0
        self.episode_active = True
        self.turn_history = []
        
        return {
            "episode_id": episode_id,
            "turn": 0,
            "max_turns": max_turns,
            "status": "started"
        }
    
    def advance_turn(
        self,
        action: str,
        observation: str,
        reward: Optional[float] = None
    ) -> Dict[str, Any]:
        """
        Advance to the next turn and record history.
        
        Args:
            action: Action taken in current turn
            observation: Observation received
            reward: Optional reward for this turn
            
        Returns:
            Turn info dictionary
        """
        if not self.episode_active:
            raise RuntimeError("Cannot advance turn: no active episode")
        
        # Record turn history
        turn_info = {
            "turn": self.current_turn,
            "action": action,
            "observation": observation,
        }
        if reward is not None:
            turn_info["reward"] = reward
        
        self.turn_history.append(turn_info)
        
        # Advance turn counter
        self.current_turn += 1
        
        # Check if episode should end
        done = False
        if self.max_turns is not None and self.current_turn >= self.max_turns:
            done = True
            self.episode_active = False
        
        return {
            "turn": self.current_turn,
            "done": done,
            "total_turns": len(self.turn_history)
        }
    
    def end_episode(self) -> Dict[str, Any]:
        """
        End the current episode.
        
        Returns:
            Episode summary dictionary
        """
        if not self.episode_active:
            return {
                "status": "no_active_episode",
                "episode_id": self.episode_id
            }
        
        self.episode_active = False
        
        return {
            "status": "ended",
            "episode_id": self.episode_id,
            "total_turns": len(self.turn_history),
            "final_turn": self.current_turn
        }
    
    def reset(self) -> None:
        """Reset episode manager to initial state."""
        self.current_turn = 0
        self.max_turns = None
        self.episode_id = None
        self.initial_config = None
        self.episode_active = False
        self.turn_history = []
    
    def is_active(self) -> bool:
        """Check if an episode is currently active."""
        return self.episode_active
    
    def get_current_turn(self) -> int:
        """Get the current turn number."""
        return self.current_turn
    
    def get_turn_history(self) -> List[Dict[str, Any]]:
        """Get the history of all turns in current episode."""
        return self.turn_history.copy()
    
    def get_episode_info(self) -> Dict[str, Any]:
        """
        Get information about the current episode.
        
        Returns:
            Episode information dictionary
        """
        return {
            "episode_id": self.episode_id,
            "current_turn": self.current_turn,
            "max_turns": self.max_turns,
            "active": self.episode_active,
            "total_turns_completed": len(self.turn_history),
            "initial_config": self.initial_config
        }
    
    def should_terminate(self) -> bool:
        """
        Check if episode should terminate.
        
        Returns:
            True if episode should end, False otherwise
        """
        if not self.episode_active:
            return True
        
        if self.max_turns is not None and self.current_turn >= self.max_turns:
            return True
        
        return False
    
    def get_initial_config(self) -> Optional[Dict[str, Any]]:
        """Get the initial configuration for current episode."""
        return self.initial_config

