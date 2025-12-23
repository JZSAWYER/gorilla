"""
Example Usage of Tool Calling Environment

This script demonstrates various usage patterns for the environment.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from bfcl_eval.env import ToolCallingEnvironment


def example_basic_usage():
    """Example 1: Basic environment usage."""
    print("\n" + "=" * 60)
    print("Example 1: Basic Usage")
    print("=" * 60)
    
    # Create environment
    env = ToolCallingEnvironment()
    
    # Define initial configuration
    initial_config = {
        "GorillaFileSystem": {
            "root": {
                "workspace": {
                    "type": "directory",
                    "contents": {
                        "readme.txt": {
                            "type": "file",
                            "content": "Welcome to the project!"
                        },
                        "docs": {
                            "type": "directory",
                            "contents": {}
                        }
                    }
                }
            }
        }
    }
    
    # Reset environment
    obs = env.reset(initial_config, episode_id="example1", max_turns=5)
    print(f"Initial observation: {obs[:100]}...\n")
    
    # Execute some actions
    print("Action 1: Navigate to workspace")
    obs, done, info = env.step("cd(folder='workspace')")
    print(f"Observation: {obs}\n")
    
    print("Action 2: List contents")
    obs, done, info = env.step("ls()")
    print(f"Observation: {obs}\n")
    
    print("Action 3: Read readme file")
    obs, done, info = env.step("cat(file_name='readme.txt')")
    print(f"Observation: {obs}\n")
    
    # Clean up
    env.close()
    print("✓ Example 1 completed\n")


def example_multi_turn_conversation():
    """Example 2: Multi-turn conversation with state persistence."""
    print("\n" + "=" * 60)
    print("Example 2: Multi-Turn Conversation")
    print("=" * 60)
    
    env = ToolCallingEnvironment()
    
    initial_config = {
        "GorillaFileSystem": {
            "root": {
                "project": {
                    "type": "directory",
                    "contents": {}
                }
            }
        }
    }
    
    env.reset(initial_config, episode_id="example2", max_turns=10)
    
    # Simulate a multi-turn conversation
    conversation = [
        ("Navigate to project folder", "cd(folder='project')"),
        ("Create a new directory", "mkdir(dir_name='src')"),
        ("Create a Python file", "touch(file_name='main.py')"),
        ("List everything", "ls()"),
        ("Navigate into src", "cd(folder='src')"),
    ]
    
    for turn, (description, action) in enumerate(conversation, 1):
        print(f"\nTurn {turn}: {description}")
        print(f"Action: {action}")
        obs, done, info = env.step(action)
        print(f"Observation: {obs[:150]}...")
        print(f"Done: {done}, Turn: {info['turn']}")
    
    env.close()
    print("\n✓ Example 2 completed\n")


def example_multiple_actions():
    """Example 3: Multiple actions in one step."""
    print("\n" + "=" * 60)
    print("Example 3: Multiple Actions in One Step")
    print("=" * 60)
    
    env = ToolCallingEnvironment()
    
    initial_config = {
        "GorillaFileSystem": {
            "root": {
                "workspace": {
                    "type": "directory",
                    "contents": {}
                }
            }
        }
    }
    
    env.reset(initial_config, max_turns=3)
    
    # Execute multiple actions at once
    print("Executing multiple actions:")
    actions = "[cd(folder='workspace'), mkdir(dir_name='data'), mkdir(dir_name='models'), ls()]"
    print(f"Actions: {actions}\n")
    
    obs, done, info = env.step(actions)
    print(f"Observation: {obs[:200]}...")
    print(f"Action count: {info['action_count']}")
    
    env.close()
    print("\n✓ Example 3 completed\n")


def example_multi_api():
    """Example 4: Using multiple APIs together."""
    print("\n" + "=" * 60)
    print("Example 4: Multi-API Usage")
    print("=" * 60)
    
    env = ToolCallingEnvironment()
    
    initial_config = {
        "GorillaFileSystem": {
            "root": {
                "reports": {
                    "type": "directory",
                    "contents": {
                        "data.txt": {
                            "type": "file",
                            "content": "1,2,3,4,5"
                        }
                    }
                }
            }
        },
        "TwitterAPI": {
            "username": "data_scientist",
            "password": "secure123",
            "authenticated": True,
            "tweets": {},
            "tweet_counter": 0
        },
        "MathAPI": {}
    }
    
    env.reset(initial_config, max_turns=5)
    
    # File system operation
    print("Step 1: Navigate and read file")
    obs, _, _ = env.step("cd(folder='reports')")
    print(f"Navigation: {obs[:100]}...\n")
    
    obs, _, _ = env.step("cat(file_name='data.txt')")
    print(f"File content: {obs[:100]}...\n")
    
    # Math operation
    print("Step 2: Calculate statistics")
    obs, _, _ = env.step("mean(numbers=[1, 2, 3, 4, 5])")
    print(f"Mean calculation: {obs}\n")
    
    # Twitter operation
    print("Step 3: Post results to Twitter")
    obs, _, _ = env.step("post_tweet(content='Analysis complete!', tags=['#datascience'])")
    print(f"Tweet posted: {obs[:150]}...\n")
    
    env.close()
    print("✓ Example 4 completed\n")


def example_error_handling():
    """Example 5: Error handling."""
    print("\n" + "=" * 60)
    print("Example 5: Error Handling")
    print("=" * 60)
    
    env = ToolCallingEnvironment()
    
    initial_config = {
        "GorillaFileSystem": {
            "root": {
                "workspace": {
                    "type": "directory",
                    "contents": {}
                }
            }
        }
    }
    
    env.reset(initial_config, max_turns=5)
    
    # Try to read non-existent file
    print("Attempting to read non-existent file:")
    obs, _, _ = env.step("cat(file_name='missing.txt')")
    print(f"Error response: {obs}\n")
    
    # Try invalid tool
    print("Attempting to use invalid tool:")
    obs, _, _ = env.step("invalid_tool(param='value')")
    print(f"Error response: {obs}\n")
    
    # Continue with valid operation
    print("Recovering with valid operation:")
    obs, _, _ = env.step("pwd()")
    print(f"Success: {obs}\n")
    
    env.close()
    print("✓ Example 5 completed\n")


def example_state_inspection():
    """Example 6: Inspecting environment state."""
    print("\n" + "=" * 60)
    print("Example 6: State Inspection")
    print("=" * 60)
    
    env = ToolCallingEnvironment()
    
    initial_config = {
        "GorillaFileSystem": {
            "root": {
                "workspace": {
                    "type": "directory",
                    "contents": {}
                }
            }
        }
    }
    
    env.reset(initial_config, max_turns=5)
    
    # Execute some actions
    env.step("cd(folder='workspace')")
    env.step("mkdir(dir_name='test')")
    env.step("touch(file_name='file.txt')")
    
    # Get current state
    print("Current environment state:")
    state = env.get_state()
    
    print(f"\nEpisode Info:")
    for key, value in state['episode_info'].items():
        print(f"  {key}: {value}")
    
    print(f"\nAPI States:")
    for api_name, api_state in state['api_states'].items():
        print(f"  {api_name}: {len(str(api_state))} chars")
    
    # Get turn history
    print(f"\nTurn History:")
    history = env.get_turn_history()
    for i, turn in enumerate(history):
        print(f"  Turn {i}: {turn['action'][:50]}...")
    
    env.close()
    print("\n✓ Example 6 completed\n")


def run_all_examples():
    """Run all examples."""
    print("\n" + "=" * 60)
    print("TOOL CALLING ENVIRONMENT - USAGE EXAMPLES")
    print("=" * 60)
    
    example_basic_usage()
    example_multi_turn_conversation()
    example_multiple_actions()
    example_multi_api()
    example_error_handling()
    example_state_inspection()
    
    print("\n" + "=" * 60)
    print("✅ ALL EXAMPLES COMPLETED")
    print("=" * 60)


if __name__ == "__main__":
    run_all_examples()

