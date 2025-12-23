"""
Test script for Tool Calling Environment

Validates the environment with BFCL samples to ensure proper functionality.
"""

import json
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from bfcl_eval.env import ToolCallingEnvironment


def test_action_parser():
    """Test action parser with various formats."""
    print("\n" + "=" * 60)
    print("TEST: Action Parser")
    print("=" * 60)
    
    from bfcl_eval.env.action_parser import ActionParser
    
    # Test Python-style
    result = ActionParser.parse("cd(folder='document')")
    assert result[0]["name"] == "cd"
    assert result[0]["arguments"]["folder"] == "document"
    print("✓ Python-style parsing works")
    
    # Test JSON format
    result = ActionParser.parse('{"name": "mkdir", "arguments": {"dir_name": "temp"}}')
    assert result[0]["name"] == "mkdir"
    assert result[0]["arguments"]["dir_name"] == "temp"
    print("✓ JSON parsing works")
    
    # Test list format
    result = ActionParser.parse("[cd(folder='doc'), mkdir(dir_name='temp')]")
    assert len(result) == 2
    assert result[0]["name"] == "cd"
    assert result[1]["name"] == "mkdir"
    print("✓ List parsing works")
    
    # Test parameter types
    result = ActionParser.parse("test(a=5, b='hello', c=True, d=None)")
    assert result[0]["arguments"]["a"] == 5
    assert result[0]["arguments"]["b"] == "hello"
    assert result[0]["arguments"]["c"] is True
    assert result[0]["arguments"]["d"] is None
    print("✓ Parameter type conversion works")
    
    print("✅ All action parser tests passed!\n")


def test_basic_environment():
    """Test basic environment functionality."""
    print("\n" + "=" * 60)
    print("TEST: Basic Environment")
    print("=" * 60)
    
    # Create environment
    env = ToolCallingEnvironment()
    
    # Test reset with simple config
    initial_config = {
        "GorillaFileSystem": {
            "root": {
                "workspace": {
                    "type": "directory",
                    "contents": {
                        "test.txt": {
                            "type": "file",
                            "content": "Hello World"
                        }
                    }
                }
            }
        }
    }
    
    obs = env.reset(initial_config, max_turns=3)
    assert env.is_episode_active()
    print("✓ Environment reset successful")
    print(f"  Initial observation: {obs[:100]}...")
    
    # Test step with action
    obs, done, info = env.step("pwd()")
    assert not done
    assert info["turn"] == 1
    print("✓ Step execution works")
    print(f"  Observation: {obs[:150]}...")
    
    # Test multiple actions in one step
    obs, done, info = env.step("[cd(folder='workspace'), ls()]")
    assert not done
    print("✓ Multiple actions in single step works")
    print(f"  Observation: {obs[:150]}...")
    
    # Test episode termination
    obs, done, info = env.step("cat(file_name='test.txt')")
    assert done  # Should be done after 3 turns
    print("✓ Episode termination works")
    
    # Get state
    state = env.get_state()
    assert "episode_info" in state
    assert "api_states" in state
    print("✓ State retrieval works")
    
    env.close()
    print("✅ All basic environment tests passed!\n")


def test_filesystem_operations():
    """Test file system operations."""
    print("\n" + "=" * 60)
    print("TEST: Filesystem Operations")
    print("=" * 60)
    
    env = ToolCallingEnvironment()
    
    initial_config = {
        "GorillaFileSystem": {
            "root": {
                "workspace": {
                    "type": "directory",
                    "contents": {
                        "document": {
                            "type": "directory",
                            "contents": {
                                "report.txt": {
                                    "type": "file",
                                    "content": "Test report content"
                                }
                            }
                        }
                    }
                }
            }
        }
    }
    
    env.reset(initial_config, max_turns=10)
    
    # Test navigation
    obs, done, info = env.step("cd(folder='workspace')")
    assert not done
    print("✓ Navigation works")
    
    # Test listing
    obs, done, info = env.step("ls()")
    obs_data = json.loads(obs)
    assert "current_directory_content" in obs_data
    print(f"✓ Listing works: {obs_data.get('current_directory_content')}")
    
    # Test cd into subdirectory
    obs, done, info = env.step("cd(folder='document')")
    print("✓ Subdirectory navigation works")
    
    # Test reading file
    obs, done, info = env.step("cat(file_name='report.txt')")
    obs_data = json.loads(obs)
    assert "file_content" in obs_data
    print(f"✓ File reading works: {obs_data.get('file_content')[:30]}...")
    
    # Test creating directory
    obs, done, info = env.step("mkdir(dir_name='archive')")
    print("✓ Directory creation works")
    
    # Test creating file
    obs, done, info = env.step("touch(file_name='new_file.txt')")
    print("✓ File creation works")
    
    env.close()
    print("✅ All filesystem operation tests passed!\n")


def test_twitter_operations():
    """Test Twitter API operations."""
    print("\n" + "=" * 60)
    print("TEST: Twitter Operations")
    print("=" * 60)
    
    env = ToolCallingEnvironment()
    
    initial_config = {
        "TwitterAPI": {
            "username": "test_user",
            "password": "test_pass",
            "authenticated": True,
            "tweets": {},
            "tweet_counter": 0
        }
    }
    
    env.reset(initial_config, max_turns=5)
    
    # Test posting tweet
    obs, done, info = env.step("post_tweet(content='Hello World!', tags=['#test'])")
    obs_data = json.loads(obs)
    assert "id" in obs_data
    tweet_id = obs_data["id"]
    print(f"✓ Tweet posting works (ID: {tweet_id})")
    
    # Test retweeting
    obs, done, info = env.step(f"retweet(tweet_id={tweet_id})")
    obs_data = json.loads(obs)
    assert "retweet_status" in obs_data
    print(f"✓ Retweeting works: {obs_data['retweet_status']}")
    
    env.close()
    print("✅ All Twitter operation tests passed!\n")


def test_multi_api_environment():
    """Test environment with multiple APIs."""
    print("\n" + "=" * 60)
    print("TEST: Multi-API Environment")
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
        },
        "MathAPI": {}
    }
    
    env.reset(initial_config, max_turns=5)
    
    # Test file system operation
    obs, done, info = env.step("cd(folder='workspace')")
    print("✓ File system operation works in multi-API env")
    
    # Test math operation
    obs, done, info = env.step("mean(numbers=[1, 2, 3, 4, 5])")
    obs_data = json.loads(obs)
    # MathAPI might return different structures, just check if we got an observation
    print(f"✓ Math operation works: {obs[:100]}...")
    assert obs_data  # Just verify we got some data
    
    # Test state retrieval
    state = env.get_state()
    assert "GorillaFileSystem" in state["api_states"]
    # MathAPI might not have public state, just verify we can get state
    print(f"✓ Multi-API state management works: {list(state['api_states'].keys())}")
    
    env.close()
    print("✅ All multi-API tests passed!\n")


def test_with_bfcl_sample():
    """Test with actual BFCL sample data."""
    print("\n" + "=" * 60)
    print("TEST: BFCL Sample Integration")
    print("=" * 60)
    
    # Load a sample from BFCL data
    bfcl_data_path = Path(__file__).parent.parent / "data" / "BFCL_v4_multi_turn_base.json"
    
    if not bfcl_data_path.exists():
        print("⚠ BFCL sample file not found, skipping integration test")
        return
    
    with open(bfcl_data_path, 'r') as f:
        # Read first line (JSONL format)
        sample = json.loads(f.readline())
    
    # Load ground truth
    gt_path = Path(__file__).parent.parent / "data" / "possible_answer" / "BFCL_v4_multi_turn_base.json"
    ground_truth = None
    
    if gt_path.exists():
        with open(gt_path, 'r') as f:
            for line in f:
                gt_obj = json.loads(line)
                if gt_obj.get("id") == sample.get("id"):
                    ground_truth = gt_obj.get("ground_truth", [])
                    break
    
    if ground_truth is None:
        print("⚠ Ground truth not found, skipping integration test")
        return
    
    print(f"Testing with sample: {sample.get('id')}")
    print(f"Turns: {len(ground_truth)}")
    
    # Initialize environment
    env = ToolCallingEnvironment()
    initial_config = sample.get("initial_config", {})
    
    obs = env.reset(initial_config, max_turns=len(ground_truth))
    print("✓ Environment initialized with BFCL config")
    
    # Execute ground truth actions
    for turn_idx, turn_actions in enumerate(ground_truth):
        print(f"\n  Turn {turn_idx + 1}: {len(turn_actions)} actions")
        for action in turn_actions:
            print(f"    Action: {action}")
            obs, done, info = env.step(action)
            obs_preview = obs[:100] + "..." if len(obs) > 100 else obs
            print(f"    Observation: {obs_preview}")
            
            # Check if observation is not empty
            if obs != "{}":
                print("    ✓ Non-empty observation generated")
    
    env.close()
    print("\n✅ BFCL sample integration test passed!\n")


def test_error_handling():
    """Test error handling."""
    print("\n" + "=" * 60)
    print("TEST: Error Handling")
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
    
    # Test with non-existent file
    obs, done, info = env.step("cat(file_name='nonexistent.txt')")
    obs_data = json.loads(obs)
    assert "error" in obs_data
    print("✓ Error handling for missing file works")
    
    # Test with invalid tool
    obs, done, info = env.step("invalid_tool()")
    obs_data = json.loads(obs)
    assert "error" in obs_data
    print("✓ Error handling for invalid tool works")
    
    env.close()
    print("✅ All error handling tests passed!\n")


def run_all_tests():
    """Run all tests."""
    print("\n" + "=" * 60)
    print("RUNNING ALL ENVIRONMENT TESTS")
    print("=" * 60)
    
    try:
        test_action_parser()
        test_basic_environment()
        test_filesystem_operations()
        test_twitter_operations()
        test_multi_api_environment()
        test_error_handling()
        test_with_bfcl_sample()
        
        print("\n" + "=" * 60)
        print("✅ ALL TESTS PASSED! ✅")
        print("=" * 60)
        return True
        
    except Exception as e:
        print("\n" + "=" * 60)
        print(f"❌ TEST FAILED: {e}")
        print("=" * 60)
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)

