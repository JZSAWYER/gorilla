# Tool Calling Environment

A virtual interactive environment for multi-turn tool calling that generates real observations. Compatible with both SFT dataset generation and verl-based RL training.

## Overview

This environment simulates tool execution across multiple APIs (file systems, social media, messaging, etc.) and generates detailed observations for each action. It maintains state across turns within an episode and resets between episodes, making it suitable for both supervised fine-tuning and reinforcement learning.

## Features

- **Multi-API Support**: GorillaFileSystem, TwitterAPI, MessageAPI, TicketAPI, MathAPI, and more
- **Episode-Based State Management**: State persists across turns within an episode, resets between episodes
- **Flexible Action Parsing**: Supports Python-style, JSON, and list formats
- **Detailed Observations**: Generates rich, contextual observations for each action
- **verl-Compatible**: Standard RL interface (reset, step) for RL training
- **SFT Integration**: Can be used to generate real observations for SFT datasets

## Installation

The environment is part of the BFCL evaluation package. No additional installation required.

```bash
# Ensure you're in the gorilla project root
cd /data1/jzsawyer/projects/gorilla
```

## Quick Start

### Basic Usage

```python
from bfcl_eval.env import ToolCallingEnvironment

# Create environment
env = ToolCallingEnvironment()

# Define initial configuration (BFCL format)
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

# Reset for new episode
observation = env.reset(initial_config, max_turns=5)
print(f"Initial: {observation}")

# Execute actions
obs, done, info = env.step("cd(folder='workspace')")
print(f"Turn 1: {obs}")

obs, done, info = env.step("cat(file_name='test.txt')")
print(f"Turn 2: {obs}")

# Clean up
env.close()
```

### Action Formats

The environment supports multiple action formats:

```python
# 1. Python-style (recommended)
env.step("cd(folder='document')")

# 2. JSON format
env.step('{"name": "cd", "arguments": {"folder": "document"}}')

# 3. List of actions
env.step("[cd(folder='document'), mkdir(dir_name='temp')]")
```

### Multi-API Example

```python
initial_config = {
    "GorillaFileSystem": {
        "root": {
            "workspace": {"type": "directory", "contents": {}}
        }
    },
    "TwitterAPI": {
        "username": "test_user",
        "password": "test_pass",
        "authenticated": True,
        "tweets": {},
        "tweet_counter": 0
    }
}

env.reset(initial_config)

# File system operations
env.step("cd(folder='workspace')")
env.step("touch(file_name='report.txt')")

# Twitter operations
env.step("post_tweet(content='Task completed!', tags=['#automation'])")
```

## Integration with SFT Dataset Generation

The environment is integrated with `build_bfcl_sft.py` to generate real observations:

```bash
# Generate dataset with real observations (default)
python berkeley-function-call-leaderboard/bfcl_eval/data/build_bfcl_sft.py \
    --input_dir berkeley-function-call-leaderboard/bfcl_eval/data \
    --output_path output/bfcl_glaive_with_observations.json

# Generate dataset without real observations (placeholder mode)
python berkeley-function-call-leaderboard/bfcl_eval/data/build_bfcl_sft.py \
    --input_dir berkeley-function-call-leaderboard/bfcl_eval/data \
    --output_path output/bfcl_glaive_no_observations.json \
    --no_real_observations
```

## verl RL Training Integration

The environment follows verl's standard interface:

```python
from bfcl_eval.env import ToolCallingEnvironment

env = ToolCallingEnvironment()

# Training loop
for episode_data in dataset:
    # Reset environment
    obs = env.reset(
        initial_config=episode_data['initial_config'],
        episode_id=episode_data['id'],
        max_turns=len(episode_data['ground_truth'])
    )
    
    for turn in range(max_turns):
        # Policy model predicts action
        action = policy_model.predict(obs)
        
        # Environment executes action
        obs, done, info = env.step(action)
        
        # Reward computed by separate reward manager (not in environment)
        reward = reward_manager.compute_reward(action, obs, info)
        
        # Store transition for training
        buffer.add(obs, action, reward, done, info)
        
        if done:
            break
    
    env.close()
```

### Key Points for verl Integration

1. **No Reward Computation**: The environment doesn't compute rewards - this is handled by verl's separate reward manager
2. **Episode Management**: State persists across turns within one episode, resets between episodes
3. **Observation Format**: Returns observations as JSON strings for compatibility
4. **Info Dict**: Contains additional information (turn number, action count, results)

## API Coverage

The environment supports the following APIs:

- **GorillaFileSystem**: File operations (cd, ls, cat, mkdir, mv, cp, rm, etc.)
- **TwitterAPI**: Social media operations (post_tweet, retweet, comment)
- **MessageAPI**: Messaging operations (send_message, view_messages)
- **TicketAPI**: Ticket management (create_ticket, resolve_ticket)
- **MathAPI**: Mathematical operations (mean, std_dev, logarithm)
- **TradingBot**: Trading operations
- **TravelAPI**: Travel booking operations
- **VehicleControlAPI**: Vehicle control operations
- **MemoryAPI**: Memory operations (key-value, vector, recency summary)
- **WebSearchAPI**: Web search operations

## Testing

Run comprehensive tests:

```bash
python berkeley-function-call-leaderboard/bfcl_eval/env/test_environment.py
```

Tests include:
- Action parser validation
- Basic environment operations
- File system operations
- Twitter operations
- Multi-API environments
- Error handling
- BFCL sample integration

## Architecture

```
ToolCallingEnvironment
├── ActionParser: Parse tool calls in various formats
├── APIManager: Load and route to API instances
├── ObservationGenerator: Generate detailed observations
└── EpisodeManager: Manage episode lifecycle
```

### Component Details

#### ActionParser
- Parses Python-style, JSON, and list formats
- Converts parameter types automatically
- Handles nested structures (lists, dicts)

#### APIManager
- Loads API classes from func_source_code/
- Routes tool calls to appropriate API methods
- Manages API instances and states
- Loads tool documentation

#### ObservationGenerator
- Formats API responses as JSON observations
- Adds contextual information
- Handles errors gracefully
- Compatible with Glaive format

#### EpisodeManager
- Tracks current turn
- Maintains turn history
- Manages episode termination
- Stores episode metadata

## Observation Format

Observations are returned as JSON strings with the following structure:

```json
{
  "result_key": "result_value",
  "context": "Human-readable context about the action"
}
```

For errors:

```json
{
  "error": "Error message describing what went wrong"
}
```

For multiple actions in one step:

```json
{
  "multi_action": true,
  "action_count": 2,
  "observations": [
    {"result1": "value1"},
    {"result2": "value2"}
  ]
}
```

## State Management

- **Within Episode**: State persists across all turns
- **Between Episodes**: State is completely reset
- **API Instances**: Each episode gets fresh API instances
- **Turn History**: Full history maintained for each episode

## Error Handling

The environment handles errors gracefully:

- **Invalid Actions**: Returns error observation
- **Missing Files**: Returns file not found error
- **Authentication Issues**: Returns authentication error
- **Invalid Tool Names**: Returns tool not found error

All errors are returned as observations, not exceptions, to maintain training stability.

## Performance Considerations

- **Lazy Loading**: APIs are only instantiated when needed
- **State Copying**: Episode configurations are deep-copied to prevent mutations
- **Memory Management**: Episodes are cleaned up after closing
- **Error Recovery**: Failed actions don't crash the environment

## Troubleshooting

### Import Errors

If you encounter import errors for specific APIs (e.g., `html2text`), the environment will gracefully skip those APIs and continue with available ones.

### Empty Observations

If observations are empty (`{}`), check:
1. API is properly initialized in `initial_config`
2. Tool name matches documentation
3. Required parameters are provided

### Episode Not Terminating

Ensure `max_turns` is set when calling `reset()`. Without it, episodes run indefinitely.

## Future Enhancements

Potential improvements:
- Async action execution for faster processing
- Observation caching for repeated actions
- State checkpointing for long episodes
- Custom reward integration hooks
- Observation compression for large states

## Contributing

To add a new API:

1. Create API class in `func_source_code/`
2. Add to `API_CLASSES` in `api_manager.py`
3. Add observation formatting in `observation_generator.py`
4. Add tool documentation to `multi_turn_func_doc/`
5. Add tests in `test_environment.py`

## License

Part of the Gorilla project. See main repository for license information.

