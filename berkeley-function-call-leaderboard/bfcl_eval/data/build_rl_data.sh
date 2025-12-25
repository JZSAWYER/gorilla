#!/bin/bash

# Load machine-specific configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CONFIG_FILE="${SCRIPT_DIR}/config.local.sh"

if [ ! -f "$CONFIG_FILE" ]; then
    echo "Error: config.local.sh not found!"
    echo "Please copy config.local.sh.example to config.local.sh and set your MODEL_ROOT_PATH"
    exit 1
fi

source "$CONFIG_FILE"

# Verify required variables are set
if [ -z "$MODEL_ROOT_PATH" ]; then
    echo "Error: MODEL_ROOT_PATH is not set in config.local.sh"
    exit 1
fi

if [ -z "$INPUT_DIR" ]; then
    echo "Error: INPUT_DIR is not set in config.local.sh"
    exit 1
fi

python build_rl_trajectories.py \
  --input_dir "${INPUT_DIR}/berkeley-function-call-leaderboard/bfcl_eval/data" \
  --output_path ./rl_trajectories.json \
  --expert_model_path "${MODEL_ROOT_PATH}/Qwen2.5-7B-Instruct" \
  --intermediate_model_path "${MODEL_ROOT_PATH}/Qwen2.5-1.5B-Instruct" \
  --weak_model_path "${MODEL_ROOT_PATH}/Qwen2.5-0.5B-Instruct" \
  --num_gpus 8 \
  --trajectories_per_sample 2

# Generate for a specific API with n samples:
# python build_rl_trajectories.py \
#   --input_dir "${INPUT_DIR}/berkeley-function-call-leaderboard/bfcl_eval/data" \
#   --output_path /path/to/output_filesystem.json \
#   --expert_model_path "${MODEL_ROOT_PATH}/Qwen2.5-7B-Instruct" \
#   --intermediate_model_path "${MODEL_ROOT_PATH}/Qwen2.5-1.5B-Instruct" \
#   --weak_model_path "${MODEL_ROOT_PATH}/Qwen2.5-0.5B-Instruct" \
#   --test_mode \
#   --test_api GorillaFileSystem \
#   --samples_per_api_file 5

# full
# python build_rl_trajectories.py \
#   --input_dir "${INPUT_DIR}/berkeley-function-call-leaderboard/bfcl_eval/data" \
#   --output_path /path/to/full_output.json \
#   --expert_model_path "${MODEL_ROOT_PATH}/Qwen2.5-7B-Instruct" \
#   --intermediate_model_path "${MODEL_ROOT_PATH}/Qwen2.5-1.5B-Instruct" \
#   --weak_model_path "${MODEL_ROOT_PATH}/Qwen2.5-0.5B-Instruct"