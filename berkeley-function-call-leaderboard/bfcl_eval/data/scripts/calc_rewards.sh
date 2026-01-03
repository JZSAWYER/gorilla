#!/bin/bash
# Calculate trajectory rewards by comparing model-generated trajectories against ground truth
#
# Usage: ./calc_rewards.sh [--split-models]
#
# Options:
#   --split-models  Split results by model tier (strong/intermediate/weak) in histogram

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

python3 evaluate_trajectory_rewards.py \
    --reference ../generated/sft/bfcl_glaive_dataset.json \
    --input ../generated/rl/rl_trajectories.json \
    --output ../generated/evaluation/trajectory_reward_results.json \
    --plot ../outputs/plots/reward_histogram.png \
    --interval 0.05 \
    --split-models