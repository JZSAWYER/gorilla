#!/bin/bash
# =============================================================================
# Build Expanded WMFT and SFT Datasets for BFCL (RC-GRPO Aligned)
# =============================================================================
#
# This script generates expanded datasets that align with the RL training format
# used in verl/examples/bfcl for RC-GRPO training.
#
# Outputs:
#   - expanded_wm_train.json - WMFT training (all trajectories, binary RC tokens)
#   - expanded_wm_test.json - WMFT test (all trajectories, binary RC tokens)
#   - expanded_sft_train.json - SFT training (success-only, no RC tokens)
#   - expanded_sft_test.json - SFT test (success-only, no RC tokens)
#
# Usage:
#   cd data/scripts
#   ./build_expanded_data.sh
# =============================================================================

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DATA_DIR="$(dirname "$SCRIPT_DIR")"

# Input paths
RL_TRAJECTORIES="${DATA_DIR}/generated/qwen2.5-7b-instruct/rl/rl_trajectories.json"
REWARD_RESULTS="${DATA_DIR}/generated/evaluation/trajectory_reward_results.json"

# Output directory
OUTPUT_DIR="${DATA_DIR}/generated/expanded"

# Configuration
FORMAT="openai"  # openai or glaive
TRAIN_RATIO="0.8"
SEED="42"

echo "============================================================"
echo "Building Expanded WMFT and SFT Datasets for BFCL"
echo "============================================================"
echo ""
echo "Configuration:"
echo "  RL Trajectories: ${RL_TRAJECTORIES}"
echo "  Reward Results:  ${REWARD_RESULTS}"
echo "  Output Dir:      ${OUTPUT_DIR}"
echo "  Format:          ${FORMAT}"
echo "  Train Ratio:     ${TRAIN_RATIO}"
echo "  Seed:            ${SEED}"
echo ""

# Check inputs exist
if [ ! -f "$RL_TRAJECTORIES" ]; then
    echo "ERROR: RL trajectories file not found: $RL_TRAJECTORIES"
    echo ""
    echo "Available RL trajectory files:"
    find "${DATA_DIR}/generated" -name "rl_trajectories*.json" 2>/dev/null || echo "  None found"
    exit 1
fi

if [ ! -f "$REWARD_RESULTS" ]; then
    echo "ERROR: Reward results file not found: $REWARD_RESULTS"
    echo ""
    echo "You may need to run evaluate_trajectory_rewards.py first."
    exit 1
fi

# Create output directory
mkdir -p "$OUTPUT_DIR"

# Run the builder
echo "Running build_expanded_wmft_sft.py..."
echo ""

python3 "${SCRIPT_DIR}/build_expanded_wmft_sft.py" \
    --rl-trajectories "$RL_TRAJECTORIES" \
    --reward-results "$REWARD_RESULTS" \
    --output-dir "$OUTPUT_DIR" \
    --format "$FORMAT" \
    --split \
    --train-ratio "$TRAIN_RATIO" \
    --seed "$SEED"

echo ""
echo "============================================================"
echo "Output Files:"
echo "============================================================"
ls -la "$OUTPUT_DIR"/*.json 2>/dev/null || echo "No output files found"

echo ""
echo "============================================================"
echo "Sample counts:"
echo "============================================================"
for f in "$OUTPUT_DIR"/*.json; do
    if [ -f "$f" ]; then
        count=$(python3 -c "import json; print(len(json.load(open('$f'))))")
        echo "  $(basename $f): $count samples"
    fi
done

echo ""
echo "Done!"
