#!/bin/bash
# Convert BFCL v4 multi-turn JSON files to Glaive toolcall dataset format
# Each multi-turn episode is converted to one Glaive record with conversations array
# Tool documentation is loaded from multi_turn_func_doc/ directory
#
# RC-GRPO: Uses ID-based splitting by default to achieve 0% question overlap
# between train and test sets. The BFCL dataset has questions that repeat across
# categories (base, long_context, miss_func), but they share the same ID.
# Splitting by ID ensures questions don't leak between train and test.

python build_bfcl_sft.py \
  --input_dir ../ \
  --output_path ../generated/sft-id/bfcl_glaive_dataset.json \
  --split \
  --train_ratio 0.75 \
  --split_method id \
  --verify_split \
  --no_reward_prompt