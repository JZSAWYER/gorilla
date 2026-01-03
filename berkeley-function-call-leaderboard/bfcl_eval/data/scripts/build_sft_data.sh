#!/bin/bash
# Convert BFCL v4 multi-turn JSON files to Glaive toolcall dataset format
# Each multi-turn episode is converted to one Glaive record with conversations array
# Tool documentation is loaded from multi_turn_func_doc/ directory

python build_bfcl_sft.py \
  --input_dir ../ \
  --output_path ../generated/sft/bfcl_glaive_dataset.json \
  --split \
  --train_ratio 0.75