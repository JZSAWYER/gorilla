python build_rl_trajectories.py \
  --input_dir /data1/jzsawyer/projects/gorilla/berkeley-function-call-leaderboard/bfcl_eval/data \
  --output_path ./rl_trajectories.json \
  --expert_model_path /data1/jzsawyer/models/Qwen2.5-7B-Instruct \
  --intermediate_model_path /data1/jzsawyer/models/Qwen2.5-1.5B-Instruct \
  --weak_model_path /data1/jzsawyer/models/Qwen2.5-0.5B-Instruct \
  --test_mode \
  --samples_per_api_file 3

# Generate for a specific API with n samples:
# python build_rl_trajectories.py \
#   --input_dir /path/to/bfcl/data \
#   --output_path /path/to/output_filesystem.json \
#   --expert_model_path /path/to/expert/model \
#   --intermediate_model_path /path/to/intermediate/model \
#   --weak_model_path /path/to/weak/model \
#   --test_mode \
#   --test_api GorillaFileSystem \
#   --samples_per_api_file 5

# full
# python build_rl_trajectories.py \
#   --input_dir /path/to/bfcl/data \
#   --output_path /path/to/full_output.json \
#   --expert_model_path /path/to/expert/model \
#   --intermediate_model_path /path/to/intermediate/model \
#   --weak_model_path /path/to/weak/model