import json
from collections import Counter

file_path = "/data1/jzsawyer/projects/gorilla/berkeley-function-call-leaderboard/bfcl_eval/data/BFCL_v4_multi_turn_miss_func.json"

# Track all unique key sets
key_sets = []
entries_with_keys = {}

with open(file_path, 'r') as f:
    for line_num, line in enumerate(f, 1):
        try:
            entry = json.loads(line.strip())
            keys = tuple(sorted(entry.keys()))
            key_sets.append(keys)
            entries_with_keys[line_num] = keys
        except json.JSONDecodeError as e:
            print(f"Error parsing line {line_num}: {e}")

# Count unique key structures
key_counter = Counter(key_sets)

print(f"Total entries: {len(key_sets)}")
print(f"\nUnique key structures found: {len(key_counter)}")
print("\n" + "="*60)

# Show all unique key structures
for idx, (keys, count) in enumerate(key_counter.items(), 1):
    print(f"\nStructure {idx} (appears {count} times):")
    print(f"  Keys: {list(keys)}")
    print(f"  Entry IDs with this structure:")
    matching_lines = [line for line, k in entries_with_keys.items() if k == keys]
    if count <= 10:
        print(f"    Lines: {matching_lines}")
    else:
        print(f"    Lines: {matching_lines[:10]}... (and {count-10} more)")

# Check for differences
if len(key_counter) > 1:
    print("\n" + "="*60)
    print("DIFFERENCES FOUND:")
    print("="*60)
    
    # Find common keys and unique keys per structure
    all_keys = set()
    for keys in key_counter.keys():
        all_keys.update(keys)
    
    print(f"\nAll keys found across all entries: {sorted(all_keys)}")
    
    # Show which keys are missing in which structures
    for keys, count in key_counter.items():
        missing_keys = all_keys - set(keys)
        if missing_keys:
            print(f"\nStructure with keys {list(keys)} is missing: {sorted(missing_keys)}")
else:
    print("\n✓ All entries have the same key structure!")
    print(f"  Common keys: {list(key_counter.keys())[0]}")