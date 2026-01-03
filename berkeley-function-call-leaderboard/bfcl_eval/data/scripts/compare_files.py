#!/usr/bin/env python3
"""
Compare two JSONL files to check if they are identical except for the 'id' and 'excluded_function' fields.
Writes differences to a JSON file.
"""

import json
import sys
import re
from pathlib import Path

# Fields to ignore when comparing
IGNORED_FIELDS = {'id', 'excluded_function'}


def remove_ignored_fields(obj):
    """Remove ignored fields ('id' and 'excluded_function') from a dictionary."""
    if isinstance(obj, dict):
        return {k: remove_ignored_fields(v) for k, v in obj.items() if k not in IGNORED_FIELDS}
    elif isinstance(obj, list):
        return [remove_ignored_fields(item) for item in obj]
    else:
        return obj


def extract_id_number(id_str):
    """Extract the number from an id string (e.g., 'multi_turn_long_context_0' -> '0')."""
    if not id_str or id_str == 'N/A':
        return None
    # Try to find a number at the end of the id
    match = re.search(r'(\d+)$', id_str)
    if match:
        return match.group(1)
    return None


def generate_output_filename(file1_path, file2_path):
    """Generate a human-readable output filename from two input filenames."""
    file1_name = Path(file1_path).stem  # Get filename without extension
    file2_name = Path(file2_path).stem
    
    # Remove common prefixes/suffixes to make it cleaner
    # If both start with the same prefix, remove it from the second
    if file1_name.startswith('BFCL_') and file2_name.startswith('BFCL_'):
        file1_clean = file1_name.replace('BFCL_', '').replace('_', ' ')
        file2_clean = file2_name.replace('BFCL_', '').replace('_', ' ')
    else:
        file1_clean = file1_name.replace('_', ' ')
        file2_clean = file2_name.replace('_', ' ')
    
    output_name = f"{file1_name}_vs_{file2_name}_differences.json"
    return Path(file1_path).parent / output_name


def compare_files(file1_path, file2_path, output_file=None):
    """Compare two JSONL files ignoring the 'id' and 'excluded_function' fields.
    Writes differences to a JSON file.
    """
    file1_path = Path(file1_path)
    file2_path = Path(file2_path)
    
    if not file1_path.exists():
        print(f"Error: File '{file1_path}' does not exist.")
        return False, None
    
    if not file2_path.exists():
        print(f"Error: File '{file2_path}' does not exist.")
        return False, None
    
    # Generate output filename if not provided
    if output_file is None:
        output_file = generate_output_filename(file1_path, file2_path)
    else:
        output_file = Path(output_file)
    
    differences_json = []
    differences_text = []
    
    with open(file1_path, 'r', encoding='utf-8') as f1, \
         open(file2_path, 'r', encoding='utf-8') as f2:
        
        line_num = 0
        
        for line1, line2 in zip(f1, f2):
            line_num += 1
            
            try:
                obj1 = json.loads(line1.strip())
                obj2 = json.loads(line2.strip())
            except json.JSONDecodeError as e:
                differences_text.append(f"Line {line_num}: JSON decode error - {e}")
                continue
            
            # Remove ignored fields from both objects
            obj1_filtered = remove_ignored_fields(obj1)
            obj2_filtered = remove_ignored_fields(obj2)
            
            # Compare the objects
            if obj1_filtered != obj2_filtered:
                id1 = obj1.get('id', 'N/A')
                id2 = obj2.get('id', 'N/A')
                id_number = extract_id_number(id1) or extract_id_number(id2) or str(line_num)
                
                differences_text.append(f"Line {line_num}: Objects differ")
                differences_text.append(f"  File1 id: {id1}")
                differences_text.append(f"  File2 id: {id2}")
                
                # Build difference record
                diff_record = {
                    "id": id_number,
                    "file1_id": id1,
                    "file2_id": id2,
                    "line_number": line_num,
                    "differences": {}
                }
                
                # Find specific differences
                if isinstance(obj1_filtered, dict) and isinstance(obj2_filtered, dict):
                    all_keys = set(obj1_filtered.keys()) | set(obj2_filtered.keys())
                    for key in all_keys:
                        if key not in obj1_filtered:
                            differences_text.append(f"    Key '{key}' missing in file1")
                            diff_record["differences"][key] = {
                                "file1": None,
                                "file2": obj2_filtered[key]
                            }
                        elif key not in obj2_filtered:
                            differences_text.append(f"    Key '{key}' missing in file2")
                            diff_record["differences"][key] = {
                                "file1": obj1_filtered[key],
                                "file2": None
                            }
                        elif obj1_filtered[key] != obj2_filtered[key]:
                            differences_text.append(f"    Key '{key}' differs")
                            diff_record["differences"][key] = {
                                "file1": obj1_filtered[key],
                                "file2": obj2_filtered[key]
                            }
                
                differences_json.append(diff_record)
        
        # Check if files have different number of lines
        remaining_lines1 = list(f1)
        remaining_lines2 = list(f2)
        
        if remaining_lines1:
            differences_text.append(f"File1 has {len(remaining_lines1)} more lines")
        
        if remaining_lines2:
            differences_text.append(f"File2 has {len(remaining_lines2)} more lines")
    
    # Write differences to JSON file with proper formatting
    if differences_json:
        with open(output_file, 'w', encoding='utf-8') as f:
            for i, diff_record in enumerate(differences_json):
                # Write formatted JSON with indentation
                formatted_json = json.dumps(diff_record, ensure_ascii=False, indent=2)
                f.write(formatted_json)
                # Add blank line between records for readability (except after last one)
                if i < len(differences_json) - 1:
                    f.write('\n\n')
        print(f"✓ Differences written to: {output_file}")
        print(f"  Found {len(differences_json)} lines with differences")
    
    # Print text summary
    if differences_text:
        print(f"\nFiles are NOT identical (besides {', '.join(IGNORED_FIELDS)} fields):")
        print("\n".join(differences_text[:20]))  # Show first 20 lines
        if len(differences_text) > 20:
            print(f"... ({len(differences_text) - 20} more lines)")
        return False, output_file
    else:
        print(f"✓ Files are identical (besides {', '.join(IGNORED_FIELDS)} fields). Compared {line_num} lines.")
        return True, None


if __name__ == "__main__":
    # Default file paths
    script_dir = Path(__file__).parent
    file1 = script_dir / "BFCL_v4_multi_turn_long_context.json"
    file2 = script_dir / "BFCL_v4_multi_turn_base.json"
    
    # Allow command line arguments
    if len(sys.argv) >= 3:
        file1 = Path(sys.argv[1])
        file2 = Path(sys.argv[2])
    elif len(sys.argv) == 2:
        print("Usage: python compare_files.py [file1] [file2]")
        print(f"Using default files: {file1} and {file2}")
    
    print(f"Comparing:")
    print(f"  File 1: {file1}")
    print(f"  File 2: {file2}")
    print()
    
    are_identical, output_file = compare_files(file1, file2)
    if output_file:
        print(f"\nOutput file: {output_file}")
    sys.exit(0 if are_identical else 1)

