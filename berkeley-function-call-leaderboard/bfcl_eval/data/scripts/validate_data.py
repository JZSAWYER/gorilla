#!/usr/bin/env python3
"""Validate WM/SFT data for LLaMA-Factory compatibility."""
import json
import sys

def validate_file(path):
    print(f"\n=== Validating: {path} ===")
    
    with open(path) as f:
        data = json.load(f)
    
    print(f"Total samples: {len(data)}")
    
    issues = []
    for i, sample in enumerate(data):
        # Check tools
        tools = sample.get('tools')
        if not isinstance(tools, str):
            issues.append(f"Sample {i}: tools is {type(tools).__name__}, should be str")
        
        # Check messages
        for j, msg in enumerate(sample.get('messages', [])):
            # Check tool_calls
            if 'tool_calls' in msg:
                tc = msg['tool_calls']
                if tc is None:
                    issues.append(f"Sample {i}, Msg {j}: tool_calls is None")
                elif not isinstance(tc, list):
                    issues.append(f"Sample {i}, Msg {j}: tool_calls is {type(tc).__name__}")
            
            # Check content
            content = msg.get('content')
            if content is None and msg.get('role') == 'assistant':
                tc = msg.get('tool_calls')
                if not tc or len(tc) == 0:
                    issues.append(f"Sample {i}, Msg {j}: assistant has content=None but no tool_calls")
    
    if issues:
        print(f"ERRORS: {len(issues)} issues found:")
        for issue in issues[:20]:
            print(f"  - {issue}")
        return False
    else:
        print("OK: All checks passed!")
        return True

if __name__ == "__main__":
    base = "/home/dataset-local/projects/gorilla/berkeley-function-call-leaderboard/bfcl_eval/data/generated/aligned"
    # Also try remote path
    try:
        base = "/inspire/hdd/project/chemicalreaction/dijixiu-CZXS25220051/projects/gorilla/berkeley-function-call-leaderboard/bfcl_eval/data/generated/aligned"
    except:
        pass
    
    files = [
        f"{base}/aligned_wm_train.json",
        f"{base}/aligned_wm_test.json",
    ]
    
    all_ok = True
    for f in files:
        try:
            if not validate_file(f):
                all_ok = False
        except FileNotFoundError:
            print(f"File not found: {f}")
            all_ok = False
    
    sys.exit(0 if all_ok else 1)
