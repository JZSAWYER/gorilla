#!/usr/bin/env python3
"""
Align WM/SFT data train/test split with RL data split.

This ensures that:
- WM train contains tasks from RL train
- WM test contains tasks from RL test
- No train/test leakage between WM and RL

Usage:
    python align_wm_with_rl_split.py \
        --rl-train /path/to/bfcl_rl_train.json \
        --rl-test /path/to/bfcl_rl_test.json \
        --wm-train /path/to/aligned_wm_train.json \
        --wm-test /path/to/aligned_wm_test.json \
        --output-dir /path/to/output
"""

import argparse
import json
import logging
from pathlib import Path
from collections import defaultdict

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description="Align WM/SFT split with RL split")
    parser.add_argument("--rl-train", required=True, help="Path to RL train data")
    parser.add_argument("--rl-test", required=True, help="Path to RL test data")
    parser.add_argument("--wm-train", required=True, help="Path to WM train data")
    parser.add_argument("--wm-test", required=True, help="Path to WM test data")
    parser.add_argument("--output-dir", required=True, help="Output directory")
    parser.add_argument("--prefix", default="aligned", help="Output file prefix")
    
    args = parser.parse_args()
    
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load RL data to get task_id splits
    logger.info("Loading RL data...")
    with open(args.rl_train) as f:
        rl_train = json.load(f)
    with open(args.rl_test) as f:
        rl_test = json.load(f)
    
    rl_train_ids = set(s['task_id'] for s in rl_train)
    rl_test_ids = set(s['task_id'] for s in rl_test)
    
    logger.info(f"RL train task_ids: {len(rl_train_ids)}")
    logger.info(f"RL test task_ids: {len(rl_test_ids)}")
    
    # Load all WM data
    logger.info("Loading WM data...")
    with open(args.wm_train) as f:
        wm_train = json.load(f)
    with open(args.wm_test) as f:
        wm_test = json.load(f)
    
    # Combine all WM samples
    all_wm_samples = wm_train + wm_test
    logger.info(f"Total WM samples: {len(all_wm_samples)}")
    
    # Group by task_id (in case there are multiple trajectories per task)
    wm_by_task = defaultdict(list)
    for sample in all_wm_samples:
        wm_by_task[sample['task_id']].append(sample)
    
    logger.info(f"Unique WM task_ids: {len(wm_by_task)}")
    
    # Realign based on RL split
    new_wm_train = []
    new_wm_test = []
    
    unmatched_rl_train = []
    unmatched_rl_test = []
    
    for task_id in rl_train_ids:
        if task_id in wm_by_task:
            new_wm_train.extend(wm_by_task[task_id])
        else:
            unmatched_rl_train.append(task_id)
    
    for task_id in rl_test_ids:
        if task_id in wm_by_task:
            new_wm_test.extend(wm_by_task[task_id])
        else:
            unmatched_rl_test.append(task_id)
    
    # Check for WM tasks not in RL
    wm_task_ids = set(wm_by_task.keys())
    extra_wm = wm_task_ids - rl_train_ids - rl_test_ids
    
    logger.info("")
    logger.info("=== Alignment Results ===")
    logger.info(f"New WM train: {len(new_wm_train)} samples")
    logger.info(f"New WM test: {len(new_wm_test)} samples")
    
    if unmatched_rl_train:
        logger.warning(f"RL train task_ids not found in WM: {len(unmatched_rl_train)}")
        for tid in unmatched_rl_train[:5]:
            logger.warning(f"  - {tid}")
    
    if unmatched_rl_test:
        logger.warning(f"RL test task_ids not found in WM: {len(unmatched_rl_test)}")
        for tid in unmatched_rl_test[:5]:
            logger.warning(f"  - {tid}")
    
    if extra_wm:
        logger.warning(f"WM task_ids not in RL: {len(extra_wm)}")
        for tid in list(extra_wm)[:5]:
            logger.warning(f"  - {tid}")
    
    # Verify no overlap
    new_train_ids = set(s['task_id'] for s in new_wm_train)
    new_test_ids = set(s['task_id'] for s in new_wm_test)
    overlap = new_train_ids & new_test_ids
    
    if overlap:
        logger.error(f"ERROR: Train/test overlap detected: {len(overlap)} task_ids")
        return 1
    
    logger.info("")
    logger.info("Verified: No train/test overlap ✓")
    
    # Save aligned data
    wm_train_path = output_dir / f"{args.prefix}_wm_train.json"
    wm_test_path = output_dir / f"{args.prefix}_wm_test.json"
    
    with open(wm_train_path, 'w') as f:
        json.dump(new_wm_train, f, indent=2, ensure_ascii=False)
    logger.info(f"Saved: {wm_train_path}")
    
    with open(wm_test_path, 'w') as f:
        json.dump(new_wm_test, f, indent=2, ensure_ascii=False)
    logger.info(f"Saved: {wm_test_path}")
    
    # Also do the same for SFT if available
    sft_train_path = Path(args.wm_train).parent / Path(args.wm_train).name.replace('wm', 'sft')
    sft_test_path = Path(args.wm_test).parent / Path(args.wm_test).name.replace('wm', 'sft')
    
    if sft_train_path.exists() and sft_test_path.exists():
        logger.info("")
        logger.info("Processing SFT data...")
        
        with open(sft_train_path) as f:
            sft_train = json.load(f)
        with open(sft_test_path) as f:
            sft_test = json.load(f)
        
        all_sft_samples = sft_train + sft_test
        sft_by_task = defaultdict(list)
        for sample in all_sft_samples:
            sft_by_task[sample['task_id']].append(sample)
        
        new_sft_train = []
        new_sft_test = []
        
        for task_id in rl_train_ids:
            if task_id in sft_by_task:
                new_sft_train.extend(sft_by_task[task_id])
        
        for task_id in rl_test_ids:
            if task_id in sft_by_task:
                new_sft_test.extend(sft_by_task[task_id])
        
        logger.info(f"New SFT train: {len(new_sft_train)} samples")
        logger.info(f"New SFT test: {len(new_sft_test)} samples")
        
        sft_train_out = output_dir / f"{args.prefix}_sft_train.json"
        sft_test_out = output_dir / f"{args.prefix}_sft_test.json"
        
        with open(sft_train_out, 'w') as f:
            json.dump(new_sft_train, f, indent=2, ensure_ascii=False)
        logger.info(f"Saved: {sft_train_out}")
        
        with open(sft_test_out, 'w') as f:
            json.dump(new_sft_test, f, indent=2, ensure_ascii=False)
        logger.info(f"Saved: {sft_test_out}")
    
    logger.info("")
    logger.info("=== Summary ===")
    logger.info(f"WM Train: {len(new_wm_train)} samples ({len(new_train_ids)} unique tasks)")
    logger.info(f"WM Test: {len(new_wm_test)} samples ({len(new_test_ids)} unique tasks)")
    logger.info("Train/Test split now aligned with RL data ✓")
    
    return 0


if __name__ == "__main__":
    exit(main())
