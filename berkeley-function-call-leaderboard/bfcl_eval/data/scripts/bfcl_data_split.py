#!/usr/bin/env python3
"""
BFCL Data Splitting Utilities

Provides stratified train/test splitting for BFCL datasets.
Used by both build_bfcl_sft.py and build_rl_trajectories.py to ensure
corresponding splits when using the same seed.
"""

import logging
import random
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Tuple

logger = logging.getLogger(__name__)


def get_stratification_key(bfcl_obj: Dict[str, Any]) -> str:
    """
    Get the stratification key for a BFCL object.
    
    Uses the sorted, joined involved_classes as the key to ensure
    objects with the same API combinations are grouped together.
    
    Args:
        bfcl_obj: BFCL object dictionary
        
    Returns:
        Stratification key string (e.g., "GorillaFileSystem,TwitterAPI")
    """
    involved_classes = bfcl_obj.get("involved_classes", [])
    if not involved_classes:
        return "unknown"
    
    # Sort to ensure consistent key regardless of order in the list
    return ",".join(sorted(involved_classes))


def stratified_split_bfcl_objects(
    bfcl_objects: List[Tuple[Path, Dict[str, Any]]],
    train_ratio: float = 0.8,
    seed: int = 42
) -> Tuple[List[Tuple[Path, Dict[str, Any]]], List[Tuple[Path, Dict[str, Any]]]]:
    """
    Split BFCL objects with stratification by involved_classes.
    
    This function ensures that:
    1. Objects with the same involved_classes combination are split proportionally
    2. The split is reproducible given the same seed
    3. Both build_bfcl_sft.py and build_rl_trajectories.py get the same split
       when using the same seed
    
    Args:
        bfcl_objects: List of (file_path, bfcl_object) tuples
        train_ratio: Fraction of data to use for training (default: 0.8)
        seed: Random seed for reproducible splits (default: 42)
        
    Returns:
        Tuple of (train_objects, test_objects)
    """
    if not 0 < train_ratio < 1:
        raise ValueError(f"train_ratio must be between 0 and 1, got {train_ratio}")
    
    # Group objects by stratification key (involved_classes combination)
    strata = defaultdict(list)
    for file_path, bfcl_obj in bfcl_objects:
        key = get_stratification_key(bfcl_obj)
        strata[key].append((file_path, bfcl_obj))
    
    logger.info(f"Stratified split: found {len(strata)} unique API combinations")
    for key, objects in sorted(strata.items()):
        logger.debug(f"  {key}: {len(objects)} samples")
    
    # Set random seed for reproducibility
    rng = random.Random(seed)
    
    train_objects = []
    test_objects = []
    
    # Split each stratum independently
    for key in sorted(strata.keys()):  # Sort keys for deterministic order
        stratum_objects = strata[key]
        
        # Shuffle within stratum using seeded RNG
        # Create a copy to avoid modifying original list
        shuffled = stratum_objects.copy()
        rng.shuffle(shuffled)
        
        # Calculate split point
        n_train = max(1, int(len(shuffled) * train_ratio))
        
        # Handle edge case: if only 1 sample, put in train
        if len(shuffled) == 1:
            train_objects.extend(shuffled)
            logger.debug(f"  {key}: 1 sample -> all in train")
        else:
            train_objects.extend(shuffled[:n_train])
            test_objects.extend(shuffled[n_train:])
            logger.debug(f"  {key}: {len(shuffled)} samples -> {n_train} train, {len(shuffled) - n_train} test")
    
    logger.info(f"Split complete: {len(train_objects)} train, {len(test_objects)} test")
    
    return train_objects, test_objects


def get_split_output_paths(output_path: Path) -> Tuple[Path, Path]:
    """
    Generate train and test output paths from the base output path.
    
    Args:
        output_path: Base output path (e.g., "output.json")
        
    Returns:
        Tuple of (train_path, test_path)
        E.g., ("output_train.json", "output_test.json")
    """
    stem = output_path.stem
    suffix = output_path.suffix
    parent = output_path.parent
    
    train_path = parent / f"{stem}_train{suffix}"
    test_path = parent / f"{stem}_test{suffix}"
    
    return train_path, test_path


def log_split_statistics(
    train_objects: List[Tuple[Path, Dict[str, Any]]],
    test_objects: List[Tuple[Path, Dict[str, Any]]]
) -> None:
    """
    Log detailed statistics about the train/test split.
    
    Args:
        train_objects: List of training (file_path, bfcl_object) tuples
        test_objects: List of test (file_path, bfcl_object) tuples
    """
    # Count by API class in each split
    train_api_counts = defaultdict(int)
    test_api_counts = defaultdict(int)
    
    for _, obj in train_objects:
        for api_class in obj.get("involved_classes", []):
            train_api_counts[api_class] += 1
    
    for _, obj in test_objects:
        for api_class in obj.get("involved_classes", []):
            test_api_counts[api_class] += 1
    
    # Get all unique APIs
    all_apis = sorted(set(train_api_counts.keys()) | set(test_api_counts.keys()))
    
    logger.info("Split statistics by API class:")
    for api in all_apis:
        train_count = train_api_counts.get(api, 0)
        test_count = test_api_counts.get(api, 0)
        total = train_count + test_count
        train_pct = (train_count / total * 100) if total > 0 else 0
        logger.info(f"  {api}: {train_count} train ({train_pct:.1f}%), {test_count} test")


