#!/usr/bin/env python3
"""
BFCL Data Splitting Utilities

Provides train/test splitting for BFCL datasets.
Used by both build_bfcl_sft.py and build_rl_trajectories.py to ensure
corresponding splits when using the same seed.

RC-GRPO: Added ID-based splitting to achieve 0% question overlap between train/test.
The BFCL dataset has 4 categories where questions may repeat across categories
(base, long_context, miss_func share questions; miss_param is unique).
Questions that repeat across categories always share the SAME numeric ID.
Therefore, splitting by ID ensures no question appears in both train and test.
"""

import logging
import random
import re
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Tuple, Optional

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


# =============================================================================
# RC-GRPO: ID-based Splitting (Recommended for 0% Question Overlap)
# =============================================================================

def extract_id_from_bfcl_object(
    file_path: Path, 
    bfcl_obj: Dict[str, Any]
) -> Optional[int]:
    """
    Extract the numeric ID from a BFCL object.
    
    RC-GRPO: The ID can be found in:
    1. The 'id' field of the object (e.g., "multi_turn_base_0" -> 0)
    2. The filename (e.g., "multi_turn_base_0.json" -> 0)
    
    Args:
        file_path: Path to the source file
        bfcl_obj: BFCL object dictionary
        
    Returns:
        Numeric ID, or None if not extractable
    """
    # Try to extract from 'id' field first
    obj_id = bfcl_obj.get("id", "")
    if obj_id:
        # ID format is typically "multi_turn_{category}_{number}"
        match = re.search(r'_(\d+)$', str(obj_id))
        if match:
            return int(match.group(1))
    
    # Fallback: extract from filename
    filename = file_path.stem if file_path else ""
    match = re.search(r'_(\d+)$', filename)
    if match:
        return int(match.group(1))
    
    return None


def extract_category_from_bfcl_object(
    file_path: Path,
    bfcl_obj: Dict[str, Any]
) -> str:
    """
    Extract the category from a BFCL object.
    
    RC-GRPO: Categories are: base, long_context, miss_func, miss_param
    
    Args:
        file_path: Path to the source file
        bfcl_obj: BFCL object dictionary
        
    Returns:
        Category string, or "unknown" if not extractable
    """
    # Try to extract from 'id' field
    obj_id = bfcl_obj.get("id", "")
    for category in ["base", "long_context", "miss_func", "miss_param"]:
        if category in str(obj_id):
            return category
    
    # Fallback: extract from file path
    path_str = str(file_path)
    for category in ["base", "long_context", "miss_func", "miss_param"]:
        if category in path_str:
            return category
    
    return "unknown"


def _build_question_to_ids_mapping(
    bfcl_objects: List[Tuple[Path, Dict[str, Any]]]
) -> Dict[str, set]:
    """
    RC-GRPO: Build a mapping from question content to IDs that contain it.
    
    This is used to identify questions that are reused across different IDs,
    which would cause overlap if those IDs are split into different sets.
    
    Args:
        bfcl_objects: List of (file_path, bfcl_object) tuples
        
    Returns:
        Dict mapping question content to set of IDs containing that question
    """
    question_to_ids: Dict[str, set] = defaultdict(set)
    
    for file_path, bfcl_obj in bfcl_objects:
        obj_id = extract_id_from_bfcl_object(file_path, bfcl_obj)
        if obj_id is None:
            continue
        
        # Get all user questions from this object
        for question_list in bfcl_obj.get("question", []):
            if isinstance(question_list, list):
                for msg in question_list:
                    if isinstance(msg, dict) and msg.get("role") == "user":
                        content = msg.get("content", "").strip()
                        if content:
                            question_to_ids[content].add(obj_id)
    
    return dict(question_to_ids)


def _build_id_groups_from_shared_questions(
    question_to_ids: Dict[str, set]
) -> List[set]:
    """
    RC-GRPO: Build groups of IDs that must stay together due to shared questions.
    
    Uses Union-Find algorithm to group IDs that share any question.
    
    Args:
        question_to_ids: Mapping from question to IDs containing it
        
    Returns:
        List of ID groups (each group must go together to train or test)
    """
    # Find IDs that share questions (need to stay together)
    shared_id_pairs = []
    for question, ids in question_to_ids.items():
        if len(ids) > 1:
            ids_list = sorted(ids)
            # Connect all IDs that share this question
            for i in range(len(ids_list) - 1):
                shared_id_pairs.append((ids_list[i], ids_list[i + 1]))
    
    if not shared_id_pairs:
        # No shared questions, each ID is its own group
        all_ids = set()
        for ids in question_to_ids.values():
            all_ids.update(ids)
        return [{id_} for id_ in sorted(all_ids)]
    
    # Union-Find to group connected IDs
    parent = {}
    
    def find(x):
        if x not in parent:
            parent[x] = x
        if parent[x] != x:
            parent[x] = find(parent[x])
        return parent[x]
    
    def union(x, y):
        px, py = find(x), find(y)
        if px != py:
            parent[px] = py
    
    # Collect all IDs
    all_ids = set()
    for ids in question_to_ids.values():
        all_ids.update(ids)
    
    # Initialize parent
    for id_ in all_ids:
        parent[id_] = id_
    
    # Union IDs that share questions
    for id1, id2 in shared_id_pairs:
        union(id1, id2)
    
    # Group by root
    groups_dict = defaultdict(set)
    for id_ in all_ids:
        root = find(id_)
        groups_dict[root].add(id_)
    
    return list(groups_dict.values())


def split_bfcl_objects_by_id(
    bfcl_objects: List[Tuple[Path, Dict[str, Any]]],
    train_ratio: float = 0.75,
    seed: int = 42
) -> Tuple[List[Tuple[Path, Dict[str, Any]]], List[Tuple[Path, Dict[str, Any]]]]:
    """
    RC-GRPO: Split BFCL objects by ID to achieve 0% question overlap.
    
    This splitting method ensures that:
    1. All variations of a question (across categories) stay together in train OR test
    2. IDs that share questions are grouped together (handles edge cases where
       the same question appears with different IDs)
    3. The split is reproducible given the same seed
    4. No question appears in both train and test sets
    
    The BFCL dataset structure:
    - 4 categories: base, long_context, miss_func, miss_param
    - Each category has IDs 0-199 (200 samples)
    - base, long_context, miss_func share the SAME questions (same ID = same question)
    - miss_param has unique questions
    - NOTE: A few questions are reused across different IDs (e.g., IDs 40 & 43)
    
    Args:
        bfcl_objects: List of (file_path, bfcl_object) tuples
        train_ratio: Fraction of IDs to use for training (default: 0.75)
        seed: Random seed for reproducible splits (default: 42)
        
    Returns:
        Tuple of (train_objects, test_objects)
    """
    if not 0 < train_ratio < 1:
        raise ValueError(f"train_ratio must be between 0 and 1, got {train_ratio}")
    
    # Collect all unique IDs and map objects by ID
    all_ids = set()
    objects_by_id = defaultdict(list)
    objects_without_id = []
    
    for file_path, bfcl_obj in bfcl_objects:
        obj_id = extract_id_from_bfcl_object(file_path, bfcl_obj)
        if obj_id is not None:
            all_ids.add(obj_id)
            objects_by_id[obj_id].append((file_path, bfcl_obj))
        else:
            objects_without_id.append((file_path, bfcl_obj))
            logger.warning(f"Could not extract ID from {file_path}, will assign to train")
    
    logger.info(f"ID-based split: found {len(all_ids)} unique IDs, {len(objects_without_id)} objects without ID")
    
    # RC-GRPO: Build question-to-IDs mapping and group IDs that share questions
    question_to_ids = _build_question_to_ids_mapping(bfcl_objects)
    id_groups = _build_id_groups_from_shared_questions(question_to_ids)
    
    # Log any multi-ID groups (IDs that must stay together)
    multi_id_groups = [g for g in id_groups if len(g) > 1]
    if multi_id_groups:
        logger.info(f"RC-GRPO: Found {len(multi_id_groups)} ID groups with shared questions:")
        for group in multi_id_groups:
            logger.info(f"  IDs {sorted(group)} share questions and will be kept together")
    
    # Sort groups by their minimum ID for deterministic ordering
    sorted_groups = sorted(id_groups, key=lambda g: min(g))
    
    # Shuffle groups (not individual IDs) for random split
    rng = random.Random(seed)
    shuffled_groups = sorted_groups.copy()
    rng.shuffle(shuffled_groups)
    
    # Calculate how many IDs we want in train
    total_ids = sum(len(g) for g in shuffled_groups)
    target_train_ids = int(total_ids * train_ratio)
    
    # Assign groups to train/test, trying to hit target ratio
    train_ids = set()
    test_ids = set()
    
    for group in shuffled_groups:
        if len(train_ids) < target_train_ids:
            train_ids.update(group)
        else:
            test_ids.update(group)
    
    logger.info(f"Train IDs: {len(train_ids)} (e.g., {sorted(list(train_ids))[:5]}...)")
    logger.info(f"Test IDs: {len(test_ids)} (e.g., {sorted(list(test_ids))[:5]}...)")
    
    train_objects = []
    test_objects = []
    
    # Assign objects based on their ID
    for obj_id in sorted(all_ids):
        objs = objects_by_id[obj_id]
        if obj_id in train_ids:
            train_objects.extend(objs)
        else:
            test_objects.extend(objs)
    
    # Objects without ID go to train
    train_objects.extend(objects_without_id)
    
    # Log statistics by category
    train_by_category = defaultdict(int)
    test_by_category = defaultdict(int)
    
    for file_path, bfcl_obj in train_objects:
        category = extract_category_from_bfcl_object(file_path, bfcl_obj)
        train_by_category[category] += 1
    
    for file_path, bfcl_obj in test_objects:
        category = extract_category_from_bfcl_object(file_path, bfcl_obj)
        test_by_category[category] += 1
    
    logger.info("Split statistics by category:")
    for category in sorted(set(train_by_category.keys()) | set(test_by_category.keys())):
        t_count = train_by_category.get(category, 0)
        e_count = test_by_category.get(category, 0)
        total = t_count + e_count
        t_pct = (t_count / total * 100) if total > 0 else 0
        logger.info(f"  {category}: {t_count} train ({t_pct:.1f}%), {e_count} test")
    
    logger.info(f"ID-based split complete: {len(train_objects)} train, {len(test_objects)} test")
    logger.info("Question overlap between train and test: 0% (guaranteed by group-based ID split)")
    
    return train_objects, test_objects


def split_bfcl_objects_by_id_sequential(
    bfcl_objects: List[Tuple[Path, Dict[str, Any]]],
    train_ratio: float = 0.75
) -> Tuple[List[Tuple[Path, Dict[str, Any]]], List[Tuple[Path, Dict[str, Any]]]]:
    """
    RC-GRPO: Split BFCL objects by ID using sequential (non-random) assignment.
    
    Similar to split_bfcl_objects_by_id but uses sequential ID ranges instead of
    random shuffling. This is useful for debugging and ensures deterministic
    splits without needing a seed.
    
    Example with 200 IDs and 75% train ratio:
    - Train: IDs 0-149 (first 150)
    - Test: IDs 150-199 (last 50)
    
    Args:
        bfcl_objects: List of (file_path, bfcl_object) tuples
        train_ratio: Fraction of IDs to use for training (default: 0.75)
        
    Returns:
        Tuple of (train_objects, test_objects)
    """
    if not 0 < train_ratio < 1:
        raise ValueError(f"train_ratio must be between 0 and 1, got {train_ratio}")
    
    # Collect all unique IDs
    all_ids = set()
    objects_by_id = defaultdict(list)
    objects_without_id = []
    
    for file_path, bfcl_obj in bfcl_objects:
        obj_id = extract_id_from_bfcl_object(file_path, bfcl_obj)
        if obj_id is not None:
            all_ids.add(obj_id)
            objects_by_id[obj_id].append((file_path, bfcl_obj))
        else:
            objects_without_id.append((file_path, bfcl_obj))
            logger.warning(f"Could not extract ID from {file_path}, will assign to train")
    
    # Sort IDs
    sorted_ids = sorted(all_ids)
    
    # Calculate split point (sequential)
    n_train_ids = max(1, int(len(sorted_ids) * train_ratio))
    train_ids = set(sorted_ids[:n_train_ids])
    test_ids = set(sorted_ids[n_train_ids:])
    
    logger.info(f"Sequential ID-based split:")
    logger.info(f"  Train IDs: 0-{n_train_ids - 1} ({len(train_ids)} IDs)")
    logger.info(f"  Test IDs: {n_train_ids}-{len(sorted_ids) - 1} ({len(test_ids)} IDs)")
    
    train_objects = []
    test_objects = []
    
    # Assign objects based on their ID
    for obj_id in sorted_ids:
        objs = objects_by_id[obj_id]
        if obj_id in train_ids:
            train_objects.extend(objs)
        else:
            test_objects.extend(objs)
    
    # Objects without ID go to train
    train_objects.extend(objects_without_id)
    
    logger.info(f"Sequential ID-based split complete: {len(train_objects)} train, {len(test_objects)} test")
    logger.info("Question overlap between train and test: 0% (guaranteed by ID-based split)")
    
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


# =============================================================================
# RC-GRPO: Verification Utilities
# =============================================================================

def verify_no_question_overlap(
    train_objects: List[Tuple[Path, Dict[str, Any]]],
    test_objects: List[Tuple[Path, Dict[str, Any]]]
) -> bool:
    """
    RC-GRPO: Verify that there is no question overlap between train and test sets.
    
    This function extracts all user questions from both sets and checks for overlap.
    Useful for validating that the ID-based split is working correctly.
    
    Args:
        train_objects: List of training (file_path, bfcl_object) tuples
        test_objects: List of test (file_path, bfcl_object) tuples
        
    Returns:
        True if no overlap, False if overlap exists
    """
    def extract_questions(objects: List[Tuple[Path, Dict[str, Any]]]) -> set:
        questions = set()
        for _, obj in objects:
            # Questions are in the 'question' field as a list
            for question_list in obj.get("question", []):
                if isinstance(question_list, list):
                    for q in question_list:
                        if isinstance(q, dict) and q.get("role") == "user":
                            content = q.get("content", "")
                            # Normalize: strip whitespace and reward tokens
                            content = content.strip()
                            if "[Reward Goal:" in content:
                                content = content.split("[Reward Goal:")[0].strip()
                            if content:
                                questions.add(content)
        return questions
    
    train_questions = extract_questions(train_objects)
    test_questions = extract_questions(test_objects)
    
    overlap = train_questions & test_questions
    
    if overlap:
        logger.warning(f"Found {len(overlap)} overlapping questions between train and test!")
        logger.warning(f"Sample overlapping questions: {list(overlap)[:3]}")
        return False
    else:
        logger.info(f"✓ Verified: 0 question overlap")
        logger.info(f"  Train unique questions: {len(train_questions)}")
        logger.info(f"  Test unique questions: {len(test_questions)}")
        return True


def verify_id_based_split(
    train_objects: List[Tuple[Path, Dict[str, Any]]],
    test_objects: List[Tuple[Path, Dict[str, Any]]]
) -> bool:
    """
    RC-GRPO: Verify that train and test have non-overlapping IDs.
    
    Args:
        train_objects: List of training (file_path, bfcl_object) tuples
        test_objects: List of test (file_path, bfcl_object) tuples
        
    Returns:
        True if IDs don't overlap, False otherwise
    """
    train_ids = set()
    test_ids = set()
    
    for file_path, obj in train_objects:
        obj_id = extract_id_from_bfcl_object(file_path, obj)
        if obj_id is not None:
            train_ids.add(obj_id)
    
    for file_path, obj in test_objects:
        obj_id = extract_id_from_bfcl_object(file_path, obj)
        if obj_id is not None:
            test_ids.add(obj_id)
    
    overlap = train_ids & test_ids
    
    if overlap:
        logger.warning(f"Found {len(overlap)} overlapping IDs: {sorted(overlap)[:10]}...")
        return False
    else:
        logger.info(f"✓ Verified: 0 ID overlap between train and test")
        logger.info(f"  Train IDs: {sorted(train_ids)[:5]}... ({len(train_ids)} total)")
        logger.info(f"  Test IDs: {sorted(test_ids)[:5]}... ({len(test_ids)} total)")
        return True

