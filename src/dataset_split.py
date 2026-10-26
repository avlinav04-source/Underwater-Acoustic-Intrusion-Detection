"""
dataset_split.py
----------------
Strict recording-level dataset splitting to guarantee ZERO data leakage.

Ensures that segments from any given audio recording strictly appear in
either train, validation, or test sets, and NEVER across multiple splits.
Handles class imbalance and scarce classes (e.g., Tug with only 3 recordings)
with documented, defensible stratification.
"""

import os
import sys
import argparse
from pathlib import Path
from typing import Dict, List, Any, Tuple, Optional
import pandas as pd
import numpy as np

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.utils import (
    load_config,
    ensure_dir,
    save_json,
    load_json,
    CANONICAL_CLASSES,
    CLASS_TO_IDX,
    set_seed
)


def create_recording_level_split(
    metadata_df: pd.DataFrame,
    train_ratio: float = 0.70,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    seed: int = 42
) -> Dict[str, Any]:
    """
    Performs recording-level train/validation/test splitting.

    Args:
        metadata_df: DataFrame with inspected audio records.
        train_ratio: Proportion of recordings for training.
        val_ratio: Proportion of recordings for validation.
        test_ratio: Proportion of recordings for testing.
        seed: Random seed for reproducibility.

    Returns:
        Dict with 'train', 'val', 'test' lists of records and audit report.
    """
    set_seed(seed)

    # Filter only valid recordings
    valid_df = metadata_df[metadata_df["is_valid"]].copy()
    if valid_df.empty:
        raise ValueError("No valid audio recordings found in metadata dataframe.")

    train_records = []
    val_records = []
    test_records = []

    split_audit = {}

    for cls_name in CANONICAL_CLASSES:
        cls_df = valid_df[valid_df["class_name"] == cls_name]
        recordings = cls_df.to_dict(orient="records")
        n_total = len(recordings)

        if n_total == 0:
            continue

        # Shuffle recordings reproducibly
        np.random.seed(seed)
        shuffled = np.random.permutation(recordings).tolist()

        if n_total == 3:
            # Special documented case for Tug (only 3 recordings):
            # To ensure representation in train, val, and test without leakage,
            # assign exactly 1 recording to each split.
            n_train = 1
            n_val = 1
            n_test = 1
        elif n_total < 3:
            # If fewer than 3, cannot represent in all splits
            n_train = max(1, n_total - 1)
            n_val = 1 if n_total == 2 else 0
            n_test = 0
        else:
            # Proportional split
            n_val = max(1, int(round(n_total * val_ratio)))
            n_test = max(1, int(round(n_total * test_ratio)))
            n_train = n_total - n_val - n_test

            # Ensure train has at least 1
            if n_train <= 0:
                n_train = 1
                if n_val > 1:
                    n_val -= 1
                elif n_test > 1:
                    n_test -= 1

        cls_train = shuffled[:n_train]
        cls_val = shuffled[n_train : n_train + n_val]
        cls_test = shuffled[n_train + n_val :]

        train_records.extend(cls_train)
        val_records.extend(cls_val)
        test_records.extend(cls_test)

        split_audit[cls_name] = {
            "total_recordings": n_total,
            "train_recordings": len(cls_train),
            "val_recordings": len(cls_val),
            "test_recordings": len(cls_test),
        }

    # Strict leakage validation: Verify intersection of unique recording identifiers is empty
    def get_rec_id(r):
        return r.get("relative_to_data_root") or f"{r['class_name']}/{r['file_name']}"

    train_ids = set(get_rec_id(r) for r in train_records)
    val_ids = set(get_rec_id(r) for r in val_records)
    test_ids = set(get_rec_id(r) for r in test_records)

    overlap_train_val = train_ids.intersection(val_ids)
    overlap_train_test = train_ids.intersection(test_ids)
    overlap_val_test = val_ids.intersection(test_ids)

    has_leakage = bool(overlap_train_val or overlap_train_test or overlap_val_test)
    if has_leakage:
        raise RuntimeError(
            f"CRITICAL: Recording leakage detected! "
            f"Train/Val overlap: {overlap_train_val}, "
            f"Train/Test overlap: {overlap_train_test}, "
            f"Val/Test overlap: {overlap_val_test}"
        )

    # Calculate class distribution in training set for class weights
    train_class_counts = {}
    for r in train_records:
        cls_name = r["class_name"]
        train_class_counts[cls_name] = train_class_counts.get(cls_name, 0) + 1

    total_train_recs = len(train_records)
    n_classes = len(CANONICAL_CLASSES)

    # Balanced class weights: total / (n_classes * count)
    class_weights = {}
    for cls_name in CANONICAL_CLASSES:
        cnt = train_class_counts.get(cls_name, 0)
        idx = CLASS_TO_IDX[cls_name]
        if cnt > 0:
            class_weights[idx] = float(total_train_recs / (n_classes * cnt))
        else:
            class_weights[idx] = 1.0

    manifest = {
        "random_seed": seed,
        "leakage_audit": {
            "has_leakage": False,
            "train_val_overlap_count": 0,
            "train_test_overlap_count": 0,
            "val_test_overlap_count": 0,
        },
        "summary": {
            "train_total_recordings": len(train_records),
            "val_total_recordings": len(val_records),
            "test_total_recordings": len(test_records),
            "total_recordings": len(train_records) + len(val_records) + len(test_records),
        },
        "class_audit": split_audit,
        "class_weights": class_weights,
        "splits": {
            "train": train_records,
            "val": val_records,
            "test": test_records,
        }
    }

    return manifest


def print_split_summary(manifest: Dict[str, Any]):
    """Prints a clean tabular summary of the recording-level split."""
    print("\n" + "=" * 65)
    print(" RECORDING-LEVEL DATASET SPLIT SUMMARY (ZERO LEAKAGE)")
    print("=" * 65)
    print(f"{'Class Name':<16} | {'Total':<6} | {'Train':<6} | {'Val':<6} | {'Test':<6}")
    print("-" * 65)
    for cls_name, aud in manifest["class_audit"].items():
        print(
            f"{cls_name:<16} | "
            f"{aud['total_recordings']:<6} | "
            f"{aud['train_recordings']:<6} | "
            f"{aud['val_recordings']:<6} | "
            f"{aud['test_recordings']:<6}"
        )
    print("-" * 65)
    summ = manifest["summary"]
    print(
        f"{'TOTALS':<16} | "
        f"{summ['total_recordings']:<6} | "
        f"{summ['train_total_recordings']:<6} | "
        f"{summ['val_total_recordings']:<6} | "
        f"{summ['test_total_recordings']:<6}"
    )
    print("=" * 65)
    print(f"[+] Recording Overlap Check: PASSED (Zero leakage confirmed)")
    print("[+] Computed Training Class Weights (to balance loss):")
    for idx_str, w in manifest["class_weights"].items():
        cls_name = CANONICAL_CLASSES[int(idx_str)]
        print(f"    * Class {idx_str} ({cls_name:<14}): weight = {w:.3f}")
    print("=" * 65 + "\n")


def run_split(metadata_csv: Optional[str] = None, output_json: Optional[str] = None):
    """Executes dataset splitting and saves manifest JSON."""
    config = load_config()
    meta_path = Path(metadata_csv or (PROJECT_ROOT / config["paths"]["metadata_csv"]))
    out_json_path = Path(output_json or (PROJECT_ROOT / config["paths"]["split_manifest"]))

    if not meta_path.is_file():
        raise FileNotFoundError(
            f"Metadata CSV not found at '{meta_path.resolve()}'. "
            f"Please run 'src/analyze_audio.py' first."
        )

    df = pd.read_csv(meta_path)
    split_cfg = config.get("split", {})
    manifest = create_recording_level_split(
        df,
        train_ratio=split_cfg.get("train_ratio", 0.70),
        val_ratio=split_cfg.get("val_ratio", 0.15),
        test_ratio=split_cfg.get("test_ratio", 0.15),
        seed=split_cfg.get("random_seed", 42)
    )

    save_json(manifest, out_json_path)
    print(f"[+] Dataset split manifest saved to: {out_json_path.resolve()}")
    print_split_summary(manifest)
    return manifest


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Create recording-level split to prevent data leakage.")
    parser.add_argument("--metadata_csv", type=str, default=None, help="Path to audio metadata CSV")
    parser.add_argument("--output_json", type=str, default=None, help="Path to save split manifest JSON")
    args = parser.parse_args()

    run_split(args.metadata_csv, args.output_json)
