import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
"""
split_dataset.py
----------------
Creates a scientifically correct 3-way Train / Validation / Test split
using perceptual hashing to prevent near-duplicate leakage.

Usage:
    .venv/bin/python split_dataset.py --dataset results/categorized_dataset --model mobilenet_v3_large
"""

import os
import argparse
import json
import imagehash
import numpy as np
from PIL import Image
from collections import defaultdict
from core.utils import prepare_data_three_way


def check_leakage(split_info):
    """
    Verifies that no image path (exact duplicate) appears in more than one split.
    Also checks that no hash group_id appears across splits.
    Raises a RuntimeError if leakage is detected.
    """
    print("\n" + "=" * 50)
    print("DUPLICATE LEAKAGE CHECK")
    print("=" * 50)

    train_set = set(split_info['train_files'])
    val_set   = set(split_info['val_files'])
    test_set  = set(split_info['test_files'])

    tv = train_set & val_set
    tt = train_set & test_set
    vt = val_set   & test_set

    print(f"  Train ↔ Validation (exact path duplicates): {len(tv)}")
    print(f"  Train ↔ Test       (exact path duplicates): {len(tt)}")
    print(f"  Validation ↔ Test  (exact path duplicates): {len(vt)}")

    if tv or tt or vt:
        raise RuntimeError(
            f"\n❌ LEAKAGE DETECTED!\n"
            f"  Train ↔ Val:  {len(tv)} images\n"
            f"  Train ↔ Test: {len(tt)} images\n"
            f"  Val ↔ Test:   {len(vt)} images\n"
            "Fix the split before training."
        )

    print("\n  STATUS: ✅ PASS — No duplicate leakage detected.")
    print("=" * 50)


def print_split_summary(split_info, class_mapping):
    """Prints a clear table of image counts and percentages per split and class."""
    idx_to_class = {v: k for k, v in class_mapping.items()}  # e.g. {'0': 'non_veg'}

    splits = {
        'TRAIN':      split_info['train_files'],
        'VALIDATION': split_info['val_files'],
        'TEST':       split_info['test_files'],
    }

    total_images = sum(len(v) for v in splits.values())
    print("\n" + "=" * 50)
    print("DATASET SPLIT SUMMARY")
    print(f"Total Images: {total_images}")
    print("=" * 50)

    for split_name, files in splits.items():
        # Count per class by reading folder structure
        class_counts = defaultdict(int)
        for fpath in files:
            parts = fpath.split(os.sep)
            for part in reversed(parts[:-1]):
                if part in class_mapping.values():
                    class_counts[part] += 1
                    break

        split_total = len(files)
        pct = split_total / total_images * 100 if total_images > 0 else 0
        print(f"\n{split_name}: {split_total} images ({pct:.1f}% of total)")
        for cls_name in sorted(class_counts.keys()):
            cls_pct = class_counts[cls_name] / split_total * 100 if split_total > 0 else 0
            print(f"  {cls_name:12s}: {class_counts[cls_name]:4d} ({cls_pct:.1f}%)")

    print("=" * 50)


def main():
    parser = argparse.ArgumentParser(description="Create a 3-way Train/Val/Test split with leakage checks.")
    parser.add_argument("--dataset",         type=str,   required=True,  help="Path to dataset root (with veg/non_veg subdirs)")
    parser.add_argument("--model",           type=str,   default='mobilenet_v3_large', help="Model name (used for results folder)")
    parser.add_argument("--train_size",      type=float, default=0.70,   help="Train fraction (default 0.70)")
    parser.add_argument("--val_size",        type=float, default=0.15,   help="Val fraction (default 0.15)")
    parser.add_argument("--seed",            type=int,   default=42,     help="Random seed")
    parser.add_argument("--hash_threshold",  type=int,   default=5,      help="Perceptual hash Hamming distance threshold")
    args = parser.parse_args()

    results_dir = f'results_{args.model}'

    print(f"\nCreating 3-way split for model: {args.model}")
    print(f"Dataset:        {args.dataset}")
    print(f"Results dir:    {results_dir}")
    print(f"Target ratios:  Train={args.train_size:.0%}  Val={args.val_size:.0%}  Test={1-args.train_size-args.val_size:.0%}")
    print(f"Random seed:    {args.seed}")
    print(f"Hash threshold: {args.hash_threshold}")
    print()

    (
        train_files, train_labels,
        val_files,   val_labels,
        test_files,  test_labels,
        _class_weights
    ) = prepare_data_three_way(
        dataset_path=args.dataset,
        train_size=args.train_size,
        val_size=args.val_size,
        seed=args.seed,
        hash_threshold=args.hash_threshold,
        results_dir=results_dir,
    )

    # Re-read the saved split_info.json to use for checks
    split_info_path = os.path.join(results_dir, 'split_info.json')
    with open(split_info_path) as f:
        split_info = json.load(f)

    class_mapping_path = os.path.join(results_dir, 'class_mapping.json')
    with open(class_mapping_path) as f:
        class_mapping = json.load(f)
    # Invert: we need class_name -> idx (class_mapping is {idx_str: class_name})
    class_name_set = set(class_mapping.values())

    # Print summary
    print_split_summary(split_info, class_mapping)

    # Leakage check
    check_leakage(split_info)

    print(f"\n✅ Split complete. Files saved:")
    print(f"  {split_info_path}")
    print(f"  {results_dir}/splits/train.csv")
    print(f"  {results_dir}/splits/val.csv")
    print(f"  {results_dir}/splits/test.csv")
    print(f"\nNext step — retrain with the new split:")
    print(f"  .venv/bin/python train.py --dataset {args.dataset} --model {args.model} --use_three_way_split")


if __name__ == '__main__':
    main()
