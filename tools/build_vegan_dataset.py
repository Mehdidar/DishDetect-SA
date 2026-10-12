"""
build_vegan_dataset.py
Constructs a dedicated Vegan vs Non-Vegan dataset directory structure
from the verified 25 Arabic dishes.
Uses symlinks (or copies) to preserve disk space while establishing
a clean binary dataset:
    datasets/vegan_dataset_v1/
        ├── vegan/
        └── non_vegan/
"""

import os
import shutil
import json

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
TAXONOMY_PATH = os.path.join(ROOT, "core", "vegan_taxonomy.json")
SRC_DATASET = os.path.join(ROOT, "datasets", "categorized_dataset")
DEST_DATASET = os.path.join(ROOT, "datasets", "vegan_dataset_v1")

def build_vegan_dataset():
    if not os.path.exists(TAXONOMY_PATH):
        raise FileNotFoundError(f"Missing {TAXONOMY_PATH}")

    with open(TAXONOMY_PATH) as f:
        taxonomy = json.load(f)

    os.makedirs(os.path.join(DEST_DATASET, "vegan"), exist_ok=True)
    os.makedirs(os.path.join(DEST_DATASET, "non_vegan"), exist_ok=True)

    valid_exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    stats = {"vegan": 0, "non_vegan": 0, "per_dish": {}}

    for dish_name, info in taxonomy.items():
        target_class = info["new_diet"]  # 'vegan' or 'non_vegan'
        legacy_dir = info["legacy_diet"]  # 'veg' or 'non_veg'
        folder_alias = info.get("folder_alias", dish_name)

        # Locate source folder
        possible_sources = [
            os.path.join(SRC_DATASET, legacy_dir, folder_alias),
            os.path.join(SRC_DATASET, legacy_dir, dish_name),
            os.path.join(SRC_DATASET, legacy_dir, dish_name.replace(" ", "_")),
            os.path.join(SRC_DATASET, legacy_dir, dish_name.replace("_", " ")),
        ]

        src_dish_dir = None
        for p in possible_sources:
            if os.path.exists(p):
                src_dish_dir = p
                break

        if not src_dish_dir:
            print(f"⚠️  Warning: Could not find source directory for {dish_name}")
            continue

        dest_dish_dir = os.path.join(DEST_DATASET, target_class, dish_name.replace(" ", "_"))
        os.makedirs(dest_dish_dir, exist_ok=True)

        dish_count = 0
        for root, _, files in os.walk(src_dish_dir):
            for file in files:
                if os.path.splitext(file)[1].lower() in valid_exts:
                    src_file = os.path.join(root, file)
                    dest_file = os.path.join(dest_dish_dir, file)
                    if not os.path.exists(dest_file):
                        try:
                            # Try symlink first for speed and 0 disk overhead
                            os.symlink(src_file, dest_file)
                        except (OSError, AttributeError):
                            shutil.copy2(src_file, dest_file)
                    dish_count += 1

        stats[target_class] += dish_count
        stats["per_dish"][dish_name] = {"class": target_class, "count": dish_count}
        print(f"[{target_class.upper():<9}] Linked {dish_count:3d} images for {dish_name}")

    print("\n" + "=" * 60)
    print("VEGAN DATASET V1 GENERATION COMPLETE")
    print("=" * 60)
    print(f"Destination:     {DEST_DATASET}")
    print(f"Total Vegan:     {stats['vegan']} images across 9 dishes")
    print(f"Total Non-Vegan: {stats['non_vegan']} images across 16 dishes")
    print(f"Total Combined:  {stats['vegan'] + stats['non_vegan']} images across 25 dishes")

    # Save manifest
    with open(os.path.join(DEST_DATASET, "manifest.json"), "w") as f:
        json.dump(stats, f, indent=4)

if __name__ == "__main__":
    build_vegan_dataset()
