"""
build_vegan_vs_veg_dataset.py
Builds a dedicated dataset for:
    Vegan (100% Plant-Based) vs Vegetarian (Contains Dairy/Eggs/Ghee)
Excludes all meat, poultry, fish, and seafood dishes completely.
Target directory: datasets/vegan_vs_veg/
    ├── vegan/        (9 plant-based dishes)
    └── vegetarian/   (6 lacto-ovo vegetarian dishes with dairy/eggs/ghee)
"""

import os
import shutil
import json

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
TAXONOMY_PATH = os.path.join(ROOT, "core", "vegan_taxonomy.json")
SRC_DATASET = os.path.join(ROOT, "datasets", "categorized_dataset")
DEST_DATASET = os.path.join(ROOT, "datasets", "vegan_vs_veg")

# Explicit definition of the 15 meatless dishes
VEGAN_DISHES = [
    "hummus",
    "pomegranate_fattoush",
    "za’atar_manakish",
    "spinach_fatayer",
    "tamees_bread",
    "okra_tagine",
    "spinach_stew",
    "sesame_brittle",
    "onion_soup",
]

VEGETARIAN_DISHES = [
    "fatayer cheese",
    "rice_pudding",
    "saffron_cake",
    "tamriya",
    "pistachio_tamriya",
    "pistachio_sweet",
]

def build():
    os.makedirs(os.path.join(DEST_DATASET, "vegan"), exist_ok=True)
    os.makedirs(os.path.join(DEST_DATASET, "vegetarian"), exist_ok=True)

    valid_exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    stats = {"vegan": 0, "vegetarian": 0, "dishes": {}}

    print("=" * 65)
    print("BUILDING VEGAN VS VEGETARIAN (MEAT-FREE ONLY) DATASET")
    print("=" * 65)

    all_targets = [("vegan", d) for d in VEGAN_DISHES] + [("vegetarian", d) for d in VEGETARIAN_DISHES]

    for target_class, dish_name in all_targets:
        # In categorized_dataset, all 15 dishes are under 'veg'
        possible_sources = [
            os.path.join(SRC_DATASET, "veg", dish_name),
            os.path.join(SRC_DATASET, "veg", dish_name.replace(" ", "_")),
            os.path.join(SRC_DATASET, "veg", dish_name.replace("_", " ")),
        ]

        src_dir = None
        for p in possible_sources:
            if os.path.exists(p):
                src_dir = p
                break

        if not src_dir:
            print(f"⚠️  Directory not found for {dish_name}")
            continue

        dest_dir = os.path.join(DEST_DATASET, target_class, dish_name.replace(" ", "_"))
        os.makedirs(dest_dir, exist_ok=True)

        dish_count = 0
        for root, _, files in os.walk(src_dir):
            for file in files:
                if os.path.splitext(file)[1].lower() in valid_exts:
                    src_file = os.path.join(root, file)
                    dest_file = os.path.join(dest_dir, file)
                    if not os.path.exists(dest_file):
                        try:
                            os.symlink(src_file, dest_file)
                        except (OSError, AttributeError):
                            shutil.copy2(src_file, dest_file)
                    dish_count += 1

        stats[target_class] += dish_count
        stats["dishes"][dish_name] = {"category": target_class, "count": dish_count}
        print(f"[{target_class.upper():<10}] Linked {dish_count:3d} images for {dish_name}")

    print("-" * 65)
    print(f"TOTAL VEGAN:      {stats['vegan']} images (9 dishes)")
    print(f"TOTAL VEGETARIAN: {stats['vegetarian']} images (6 dishes)")
    print(f"TOTAL COMBINED:   {stats['vegan'] + stats['vegetarian']} images (15 meat-free dishes)")
    print(f"CLASS RATIO:      {stats['vegan'] / max(stats['vegetarian'], 1):.2f} : 1  (Nearly perfectly balanced!)")

    with open(os.path.join(DEST_DATASET, "manifest.json"), "w") as f:
        json.dump(stats, f, indent=4)
    print(f"Saved manifest to: {os.path.join(DEST_DATASET, 'manifest.json')}")

if __name__ == "__main__":
    build()
