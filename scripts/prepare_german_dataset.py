"""
prepare_german_dataset.py
--------------------------
Remaps the german_dishes/ folder (which has categories: veg, non-veg, dairy,
sea-food, soup, drinks) into the standard pipeline structure:

    german_test_set/
        veg/
            <dish_name>/
                image1.jpg ...
        non_veg/
            <dish_name>/
                image1.jpg ...

Category → veg/non_veg mapping:
  veg       → veg
  dairy     → veg   (except Königsberger Klopse → non_veg)
  soup      → veg   (except Gulaschsuppe, Leberknödelsuppe → non_veg)
  non-veg   → non_veg
  sea-food  → non_veg
  drinks    → SKIP

Usage:
    .venv/bin/python scripts/prepare_german_dataset.py \
        --input  german_dishes \
        --output german_test_set
"""

import os
import shutil
import argparse
from pathlib import Path

# ── Category-level default mapping ───────────────────────────────────────────
CATEGORY_MAP = {
    "veg":      "veg",
    "dairy":    "veg",
    "soup":     "veg",
    "non-veg":  "non_veg",
    "sea-food": "non_veg",
    "drinks":   None,       # SKIP
}

# ── Dish-level overrides (case-insensitive match on folder name) ──────────────
DISH_OVERRIDES = {
    "königsberger klopse": "non_veg",   # meat dish in dairy folder
    "gulaschsuppe":        "non_veg",   # meat soup
    "leberknödelsuppe":    "non_veg",   # liver dumpling soup
}

VALID_EXTS = {'.jpg', '.jpeg', '.png', '.bmp', '.webp'}


def main():
    parser = argparse.ArgumentParser(
        description="Remap german_dishes/ to pipeline-ready veg/non_veg structure"
    )
    parser.add_argument("--input",  type=str, default="datasets/german_dishes",
                        help="Path to the german_dishes folder")
    parser.add_argument("--output", type=str, default="datasets/german_test_set",
                        help="Where to write the remapped dataset")
    parser.add_argument("--copy",   action="store_true", default=True,
                        help="Copy files (default). Use --no-copy to symlink instead.")
    args = parser.parse_args()

    input_root  = Path(args.input)
    output_root = Path(args.output)

    if not input_root.exists():
        raise FileNotFoundError(f"Input not found: {input_root}")

    stats = {"veg": 0, "non_veg": 0, "skipped": 0, "dishes": []}

    print(f"\n{'='*60}")
    print(f"  German Dishes → veg / non_veg Remapper")
    print(f"{'='*60}")
    print(f"  Input : {input_root.resolve()}")
    print(f"  Output: {output_root.resolve()}\n")

    for category_dir in sorted(input_root.iterdir()):
        if not category_dir.is_dir():
            continue

        category = category_dir.name
        default_label = CATEGORY_MAP.get(category)

        if default_label is None:
            print(f"  ⏭️   Skipping category: {category}/")
            for dish_dir in sorted(category_dir.iterdir()):
                if dish_dir.is_dir():
                    imgs = [f for f in dish_dir.iterdir() if f.suffix.lower() in VALID_EXTS]
                    stats["skipped"] += len(imgs)
            continue

        for dish_dir in sorted(category_dir.iterdir()):
            if not dish_dir.is_dir():
                continue

            dish_name = dish_dir.name
            # Check for a dish-level override
            label = DISH_OVERRIDES.get(dish_name.lower(), default_label)

            images = [f for f in dish_dir.iterdir() if f.suffix.lower() in VALID_EXTS]
            if not images:
                print(f"  ⚠️   No images found in {category}/{dish_name} — skipping")
                continue

            dest_dir = output_root / label / dish_name
            dest_dir.mkdir(parents=True, exist_ok=True)

            for img in images:
                dest_file = dest_dir / img.name
                if not dest_file.exists():
                    shutil.copy2(str(img), str(dest_file))

            stats[label] += len(images)
            override_note = " ⚡ (override)" if dish_name.lower() in DISH_OVERRIDES else ""
            print(f"  ✅  [{label:8s}] {category}/{dish_name}: {len(images)} images{override_note}")
            stats["dishes"].append({
                "dish": dish_name,
                "category": category,
                "label": label,
                "count": len(images)
            })

    total = stats["veg"] + stats["non_veg"]
    print(f"\n{'='*60}")
    print(f"  REMAPPING COMPLETE")
    print(f"{'='*60}")
    print(f"  🥬  veg    : {stats['veg']:4d} images  ({len([d for d in stats['dishes'] if d['label']=='veg'])} dishes)")
    print(f"  🥩  non_veg: {stats['non_veg']:4d} images  ({len([d for d in stats['dishes'] if d['label']=='non_veg'])} dishes)")
    print(f"  ⏭️   skipped: {stats['skipped']:4d} images  (drinks category)")
    print(f"  📦  total  : {total:4d} images")
    print(f"\n  Output written to: {output_root.resolve()}")
    print(f"\nNext step — run YOLO-seg crop:")
    print(f"  .venv/bin/python scripts/yolo_seg_crop.py \\")
    print(f"      --input  {output_root} \\")
    print(f"      --output german_test_set_cropped \\")
    print(f"      --model  yolov8n-seg.pt")


if __name__ == "__main__":
    main()
