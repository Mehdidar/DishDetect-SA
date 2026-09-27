"""
yolo_crop.py
------------
Uses YOLOv8n to detect the food/dish in each image and crop to the
bounding box before saving. This creates a new cleaned dataset where
MobileNet V3 sees ONLY the dish, not the background.

Usage:
    .venv/bin/python yolo_crop.py \
        --input  results/categorized_dataset \
        --output results/yolo_cropped_dataset

The output folder preserves the same veg/dish_name/ folder structure
as the input, so it works as a drop-in replacement for training.
"""

import os
import argparse
from pathlib import Path
from PIL import Image
import shutil

# COCO class IDs that are food or food-container related
FOOD_CLASSES = {
    41,  # cup
    42,  # fork
    43,  # knife
    44,  # spoon
    45,  # bowl
    46,  # banana
    47,  # apple
    48,  # sandwich
    49,  # orange
    50,  # broccoli
    51,  # carrot
    52,  # hot dog
    53,  # pizza
    54,  # donut
    55,  # cake
}

VALID_EXTS = {'.jpg', '.jpeg', '.png', '.bmp', '.webp'}


def crop_with_yolo(model, img_path, padding=0.10):
    """
    Run YOLO on one image. Returns a PIL crop of the best bounding box.
    
    Priority order:
      1. Largest food-class bounding box
      2. Largest bounding box of any class (if no food class found)
      3. Center crop fallback (if YOLO finds nothing at all)

    padding: fraction of bounding box size added as border on each side
    """
    img = Image.open(img_path).convert('RGB')
    W, H = img.size

    results = model(img_path, verbose=False)[0]
    boxes = results.boxes

    best_box = None
    best_area = -1
    food_best_box = None
    food_best_area = -1

    if boxes is not None and len(boxes) > 0:
        for box in boxes:
            cls_id = int(box.cls.item())
            x1, y1, x2, y2 = [c.item() for c in box.xyxy[0]]
            area = (x2 - x1) * (y2 - y1)

            if area > best_area:
                best_area = area
                best_box = (x1, y1, x2, y2)

            if cls_id in FOOD_CLASSES and area > food_best_area:
                food_best_area = area
                food_best_box = (x1, y1, x2, y2)

    chosen = food_best_box or best_box

    if chosen:
        x1, y1, x2, y2 = chosen
        bw, bh = x2 - x1, y2 - y1
        pad_x, pad_y = bw * padding, bh * padding
        x1 = max(0, x1 - pad_x)
        y1 = max(0, y1 - pad_y)
        x2 = min(W, x2 + pad_x)
        y2 = min(H, y2 + pad_y)
        crop = img.crop((x1, y1, x2, y2))
        used_fallback = False
    else:
        # Center crop fallback: take the central 80% of the image
        margin_x, margin_y = int(W * 0.10), int(H * 0.10)
        crop = img.crop((margin_x, margin_y, W - margin_x, H - margin_y))
        used_fallback = True

    return crop, used_fallback


def main():
    parser = argparse.ArgumentParser(description="YOLOv8 dish-cropping pre-processor")
    parser.add_argument("--input",   type=str, default="results/categorized_dataset",
                        help="Path to the categorized dataset (with veg/ non_veg/ subdirs)")
    parser.add_argument("--output",  type=str, default="results/yolo_cropped_dataset",
                        help="Where to save the cropped images")
    parser.add_argument("--model",   type=str, default="weights/yolov8n.pt",
                        help="YOLOv8 model variant (weights/yolov8n.pt is fastest)")
    parser.add_argument("--padding", type=float, default=0.10,
                        help="Padding fraction around bounding box (default 10%%)")
    parser.add_argument("--quality", type=int, default=92,
                        help="JPEG save quality (default 92)")
    args = parser.parse_args()

    # Import here so the file can be read without ultralytics installed
    from ultralytics import YOLO

    print(f"Loading YOLOv8 model: {args.model}")
    model = YOLO(args.model)

    input_root  = Path(args.input)
    output_root = Path(args.output)

    if not input_root.exists():
        raise FileNotFoundError(f"Input path not found: {input_root}")

    total = skipped = fallback = 0
    stats = {}

    # Walk every class folder (veg, non_veg)
    class_dirs = [d for d in input_root.iterdir() if d.is_dir()]
    for class_dir in sorted(class_dirs):
        class_name = class_dir.name
        stats[class_name] = {'total': 0, 'yolo': 0, 'fallback': 0}

        # Walk every dish subfolder
        for dish_dir in sorted(class_dir.iterdir()):
            if not dish_dir.is_dir():
                continue
            dish_name = dish_dir.name

            out_dish_dir = output_root / class_name / dish_name
            out_dish_dir.mkdir(parents=True, exist_ok=True)

            img_files = [f for f in dish_dir.iterdir()
                         if f.suffix.lower() in VALID_EXTS]

            for img_path in img_files:
                try:
                    crop, used_fallback = crop_with_yolo(model, str(img_path), args.padding)

                    # Save with same extension (convert to jpg for non-jpeg)
                    out_name = img_path.stem + '.jpg'
                    out_path = out_dish_dir / out_name
                    crop.save(str(out_path), 'JPEG', quality=args.quality)

                    total += 1
                    stats[class_name]['total'] += 1
                    if used_fallback:
                        fallback += 1
                        stats[class_name]['fallback'] += 1
                    else:
                        stats[class_name]['yolo'] += 1

                except Exception as e:
                    print(f"  ⚠️  Skipped {img_path.name}: {e}")
                    skipped += 1

    print("\n" + "=" * 55)
    print("YOLO CROPPING COMPLETE")
    print("=" * 55)
    print(f"  Total images processed: {total}")
    print(f"  YOLO-cropped (dish found): {total - fallback}")
    print(f"  Center-crop fallback:     {fallback}")
    print(f"  Skipped (errors):         {skipped}")
    print()
    for cls, s in stats.items():
        print(f"  {cls}: {s['total']} total  "
              f"({s['yolo']} YOLO + {s['fallback']} fallback)")
    print()
    print(f"  Cropped dataset saved to: {output_root}")
    print()
    print("Next step — re-split and retrain:")
    print(f"  .venv/bin/python split_dataset.py "
          f"--dataset {args.output} --model mobilenet_v3_large_yolo")
    print(f"  .venv/bin/python train.py "
          f"--dataset {args.output} --model mobilenet_v3_large_yolo "
          f"--use_three_way_split --img_size 224")


if __name__ == '__main__':
    main()
