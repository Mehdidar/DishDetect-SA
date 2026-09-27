"""
yolo_seg_crop.py
----------------
Uses YOLOv8n-seg (segmentation model) to extract a tight crop around
the food item using the segmentation MASK polygon instead of the loose
bounding box used in yolo_crop.py.

Priority order per image:
  1. Tightest bounding box derived from segmentation MASK polygon
  2. Largest detection bounding box (if no seg mask)
  3. Center crop fallback (80% of image, if YOLO finds nothing)

Usage:
    # Crop training data
    .venv/bin/python scripts/yolo_seg_crop.py \
        --input  results/categorized_dataset \
        --output results/yolo_seg_dataset \
        --model  yolov8n-seg.pt

    # Crop test data
    .venv/bin/python scripts/yolo_seg_crop.py \
        --input  test_dataset_dishes \
        --output results/yolo_seg_test \
        --model  yolov8n-seg.pt
"""

import os
import argparse
import numpy as np
from pathlib import Path
from PIL import Image

VALID_EXTS = {'.jpg', '.jpeg', '.png', '.bmp', '.webp'}


def get_mask_bbox(mask_xy, img_w, img_h):
    """
    Convert a list of (x, y) polygon points (from YOLO seg) into
    a tight axis-aligned bounding box (x1, y1, x2, y2).
    Returns None if the point list is empty.
    """
    if mask_xy is None or len(mask_xy) == 0:
        return None
    pts = np.array(mask_xy, dtype=np.float32).reshape(-1, 2)
    x1, y1 = pts[:, 0].min(), pts[:, 1].min()
    x2, y2 = pts[:, 0].max(), pts[:, 1].max()
    # Clamp to image bounds
    x1 = max(0.0, float(x1))
    y1 = max(0.0, float(y1))
    x2 = min(float(img_w), float(x2))
    y2 = min(float(img_h), float(y2))
    if x2 <= x1 or y2 <= y1:
        return None
    return (x1, y1, x2, y2)


def crop_with_yolo_seg(model, img_path, padding=0.08):
    """
    Run YOLOv8-seg on one image.
    Returns (PIL crop, used_fallback: bool, method: str).

    method is one of: 'seg_mask', 'det_box', 'center_crop'
    """
    img = Image.open(img_path).convert('RGB')
    W, H = img.size

    results = model(img_path, verbose=False)[0]

    chosen_box = None
    method = 'center_crop'

    # ── 1. Try segmentation masks first ──────────────────────────────────
    if results.masks is not None and len(results.masks) > 0:
        best_area = -1
        for mask in results.masks:
            # mask.xy is a list of numpy arrays (one per polygon)
            for polygon in mask.xy:
                bbox = get_mask_bbox(polygon, W, H)
                if bbox is None:
                    continue
                x1, y1, x2, y2 = bbox
                area = (x2 - x1) * (y2 - y1)
                if area > best_area:
                    best_area = area
                    chosen_box = bbox
                    method = 'seg_mask'

    # ── 2. Fall back to detection bounding boxes ──────────────────────────
    if chosen_box is None and results.boxes is not None and len(results.boxes) > 0:
        best_area = -1
        for box in results.boxes:
            x1, y1, x2, y2 = [c.item() for c in box.xyxy[0]]
            area = (x2 - x1) * (y2 - y1)
            if area > best_area:
                best_area = area
                chosen_box = (x1, y1, x2, y2)
                method = 'det_box'

    # ── 3. Apply padding and crop ─────────────────────────────────────────
    if chosen_box is not None:
        x1, y1, x2, y2 = chosen_box
        bw, bh = x2 - x1, y2 - y1
        pad_x = bw * padding
        pad_y = bh * padding
        x1 = max(0.0, x1 - pad_x)
        y1 = max(0.0, y1 - pad_y)
        x2 = min(float(W), x2 + pad_x)
        y2 = min(float(H), y2 + pad_y)
        crop = img.crop((x1, y1, x2, y2))
        used_fallback = False
    else:
        # Center crop: keep central 80%
        margin_x = int(W * 0.10)
        margin_y = int(H * 0.10)
        crop = img.crop((margin_x, margin_y, W - margin_x, H - margin_y))
        used_fallback = True

    return crop, used_fallback, method


def main():
    parser = argparse.ArgumentParser(
        description="YOLOv8-Seg food cropper — uses segmentation masks for tighter crops"
    )
    parser.add_argument("--input",   type=str, required=True,
                        help="Path to input dataset (with veg/ non_veg/ subdirs, or flat class dirs)")
    parser.add_argument("--output",  type=str, required=True,
                        help="Where to save cropped images (preserves folder structure)")
    parser.add_argument("--model",   type=str, default="weights/yolov8n-seg.pt",
                        help="YOLOv8 segmentation model (default: weights/yolov8n-seg.pt — auto-downloaded)")
    parser.add_argument("--padding", type=float, default=0.08,
                        help="Padding fraction around the seg bbox (default 8%%)")
    parser.add_argument("--quality", type=int, default=95,
                        help="JPEG save quality (default 95 — higher than old script)")
    args = parser.parse_args()

    # Import here so the file can be read/imported without ultralytics
    from ultralytics import YOLO

    print(f"Loading YOLO model: {args.model}  (auto-downloads if not found)")
    model = YOLO(args.model)

    input_root  = Path(args.input)
    output_root = Path(args.output)

    if not input_root.exists():
        raise FileNotFoundError(f"Input path not found: {input_root}")

    total = skipped = fallback = det_box = seg_mask_count = 0
    stats = {}

    # ── Walk the directory tree ───────────────────────────────────────────
    # Supports both:
    #   input/veg/dish_name/*.jpg          (two-level: class → dish)
    #   input/class_name/*.jpg             (one-level: class only)
    all_image_files = []

    top_dirs = sorted([d for d in input_root.iterdir() if d.is_dir()])
    for top_dir in top_dirs:
        top_name = top_dir.name
        stats[top_name] = {'total': 0, 'seg_mask': 0, 'det_box': 0, 'fallback': 0}

        sub_dirs = sorted([d for d in top_dir.iterdir() if d.is_dir()])
        if sub_dirs:
            # Two-level: veg/dish_name/*.jpg
            for sub_dir in sub_dirs:
                imgs = [f for f in sub_dir.iterdir() if f.suffix.lower() in VALID_EXTS]
                for img in imgs:
                    rel = img.relative_to(input_root)
                    all_image_files.append((img, top_name, rel))
        else:
            # One-level: class_name/*.jpg
            imgs = [f for f in top_dir.iterdir() if f.suffix.lower() in VALID_EXTS]
            for img in imgs:
                rel = img.relative_to(input_root)
                all_image_files.append((img, top_name, rel))

    print(f"\nFound {len(all_image_files)} images in '{input_root}'")
    print(f"Output → '{output_root}'\n")

    for img_path, top_class, rel_path in all_image_files:
        try:
            crop, used_fallback, method = crop_with_yolo_seg(
                model, str(img_path), args.padding
            )

            out_path = output_root / rel_path.parent / (img_path.stem + '.jpg')
            out_path.parent.mkdir(parents=True, exist_ok=True)
            crop.save(str(out_path), 'JPEG', quality=args.quality)

            total += 1
            stats[top_class]['total'] += 1

            if used_fallback:
                fallback += 1
                stats[top_class]['fallback'] += 1
            elif method == 'seg_mask':
                seg_mask_count += 1
                stats[top_class]['seg_mask'] += 1
            else:
                det_box += 1
                stats[top_class]['det_box'] += 1

        except Exception as e:
            print(f"  ⚠️  Skipped {img_path.name}: {e}")
            skipped += 1

    # ── Summary ───────────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("YOLO-SEG CROPPING COMPLETE")
    print("=" * 60)
    print(f"  Total processed:       {total}")
    print(f"  Seg-mask crops:        {seg_mask_count}  ✅ (tightest)")
    print(f"  Detection-box crops:   {det_box}  ⚠️  (looser)")
    print(f"  Center-crop fallback:  {fallback}  ℹ️  (no detection)")
    print(f"  Skipped (errors):      {skipped}")
    print()
    for cls, s in stats.items():
        print(f"  {cls}: {s['total']} imgs  "
              f"(seg={s['seg_mask']}  det={s['det_box']}  fallback={s['fallback']})")
    print()
    print(f"  Cropped dataset saved to: {output_root}")
    print()
    print("Next steps:")
    print(f"  # Train:")
    print(f"  .venv/bin/python scripts/train.py \\")
    print(f"      --dataset {output_root} \\")
    print(f"      --model   mobilenet_v3_large_yolo_seg \\")
    print(f"      --img_size 288")
    print()
    print(f"  # Evaluate on external test set:")
    print(f"  .venv/bin/python scripts/evaluate_external.py \\")
    print(f"      --model    mobilenet_v3_large_yolo_seg \\")
    print(f"      --test_dir results/yolo_seg_test \\")
    print(f"      --img_size 288")


if __name__ == '__main__':
    main()
