"""
predict_menu.py
---------------
Single-image or batch inference for restaurant menu images.
Runs the full pipeline: YOLOv8-seg crop → MobileNetV3-Large classification.

Usage (single image):
    .venv/bin/python scripts/predict_menu.py \
        --image path/to/schnitzel.jpg

Usage (batch folder — no labels needed):
    .venv/bin/python scripts/predict_menu.py \
        --folder path/to/menu_images/ \
        --output results/menu_predictions.csv

Usage (evaluate a labelled folder — veg/ non_veg/ subfolders):
    .venv/bin/python scripts/predict_menu.py \
        --folder german_test_set_cropped \
        --labelled \
        --output results/german_menu_eval.csv
"""

import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import csv
import json
import argparse
import torch
import torch.nn.functional as F
from pathlib import Path
from PIL import Image

from core.models import FoodClassifier
from core.utils import get_device, get_transforms

VALID_EXTS = {'.jpg', '.jpeg', '.png', '.bmp', '.webp'}

MODEL_NAME   = "mobilenet_v3_large_yolo_seg"
CHECKPOINT   = "checkpoints/mobilenet_v3_large_yolo_seg_best.pth"
CLASS_MAP    = {"0": "non_veg", "1": "veg"}
IMG_SIZE     = 288

# ── YOLO crop (mirrors yolo_seg_crop.py logic) ────────────────────────────────
def get_mask_bbox(mask_xy, img_w, img_h):
    import numpy as np
    if mask_xy is None or len(mask_xy) == 0:
        return None
    pts = np.array(mask_xy, dtype=np.float32).reshape(-1, 2)
    x1, y1 = pts[:, 0].min(), pts[:, 1].min()
    x2, y2 = pts[:, 0].max(), pts[:, 1].max()
    x1 = max(0.0, float(x1)); y1 = max(0.0, float(y1))
    x2 = min(float(img_w), float(x2)); y2 = min(float(img_h), float(y2))
    if x2 <= x1 or y2 <= y1:
        return None
    return (x1, y1, x2, y2)


def crop_with_yolo_seg(model, img_path, padding=0.08):
    img = Image.open(img_path).convert('RGB')
    W, H = img.size
    results = model(str(img_path), verbose=False)[0]

    chosen_box = None
    method = 'center_crop'

    if results.masks is not None and len(results.masks) > 0:
        best_area = -1
        for mask in results.masks:
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

    if chosen_box is None and results.boxes is not None and len(results.boxes) > 0:
        best_area = -1
        for box in results.boxes:
            x1, y1, x2, y2 = [c.item() for c in box.xyxy[0]]
            area = (x2 - x1) * (y2 - y1)
            if area > best_area:
                best_area = area
                chosen_box = (x1, y1, x2, y2)
                method = 'det_box'

    if chosen_box is not None:
        x1, y1, x2, y2 = chosen_box
        bw, bh = x2 - x1, y2 - y1
        pad_x = bw * padding; pad_y = bh * padding
        x1 = max(0.0, x1 - pad_x); y1 = max(0.0, y1 - pad_y)
        x2 = min(float(W), x2 + pad_x); y2 = min(float(H), y2 + pad_y)
        crop = img.crop((x1, y1, x2, y2))
    else:
        mx = int(W * 0.10); my = int(H * 0.10)
        crop = img.crop((mx, my, W - mx, H - my))

    return crop, method


# ── Model loader ──────────────────────────────────────────────────────────────
def load_model(device):
    model = FoodClassifier(num_classes=2, freeze_backbone=False,
                           architecture=MODEL_NAME)
    ckpt = torch.load(CHECKPOINT, map_location=device)
    model.load_state_dict(ckpt)
    model = model.to(device)
    model.eval()
    return model


# ── Single image predict ──────────────────────────────────────────────────────
def predict_one(img_path, classifier, yolo_model, transform, device, threshold=0.65):
    crop, method = crop_with_yolo_seg(yolo_model, str(img_path))
    tensor = transform(crop).unsqueeze(0).to(device)

    with torch.no_grad():
        outputs = classifier(tensor)
        probs = F.softmax(outputs, dim=1)
        conf, pred_idx = torch.max(probs, 1)

    conf_val   = conf.item()
    pred_class = CLASS_MAP[str(pred_idx.item())]
    uncertain  = conf_val < threshold

    return {
        "file":       str(img_path),
        "prediction": pred_class,
        "confidence": round(conf_val, 4),
        "uncertain":  uncertain,
        "yolo_method": method,
    }


# ── Main ─────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(
        description="Restaurant menu image classifier (YOLO-seg + MobileNetV3)"
    )
    parser.add_argument("--image",     type=str, default=None,
                        help="Path to a single image")
    parser.add_argument("--folder",    type=str, default=None,
                        help="Path to a folder of images (flat or veg/non_veg subfolders)")
    parser.add_argument("--labelled",  action="store_true",
                        help="Folder has veg/ non_veg/ subfolders → compute accuracy")
    parser.add_argument("--output",    type=str, default=None,
                        help="Save results to this CSV file")
    parser.add_argument("--threshold", type=float, default=0.65,
                        help="Confidence threshold below which prediction is UNCERTAIN (default 0.65)")
    parser.add_argument("--yolo_model",type=str, default="weights/yolov8n-seg.pt",
                        help="YOLOv8 segmentation model path")
    parser.add_argument("--no_yolo",   action="store_true",
                        help="Skip YOLO crop — classify raw images directly")
    args = parser.parse_args()

    if args.image is None and args.folder is None:
        parser.error("Provide --image or --folder")

    device = get_device()
    _, transform = get_transforms(IMG_SIZE)
    classifier = load_model(device)

    # Load YOLO
    yolo_model = None
    if not args.no_yolo:
        from ultralytics import YOLO
        print(f"  Loading YOLO: {args.yolo_model}")
        yolo_model = YOLO(args.yolo_model)

    results = []

    # ── Single image mode ──────────────────────────────────────────────────
    if args.image:
        img_path = Path(args.image)
        if not img_path.exists():
            print(f"❌ Image not found: {img_path}")
            return

        if args.no_yolo:
            img = Image.open(img_path).convert('RGB')
            tensor = transform(img).unsqueeze(0).to(device)
            with torch.no_grad():
                outputs = classifier(tensor)
                probs = F.softmax(outputs, dim=1)
                conf, pred_idx = torch.max(probs, 1)
            r = {"file": str(img_path), "prediction": CLASS_MAP[str(pred_idx.item())],
                 "confidence": round(conf.item(), 4), "uncertain": conf.item() < args.threshold,
                 "yolo_method": "none"}
        else:
            r = predict_one(img_path, classifier, yolo_model, transform, device, args.threshold)

        emoji = "🥬" if r["prediction"] == "veg" else "🥩"
        uncertain_tag = "  ⚠️  UNCERTAIN" if r["uncertain"] else ""
        print(f"\n  🍽️  Image    : {img_path.name}")
        print(f"  🔍 YOLO Crop : {r['yolo_method']}")
        print(f"  {emoji} Prediction: {r['prediction'].upper()}{uncertain_tag}")
        print(f"  📊 Confidence: {r['confidence']*100:.1f}%\n")
        results.append(r)

    # ── Folder / batch mode ────────────────────────────────────────────────
    elif args.folder:
        folder = Path(args.folder)
        if not folder.exists():
            print(f"❌ Folder not found: {folder}")
            return

        # Collect images + optional labels
        image_entries = []  # (path, true_label or None)

        if args.labelled:
            for class_dir in sorted(folder.iterdir()):
                if not class_dir.is_dir():
                    continue
                cls = class_dir.name  # "veg" or "non_veg"
                for sub in sorted(class_dir.iterdir()):
                    if sub.is_dir():
                        imgs = [f for f in sub.iterdir() if f.suffix.lower() in VALID_EXTS]
                    else:
                        imgs = [sub] if sub.suffix.lower() in VALID_EXTS else []
                    for img in imgs:
                        image_entries.append((img, cls))
        else:
            for img in sorted(folder.rglob("*")):
                if img.suffix.lower() in VALID_EXTS:
                    image_entries.append((img, None))

        total = len(image_entries)
        print(f"\n  🍽️  Running on {total} images from: {folder}\n")

        correct = 0
        uncertain_count = 0

        for i, (img_path, true_label) in enumerate(image_entries, 1):
            if args.no_yolo:
                img = Image.open(img_path).convert('RGB')
                tensor = transform(img).unsqueeze(0).to(device)
                with torch.no_grad():
                    outputs = classifier(tensor)
                    probs = F.softmax(outputs, dim=1)
                    conf, pred_idx = torch.max(probs, 1)
                r = {"file": str(img_path), "prediction": CLASS_MAP[str(pred_idx.item())],
                     "confidence": round(conf.item(), 4), "uncertain": conf.item() < args.threshold,
                     "yolo_method": "none", "true_label": true_label}
            else:
                r = predict_one(img_path, classifier, yolo_model, transform, device, args.threshold)
                r["true_label"] = true_label

            if true_label:
                is_correct = r["prediction"] == true_label
                r["correct"] = is_correct
                if is_correct:
                    correct += 1
            if r["uncertain"]:
                uncertain_count += 1

            emoji = "🥬" if r["prediction"] == "veg" else "🥩"
            tick  = "✅" if r.get("correct", True) else "❌"
            unc   = " ⚠️" if r["uncertain"] else ""
            print(f"  [{i:4d}/{total}] {tick} {emoji} {r['prediction']:8s} ({r['confidence']*100:5.1f}%){unc}  {img_path.name}")
            results.append(r)

        # Summary
        print(f"\n{'='*60}")
        print(f"  BATCH RESULTS SUMMARY")
        print(f"{'='*60}")
        print(f"  Total images  : {total}")
        if args.labelled:
            print(f"  Correct       : {correct}  ({correct/total*100:.1f}%)")
            print(f"  Incorrect     : {total - correct}  ({(total-correct)/total*100:.1f}%)")
        print(f"  Uncertain     : {uncertain_count}  (confidence < {args.threshold:.0%})")
        veg_count    = sum(1 for r in results if r["prediction"] == "veg")
        nonveg_count = sum(1 for r in results if r["prediction"] == "non_veg")
        print(f"  🥬 veg        : {veg_count}")
        print(f"  🥩 non_veg    : {nonveg_count}")

    # ── Save CSV ───────────────────────────────────────────────────────────
    if args.output and results:
        out_path = Path(args.output)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        fieldnames = ["file", "true_label", "prediction", "correct", "confidence", "uncertain", "yolo_method"]
        with open(out_path, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction='ignore')
            writer.writeheader()
            writer.writerows(results)
        print(f"\n  💾 Results saved to: {out_path}")


if __name__ == "__main__":
    main()
