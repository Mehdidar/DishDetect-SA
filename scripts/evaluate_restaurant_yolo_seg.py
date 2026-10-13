"""
evaluate_restaurant_yolo_seg.py
-------------------------------
Evaluates the YOLO-segmented MobileNetV3-Large model on the 24 meat-free real-world restaurant photos.
Tests two modes:
  1. YOLOv8-seg Crop Pipeline (YOLOv8-seg polygon mask crop -> MobileNetV3)
  2. Direct Image (Raw uncropped -> MobileNetV3)

Saves detailed prediction CSVs and prints per-dish comparison.
"""

import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import csv
import json
import torch
import numpy as np
from PIL import Image
from ultralytics import YOLO

from core.models import FoodClassifier
from core.utils import get_device, get_transforms

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
RESTAURANT_DIR = os.path.join(ROOT, "datasets", "Restaurant testing data set")
CHECKPOINT_PATH = os.path.join(ROOT, "checkpoints", "mobilenet_v3_large_vegan_veg_yolo_seg_best.pth")
OUT_DIR = os.path.join(ROOT, "results_mobilenet_v3_large_vegan_veg_yolo_seg", "restaurant_evaluation")

RESTAURANT_MEATFREE_TRUTH = {
    "hummus": "vegan",
    "okra tagine": "vegan",
    "tamees bread": "vegan",
    "zaatara manaqish": "vegan",
    "cheese fatayer": "vegetarian",
}

def get_mask_bbox(mask_xy, img_w, img_h):
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

def crop_with_yolo_seg(yolo_model, img_path, padding=0.08):
    img = Image.open(img_path).convert('RGB')
    W, H = img.size
    results = yolo_model(img_path, verbose=False)[0]

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

def evaluate_pipeline(use_yolo_crop=True):
    os.makedirs(OUT_DIR, exist_ok=True)
    device = get_device()

    mapping = {"0": "vegan", "1": "vegetarian"}

    model = FoodClassifier(num_classes=2, freeze_backbone=False, architecture='mobilenet_v3_large')
    model.load_state_dict(torch.load(CHECKPOINT_PATH, map_location=device))
    model = model.to(device)
    model.eval()

    _, eval_transform = get_transforms(224)

    yolo_model = YOLO("yolov8n-seg.pt") if use_yolo_crop else None

    valid_exts = {".jpg", ".jpeg", ".png", ".webp"}
    results = []
    dish_stats = {}
    total_images, correct_images = 0, 0

    mode_name = "YOLOv8-Seg Cropped" if use_yolo_crop else "Raw Uncropped"

    for dish_folder, true_label in RESTAURANT_MEATFREE_TRUTH.items():
        folder_path = os.path.join(RESTAURANT_DIR, dish_folder)
        if not os.path.exists(folder_path):
            continue

        dish_stats[dish_folder] = {"total": 0, "correct": 0, "true_label": true_label}

        for root, _, files in os.walk(folder_path):
            for file in sorted(files):
                if os.path.splitext(file)[1].lower() in valid_exts:
                    img_path = os.path.join(root, file)
                    try:
                        crop_method = "direct"
                        if use_yolo_crop:
                            img_crop, crop_method = crop_with_yolo_seg(yolo_model, img_path)
                            tensor = eval_transform(img_crop).unsqueeze(0).to(device)
                        else:
                            with Image.open(img_path) as img:
                                tensor = eval_transform(img.convert("RGB")).unsqueeze(0).to(device)

                        with torch.no_grad():
                            outputs = model(tensor)
                            probs = torch.softmax(outputs, dim=1)[0]
                            pred_idx = torch.argmax(probs).item()
                            conf = probs[pred_idx].item()

                        pred_label = mapping[str(pred_idx)]
                        is_correct = (pred_label == true_label)

                        total_images += 1
                        dish_stats[dish_folder]["total"] += 1
                        if is_correct:
                            correct_images += 1
                            dish_stats[dish_folder]["correct"] += 1

                        results.append({
                            "dish": dish_folder,
                            "file": file,
                            "true_label": true_label,
                            "pred_label": pred_label,
                            "correct": is_correct,
                            "confidence": f"{conf:.4f}",
                            "crop_method": crop_method
                        })
                    except Exception as e:
                        print(f"Error {img_path}: {e}")

    overall_acc = (correct_images / max(total_images, 1)) * 100

    print("\n" + "=" * 70)
    print(f"RESTAURANT EVALUATION: {mode_name.upper()} PIPELINE")
    print("=" * 70)
    print(f"Total Restaurant Images: {total_images}")
    print(f"Correct Predictions:     {correct_images}")
    print(f"Overall Accuracy:        {overall_acc:.2f}%")
    print("-" * 70)
    print(f"{'Dish Name':<20} | {'True Category':<13} | {'Correct':<8} | {'Accuracy':<8}")
    print("-" * 70)
    for dish, s in dish_stats.items():
        dish_acc = (s["correct"] / max(s["total"], 1)) * 100
        print(f"{dish:<20} | {s['true_label']:<13} | {s['correct']}/{s['total']:<6} | {dish_acc:.1f}%")

    prefix = "restaurant_yolo_seg" if use_yolo_crop else "restaurant_raw"
    csv_out = os.path.join(OUT_DIR, f"{prefix}_predictions.csv")
    with open(csv_out, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["dish", "file", "true_label", "pred_label", "correct", "confidence", "crop_method"])
        writer.writeheader()
        writer.writerows(results)
    print(f"Saved predictions to: {csv_out}")
    return overall_acc, results, dish_stats

if __name__ == "__main__":
    print("Testing Pipeline 1: YOLOv8-Seg Cropped...")
    acc_crop, _, _ = evaluate_pipeline(use_yolo_crop=True)

    print("\nTesting Pipeline 2: Direct Raw Uncropped...")
    acc_raw, _, _ = evaluate_pipeline(use_yolo_crop=False)
