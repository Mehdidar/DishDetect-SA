"""
evaluate_restaurant.py
----------------------
Evaluates the Vegan vs Vegetarian (Meat-Free) model on real-world restaurant photos.
Supports both:
  1. YOLOv8-Seg Cropping Pipeline (95.83% accuracy)
  2. Uncropped Baseline Pipeline (83.33% accuracy)

Usage:
    # Run YOLOv8-seg pipeline (Recommended - 95.83% accuracy):
    .venv/bin/python Vegan_vs_Vegetarian_Classifier/scripts/evaluate_restaurant.py --yolo_seg

    # Run uncropped baseline (83.33% accuracy):
    .venv/bin/python Vegan_vs_Vegetarian_Classifier/scripts/evaluate_restaurant.py --baseline

    # Run both and display comparative benchmark:
    .venv/bin/python Vegan_vs_Vegetarian_Classifier/scripts/evaluate_restaurant.py --all
"""

import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import csv
import json
import argparse
import torch
import numpy as np
from PIL import Image
from core.models import VeganFoodClassifier
from core.utils import get_device, get_transforms

PACKAGE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
ROOT = os.path.abspath(os.path.join(PACKAGE_DIR, ".."))
RESTAURANT_DIR = os.path.join(ROOT, "datasets", "Restaurant testing data set")
OUT_DIR = os.path.join(PACKAGE_DIR, "results", "restaurant_evaluation")

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

def run_evaluation(use_yolo_seg=True):
    os.makedirs(OUT_DIR, exist_ok=True)
    device = get_device()

    mapping_path = os.path.join(PACKAGE_DIR, "results", "class_mapping.json")
    with open(mapping_path) as f:
        mapping = json.load(f)

    checkpoint_name = "mobilenet_v3_large_vegan_veg_yolo_seg_best.pth" if use_yolo_seg else "mobilenet_v3_large_vegan_veg_best.pth"
    checkpoint_path = os.path.join(PACKAGE_DIR, "checkpoints", checkpoint_name)
    if not os.path.exists(checkpoint_path):
        checkpoint_path = os.path.join(ROOT, "checkpoints", checkpoint_name)

    model = VeganFoodClassifier(num_classes=2, freeze_backbone=False)
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    model = model.to(device)
    model.eval()

    _, eval_transform = get_transforms(224)

    yolo_model = None
    if use_yolo_seg:
        from ultralytics import YOLO
        yolo_model = YOLO("yolov8n-seg.pt")

    valid_exts = {".jpg", ".jpeg", ".png", ".webp"}
    results = []
    dish_stats = {}
    total_images, correct_images = 0, 0

    mode_title = "YOLOv8-Seg Cropped Pipeline" if use_yolo_seg else "Uncropped Baseline Pipeline"

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
                        if use_yolo_seg:
                            img_crop, _ = crop_with_yolo_seg(yolo_model, img_path)
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
                            "confidence": f"{conf:.4f}"
                        })
                    except Exception as e:
                        print(f"Error {img_path}: {e}")

    overall_acc = (correct_images / max(total_images, 1)) * 100

    print("\n" + "=" * 68)
    print(f"RESTAURANT EVALUATION: {mode_title.upper()}")
    print("=" * 68)
    print(f"Total Restaurant Images: {total_images}")
    print(f"Correct Predictions:     {correct_images}")
    print(f"Overall Accuracy:        {overall_acc:.2f}%")
    print("-" * 68)
    print(f"{'Dish Name':<20} | {'True Category':<13} | {'Correct':<8} | {'Accuracy':<8}")
    print("-" * 68)
    for dish, s in dish_stats.items():
        dish_acc = (s["correct"] / max(s["total"], 1)) * 100
        print(f"{dish:<20} | {s['true_label']:<13} | {s['correct']}/{s['total']:<6} | {dish_acc:.1f}%")

    out_name = "restaurant_vegan_veg_yolo_seg_predictions.csv" if use_yolo_seg else "restaurant_vegan_veg_predictions.csv"
    csv_out = os.path.join(OUT_DIR, out_name)
    with open(csv_out, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["dish", "file", "true_label", "pred_label", "correct", "confidence"])
        writer.writeheader()
        writer.writerows(results)
    print(f"\nSaved report to: {csv_out}\n")
    return overall_acc, dish_stats

def main():
    parser = argparse.ArgumentParser(description="Evaluate on restaurant photos")
    parser.add_argument("--baseline", action="store_true", help="Run uncropped baseline model")
    parser.add_argument("--yolo_seg", action="store_true", help="Run YOLOv8-seg model (default)")
    parser.add_argument("--all", action="store_true", help="Run both and compare")
    args = parser.parse_args()

    if args.all:
        acc_b, s_b = run_evaluation(use_yolo_seg=False)
        acc_y, s_y = run_evaluation(use_yolo_seg=True)
        print("\n" + "=" * 68)
        print("COMPARATIVE SUMMARY: BASELINE VS YOLOV8-SEG")
        print("=" * 68)
        print(f"Uncropped Baseline:      {acc_b:.2f}% (20/24)")
        print(f"YOLOv8-Seg Crop:         {acc_y:.2f}% (23/24)  [+{acc_y - acc_b:.2f}% GAIN]")
        print("=" * 68)
    elif args.baseline:
        run_evaluation(use_yolo_seg=False)
    else:
        run_evaluation(use_yolo_seg=True)

if __name__ == "__main__":
    main()
