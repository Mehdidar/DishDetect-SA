"""
predict_hierarchical.py
-----------------------
Connects the Veg vs Non-Veg classifier and Vegan vs Vegetarian classifier into a
unified, two-stage hierarchical dietary intelligence pipeline:

Pipeline Architecture:
  Input Image
       │
  [YOLOv8-Seg Crop] (Isolates food, discards plate & table glare)
       │
  ▼ STAGE 1: VEG VS. NON-VEG ▼
  (MobileNetV3-Large Veg/Non-Veg Classifier)
       │
       ├──► If NON-VEG ────────► [NON-VEGETARIAN] (Meat, Poultry, Fish, Seafood)
       │
       └──► If VEG (Meat-Free) ──┐
                                 ▼ STAGE 2: VEGAN VS. VEGETARIAN ▼
                                 (MobileNetV3-Large Vegan/Veg Classifier)
                                 │
                                 ├──► [VEGAN] (100% Plant-Based: Zero animal products)
                                 │
                                 └──► [VEGETARIAN] (Lacto-Ovo: Contains Dairy, Cheese, Eggs, Ghee)

Usage:
  # Predict single image:
  .venv/bin/python scripts/predict_hierarchical.py --image "datasets/Restaurant testing data set/cheese fatayer/images (1).jpg"

  # Run full benchmark evaluation across all 9 restaurant dishes (meat + veg + vegan):
  .venv/bin/python scripts/predict_hierarchical.py --eval_restaurant
"""

import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import argparse
import csv
import json
import torch
import numpy as np
from PIL import Image
from ultralytics import YOLO

from core.models import FoodClassifier
from core.utils import get_device, get_transforms

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
CKPT_STAGE1 = os.path.join(ROOT, "checkpoints", "mobilenet_v3_large_yolo_seg_best.pth")
CKPT_STAGE2 = os.path.join(ROOT, "checkpoints", "mobilenet_v3_large_vegan_veg_yolo_seg_best.pth")
RESTAURANT_DIR = os.path.join(ROOT, "datasets", "Restaurant testing data set")
OUT_DIR = os.path.join(ROOT, "results", "hierarchical_evaluation")

# Complete 3-Way Ground Truth for the Restaurant Dishes
RESTAURANT_GROUND_TRUTH = {
    # Non-Veg
    "Mandi": "non_veg",
    "Shish tawook": "non_veg",
    "meat madfoon": "non_veg",
    "mixed-grill": "non_veg",
    # Vegetarian (Dairy)
    "cheese fatayer": "vegetarian",
    # Vegan (Plant-Based)
    "hummus": "vegan",
    "okra tagine": "vegan",
    "tamees bread": "vegan",
    "zaatara manaqish": "vegan",
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

class HierarchicalDietaryClassifier:
    def __init__(self, use_yolo=True):
        self.device = get_device()
        self.use_yolo = use_yolo
        _, self.transform = get_transforms(224)

        if self.use_yolo:
            self.yolo_model = YOLO("yolov8n-seg.pt")
        else:
            self.yolo_model = None

        # Stage 1: Veg (1) vs Non-Veg (0)
        self.model_stage1 = FoodClassifier(num_classes=2, freeze_backbone=False, architecture='mobilenet_v3_large')
        self.model_stage1.load_state_dict(torch.load(CKPT_STAGE1, map_location=self.device))
        self.model_stage1.to(self.device).eval()

        # Stage 2: Vegan (0) vs Vegetarian (1)
        self.model_stage2 = FoodClassifier(num_classes=2, freeze_backbone=False, architecture='mobilenet_v3_large')
        self.model_stage2.load_state_dict(torch.load(CKPT_STAGE2, map_location=self.device))
        self.model_stage2.to(self.device).eval()

    def predict(self, img_path):
        if not os.path.exists(img_path):
            raise FileNotFoundError(f"Image not found: {img_path}")

        crop_method = "direct"
        if self.use_yolo:
            img_crop, crop_method = crop_with_yolo_seg(self.yolo_model, img_path)
            tensor = self.transform(img_crop).unsqueeze(0).to(self.device)
        else:
            with Image.open(img_path) as img:
                tensor = self.transform(img.convert("RGB")).unsqueeze(0).to(self.device)

        # ── STAGE 1: Veg vs Non-Veg ──────────────────────────────────────────
        with torch.no_grad():
            out1 = self.model_stage1(tensor)
            probs1 = torch.softmax(out1, dim=1)[0]
            # Mapping: 0: non_veg, 1: veg
            prob_non_veg = probs1[0].item()
            prob_veg = probs1[1].item()

        is_veg = prob_veg > prob_non_veg
        s1_pred = "veg" if is_veg else "non_veg"
        s1_conf = prob_veg if is_veg else prob_non_veg

        # ── STAGE 2: Vegan vs Vegetarian (only if Stage 1 = Veg) ─────────────
        final_label = "non_veg"
        final_category = "NON-VEGETARIAN"
        dietary_note = "🍖 Contains Animal Flesh (Meat, Poultry, Fish, or Seafood). Strictly Non-Vegetarian."
        s2_pred = None
        s2_conf = None
        prob_vegan = 0.0
        prob_vegetarian = 0.0

        if is_veg:
            with torch.no_grad():
                out2 = self.model_stage2(tensor)
                probs2 = torch.softmax(out2, dim=1)[0]
                # Mapping: 0: vegan, 1: vegetarian
                prob_vegan = probs2[0].item()
                prob_vegetarian = probs2[1].item()

            if prob_vegan > prob_vegetarian:
                final_label = "vegan"
                final_category = "VEGAN (100% Plant-Based)"
                s2_pred = "vegan"
                s2_conf = prob_vegan
                dietary_note = "🌱 100% Plant-Based (Zero meat, dairy, eggs, or animal fats). Suitable for strict vegans."
            else:
                final_label = "vegetarian"
                final_category = "VEGETARIAN (Lacto-Ovo)"
                s2_pred = "vegetarian"
                s2_conf = prob_vegetarian
                dietary_note = "🧀 Meat-Free, but contains animal byproducts (Dairy Cheese, Milk, Eggs, or Ghee/Samneh). Not vegan."

        return {
            "image": os.path.basename(img_path),
            "final_label": final_label,
            "final_category": final_category,
            "stage1_pred": s1_pred,
            "stage1_conf": s1_conf,
            "prob_veg": prob_veg,
            "prob_non_veg": prob_non_veg,
            "stage2_pred": s2_pred,
            "stage2_conf": s2_conf,
            "prob_vegan": prob_vegan,
            "prob_vegetarian": prob_vegetarian,
            "dietary_note": dietary_note,
            "crop_method": crop_method
        }

def evaluate_restaurant_benchmark():
    print("\n" + "=" * 70)
    print("RUNNING COMPLETE HIERARCHICAL RESTAURANT EVALUATION (STAGE 1 + 2)")
    print("=" * 70)
    
    classifier = HierarchicalDietaryClassifier(use_yolo=True)
    os.makedirs(OUT_DIR, exist_ok=True)

    valid_exts = {".jpg", ".jpeg", ".png", ".webp"}
    results = []
    dish_summary = {}
    total_imgs, correct_imgs = 0, 0
    s1_correct = 0

    for dish_folder, true_diet in RESTAURANT_GROUND_TRUTH.items():
        folder_path = os.path.join(RESTAURANT_DIR, dish_folder)
        if not os.path.exists(folder_path):
            continue

        dish_summary[dish_folder] = {"total": 0, "correct": 0, "true_diet": true_diet, "s1_correct": 0}

        for root, _, files in os.walk(folder_path):
            for file in sorted(files):
                if os.path.splitext(file)[1].lower() in valid_exts:
                    img_path = os.path.join(root, file)
                    res = classifier.predict(img_path)

                    total_imgs += 1
                    dish_summary[dish_folder]["total"] += 1

                    # Check Stage 1 correctness
                    expected_s1 = "non_veg" if true_diet == "non_veg" else "veg"
                    if res["stage1_pred"] == expected_s1:
                        s1_correct += 1
                        dish_summary[dish_folder]["s1_correct"] += 1

                    # Check End-to-End Hierarchical correctness
                    is_correct = (res["final_label"] == true_diet)
                    if is_correct:
                        correct_imgs += 1
                        dish_summary[dish_folder]["correct"] += 1

                    results.append({
                        "dish": dish_folder,
                        "file": file,
                        "true_diet": true_diet,
                        "final_prediction": res["final_label"],
                        "stage1_pred": res["stage1_pred"],
                        "stage1_conf": f"{res['stage1_conf']:.4f}",
                        "stage2_pred": res["stage2_pred"] or "N/A",
                        "stage2_conf": f"{res['stage2_conf']:.4f}" if res["stage2_conf"] else "N/A",
                        "correct": is_correct
                    })

    overall_acc = (correct_imgs / max(total_imgs, 1)) * 100
    stage1_acc = (s1_correct / max(total_imgs, 1)) * 100

    print(f"Total Restaurant Images:         {total_imgs}")
    print(f"Stage 1 (Veg vs Non-Veg) Acc:    {stage1_acc:.2f}% ({s1_correct}/{total_imgs})")
    print(f"Final 3-Tier Hierarchical Acc:   {overall_acc:.2f}% ({correct_imgs}/{total_imgs})")
    print("-" * 70)
    print(f"{'Dish Name':<20} | {'True Diet':<12} | {'Correct':<8} | {'Accuracy':<8}")
    print("-" * 70)
    for dish, s in dish_summary.items():
        acc = (s["correct"] / max(s["total"], 1)) * 100
        print(f"{dish:<20} | {s['true_diet']:<12} | {s['correct']}/{s['total']:<6} | {acc:.1f}%")
    print("=" * 70)

    csv_out = os.path.join(OUT_DIR, "hierarchical_restaurant_predictions.csv")
    with open(csv_out, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "dish", "file", "true_diet", "final_prediction", "stage1_pred",
            "stage1_conf", "stage2_pred", "stage2_conf", "correct"
        ])
        writer.writeheader()
        writer.writerows(results)
    print(f"Predictions saved to: {csv_out}\n")
    return overall_acc

def main():
    parser = argparse.ArgumentParser(description="Hierarchical Dietary Classifier (Veg/Non-Veg -> Vegan/Vegetarian)")
    parser.add_argument("--image", type=str, help="Path to single input image")
    parser.add_argument("--eval_restaurant", action="store_true", help="Run full restaurant benchmark across all 9 dishes")
    args = parser.parse_args()

    if args.eval_restaurant:
        evaluate_restaurant_benchmark()
        return

    if args.image:
        classifier = HierarchicalDietaryClassifier(use_yolo=True)
        res = classifier.predict(args.image)

        print("\n" + "=" * 65)
        print("HIERARCHICAL DIETARY INTELLIGENCE RESULT")
        print("=" * 65)
        print(f"Image File:         {res['image']}")
        print(f"Cropping Pipeline:  YOLOv8-Seg ({res['crop_method']})")
        print("-" * 65)
        print("STAGE 1 [Veg vs. Non-Veg]:")
        print(f"  • Decision:       {res['stage1_pred'].upper()} ({res['stage1_conf'] * 100:.2f}% conf)")
        print(f"  • P(Veg):         {res['prob_veg'] * 100:.2f}%")
        print(f"  • P(Non-Veg):     {res['prob_non_veg'] * 100:.2f}%")
        print("-" * 65)

        if res["stage1_pred"] == "veg":
            print("STAGE 2 [Vegan vs. Vegetarian]:")
            print(f"  • Decision:       {res['stage2_pred'].upper()} ({res['stage2_conf'] * 100:.2f}% conf)")
            print(f"  • P(Vegan):       {res['prob_vegan'] * 100:.2f}%")
            print(f"  • P(Vegetarian):  {res['prob_vegetarian'] * 100:.2f}%")
            print("-" * 65)

        print(f"FINAL CLASSIFICATION:  {res['final_category']}")
        print(f"\n{res['dietary_note']}")
        print("=" * 65 + "\n")
    else:
        parser.print_help()

if __name__ == "__main__":
    main()
