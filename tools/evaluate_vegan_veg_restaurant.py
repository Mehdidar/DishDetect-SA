"""
evaluate_vegan_veg_restaurant.py
Evaluates the trained MobileNetV3-Large Vegan vs Vegetarian (Meat-Free) Classifier
on real-world Arabic Restaurant Test Dataset images.
Dishes evaluated:
    Vegan (Plant-based): hummus, tamees bread, zaatara manaqish, okra tagine
    Vegetarian (Dairy/Cheese): cheese fatayer
All meat dishes are excluded.
"""

import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import csv
import json
import torch
from PIL import Image
from torchvision import transforms
from core.models import FoodClassifier
from core.utils import get_device

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
RESTAURANT_DIR = os.path.join(ROOT, "datasets", "Restaurant testing data set")
CHECKPOINT_PATH = os.path.join(ROOT, "checkpoints", "mobilenet_v3_large_vegan_veg_best.pth")
OUT_DIR = os.path.join(ROOT, "results_mobilenet_v3_large_vegan_veg", "restaurant_evaluation")

RESTAURANT_MEATFREE_TRUTH = {
    "hummus": "vegan",
    "okra tagine": "vegan",
    "tamees bread": "vegan",
    "zaatara manaqish": "vegan",
    "cheese fatayer": "vegetarian",
}

def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    device = get_device()

    mapping_path = os.path.join(ROOT, "results_mobilenet_v3_large_vegan_veg", "class_mapping.json")
    with open(mapping_path) as f:
        mapping = json.load(f) # {'0': 'vegan', '1': 'vegetarian'}

    model = FoodClassifier(num_classes=2, freeze_backbone=False, architecture="mobilenet_v3_large_vegan_veg")
    model.load_state_dict(torch.load(CHECKPOINT_PATH, map_location=device))
    model = model.to(device)
    model.eval()

    eval_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    valid_exts = {".jpg", ".jpeg", ".png", ".webp"}
    results = []
    dish_stats = {}

    total_images = 0
    correct_images = 0

    for dish_folder, true_label in RESTAURANT_MEATFREE_TRUTH.items():
        folder_path = os.path.join(RESTAURANT_DIR, dish_folder)
        if not os.path.exists(folder_path):
            continue

        dish_stats[dish_folder] = {"total": 0, "correct": 0, "true_label": true_label}

        for root, _, files in os.walk(folder_path):
            for file in files:
                if os.path.splitext(file)[1].lower() in valid_exts:
                    img_path = os.path.join(root, file)
                    try:
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
                        print(f"Error processing {img_path}: {e}")

    overall_acc = (correct_images / max(total_images, 1)) * 100

    print("=" * 65)
    print("RESTAURANT EVALUATION: VEGAN VS VEGETARIAN (MEAT-FREE ONLY)")
    print("=" * 65)
    print(f"Total Images:     {total_images}")
    print(f"Correct:          {correct_images}")
    print(f"Overall Accuracy: {overall_acc:.2f}%")
    print("-" * 65)
    print(f"{'Dish Name':<20} | {'True Category':<13} | {'Correct':<8} | {'Accuracy':<8}")
    print("-" * 65)

    for dish, s in dish_stats.items():
        dish_acc = (s["correct"] / max(s["total"], 1)) * 100
        print(f"{dish:<20} | {s['true_label']:<13} | {s['correct']}/{s['total']:<6} | {dish_acc:.1f}%")

    # Save to CSV
    csv_out = os.path.join(OUT_DIR, "restaurant_vegan_veg_predictions.csv")
    with open(csv_out, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["dish", "file", "true_label", "pred_label", "correct", "confidence"])
        writer.writeheader()
        writer.writerows(results)

    print(f"\nSaved predictions to: {csv_out}")

if __name__ == "__main__":
    main()
