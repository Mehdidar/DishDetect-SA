"""
predict.py
----------
Standalone single-image inference script for the Vegan vs Vegetarian (Meat-Free) Food Classifier.
Supports:
  1. High-accuracy YOLOv8-Seg polygon cropping (--yolo_seg, default)
  2. Direct uncropped image inference (--baseline)

Usage:
    .venv/bin/python Vegan_vs_Vegetarian_Classifier/scripts/predict.py \
        --image "datasets/Restaurant testing data set/cheese fatayer/images.jpg" --yolo_seg
"""

import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import argparse
import json
import torch
import numpy as np
from PIL import Image
from core.models import VeganFoodClassifier
from core.utils import get_device, get_transforms

PACKAGE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
ROOT = os.path.abspath(os.path.join(PACKAGE_DIR, ".."))
CKPT_BASELINE = os.path.join(PACKAGE_DIR, "checkpoints", "mobilenet_v3_large_vegan_veg_best.pth")
CKPT_YOLO_SEG = os.path.join(PACKAGE_DIR, "checkpoints", "mobilenet_v3_large_vegan_veg_yolo_seg_best.pth")
DEFAULT_MAPPING = os.path.join(PACKAGE_DIR, "results", "class_mapping.json")

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

def main():
    parser = argparse.ArgumentParser(description="Predict Vegan vs Vegetarian status of meat-free dishes")
    parser.add_argument("--image", type=str, required=True, help="Path to input food image")
    parser.add_argument("--yolo_seg", action="store_true", default=True, help="Use YOLOv8-seg cropping pipeline (default)")
    parser.add_argument("--baseline", action="store_true", help="Use uncropped baseline model")
    parser.add_argument("--img_size", type=int, default=224, help="Input resolution")
    args = parser.parse_args()

    if not os.path.exists(args.image):
        raise FileNotFoundError(f"Image not found: {args.image}")

    use_seg = not args.baseline
    ckpt_path = CKPT_YOLO_SEG if use_seg else CKPT_BASELINE
    if not os.path.exists(ckpt_path):
        ckpt_path = os.path.join(ROOT, "checkpoints", os.path.basename(ckpt_path))

    device = get_device()
    with open(DEFAULT_MAPPING) as f:
        mapping = json.load(f)

    model = VeganFoodClassifier(num_classes=2, freeze_backbone=False)
    model.load_state_dict(torch.load(ckpt_path, map_location=device))
    model = model.to(device)
    model.eval()

    _, eval_transform = get_transforms(args.img_size)

    crop_info = "Direct Uncropped"
    if use_seg:
        from ultralytics import YOLO
        yolo_model = YOLO("yolov8n-seg.pt")
        img_crop, crop_method = crop_with_yolo_seg(yolo_model, args.image)
        tensor = eval_transform(img_crop).unsqueeze(0).to(device)
        crop_info = f"YOLOv8-Seg ({crop_method})"
    else:
        with Image.open(args.image) as img:
            tensor = eval_transform(img.convert("RGB")).unsqueeze(0).to(device)

    with torch.no_grad():
        outputs = model(tensor)
        probs = torch.softmax(outputs, dim=1)[0]
        pred_idx = torch.argmax(probs).item()
        confidence = probs[pred_idx].item()

    label = mapping[str(pred_idx)]

    print("\n" + "=" * 60)
    print("VEGAN VS VEGETARIAN CLASSIFICATION RESULT")
    print("=" * 60)
    print(f"Image File:     {os.path.basename(args.image)}")
    print(f"Pipeline:       {crop_info}")
    print(f"Prediction:     {label.upper()}")
    print(f"Confidence:     {confidence * 100:.2f}%")
    print("-" * 60)
    print(f"Probability [Vegan]:       {probs[0] * 100:.2f}%")
    print(f"Probability [Vegetarian]:  {probs[1] * 100:.2f}%")
    
    if label == "vegetarian":
        print("\n⚠️  Dietary Note: This dish contains animal byproducts (Dairy, Cheese, Eggs, or Ghee/Samneh). Not suitable for strict vegans.")
    else:
        print("\n🌱 Dietary Note: 100% plant-based (Zero dairy, eggs, or animal fats). Suitable for vegans.")
    print("=" * 60 + "\n")

if __name__ == "__main__":
    main()
