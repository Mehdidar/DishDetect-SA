"""
predict.py
Standalone single-image or directory inference script for the
Vegan vs Vegetarian (Meat-Free) Food Classifier.
Usage:
    .venv/bin/python Vegan_vs_Vegetarian_Classifier/scripts/predict.py --image "datasets/Restaurant testing data set/cheese fatayer/images.jpg"
"""

import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import argparse
import json
import torch
from PIL import Image
from core.models import VeganFoodClassifier
from core.utils import get_device, get_transforms

PACKAGE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
DEFAULT_CKPT = os.path.join(PACKAGE_DIR, "checkpoints", "mobilenet_v3_large_vegan_veg_best.pth")
DEFAULT_MAPPING = os.path.join(PACKAGE_DIR, "results", "class_mapping.json")

def predict_image(image_path, model, device, transform, mapping):
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image not found: {image_path}")
        
    with Image.open(image_path) as img:
        img_rgb = img.convert("RGB")
        tensor = transform(img_rgb).unsqueeze(0).to(device)

    with torch.no_grad():
        outputs = model(tensor)
        probs = torch.softmax(outputs, dim=1)[0]
        pred_idx = torch.argmax(probs).item()
        confidence = probs[pred_idx].item()

    pred_label = mapping[str(pred_idx)]
    return pred_label, confidence, probs

def main():
    parser = argparse.ArgumentParser(description="Predict Vegan vs Vegetarian status of meat-free dishes")
    parser.add_argument("--image", type=str, required=True, help="Path to input food image")
    parser.add_argument("--checkpoint", type=str, default=DEFAULT_CKPT, help="Path to model checkpoint")
    parser.add_argument("--img_size", type=int, default=224, help="Input resolution")
    args = parser.parse_args()

    device = get_device()
    with open(DEFAULT_MAPPING) as f:
        mapping = json.load(f)

    model = VeganFoodClassifier(num_classes=2, freeze_backbone=False)
    model.load_state_dict(torch.load(args.checkpoint, map_location=device))
    model = model.to(device)
    model.eval()

    _, eval_transform = get_transforms(args.img_size)

    label, conf, probs = predict_image(args.image, model, device, eval_transform, mapping)

    print("\n" + "=" * 55)
    print("VEGAN VS VEGETARIAN CLASSIFICATION RESULT")
    print("=" * 55)
    print(f"Image File:   {os.path.basename(args.image)}")
    print(f"Prediction:   {label.upper()}")
    print(f"Confidence:   {conf * 100:.2f}%")
    print("-" * 55)
    print(f"Probability [Vegan]:       {probs[0] * 100:.2f}%")
    print(f"Probability [Vegetarian]:  {probs[1] * 100:.2f}%")
    
    if label == "vegetarian":
        print("\n⚠️  Dietary Note: This dish contains animal byproducts (Dairy, Cheese, Eggs, or Ghee/Samneh). Not suitable for strict vegans.")
    else:
        print("\n🌱 Dietary Note: 100% plant-based (Zero dairy, eggs, or animal fats). Suitable for vegans.")
    print("=" * 55 + "\n")

if __name__ == "__main__":
    main()
