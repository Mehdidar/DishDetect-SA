"""
evaluate.py
Evaluates the Vegan vs Vegetarian model on either the validation or test split.
Usage:
    .venv/bin/python Vegan_vs_Vegetarian_Classifier/scripts/evaluate.py --split test
"""

import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import json
import argparse
import torch
from torch.utils.data import DataLoader
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
from core.models import VeganFoodClassifier
from core.utils import get_device, get_transforms, VeganDataset

PACKAGE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

def evaluate_split(split="test", img_size=224, batch_size=32):
    device = get_device()
    split_info_path = os.path.join(PACKAGE_DIR, "results", "split_info.json")
    mapping_path = os.path.join(PACKAGE_DIR, "results", "class_mapping.json")
    checkpoint_path = os.path.join(PACKAGE_DIR, "checkpoints", "mobilenet_v3_large_vegan_veg_best.pth")

    with open(split_info_path) as f:
        split_info = json.load(f)
    with open(mapping_path) as f:
        mapping = json.load(f)
    reverse_mapping = {v: int(k) for k, v in mapping.items()}

    eval_files = split_info[f"{split}_files"]
    eval_labels = []
    for fp in eval_files:
        cls_name = "vegan" if "/vegan/" in fp or "\\vegan\\" in fp else "vegetarian"
        eval_labels.append(reverse_mapping[cls_name])

    _, eval_transform = get_transforms(img_size)
    dataset = VeganDataset(eval_files, eval_labels, transform=eval_transform)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    model = VeganFoodClassifier(num_classes=2, freeze_backbone=False)
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    model = model.to(device)
    model.eval()

    all_preds, all_labels = [], []
    with torch.no_grad():
        for inputs, targets in loader:
            inputs = inputs.to(device)
            outputs = model(inputs)
            preds = torch.argmax(outputs, dim=1).cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(targets.numpy())

    classes = [mapping[str(i)] for i in range(len(mapping))]
    acc = accuracy_score(all_labels, all_preds)
    prec = precision_score(all_labels, all_preds, average="weighted")
    rec = recall_score(all_labels, all_preds, average="weighted")
    f1 = f1_score(all_labels, all_preds, average="weighted")
    cm = confusion_matrix(all_labels, all_preds)

    print("\n" + "=" * 55)
    print(f"EVALUATION RESULTS — {split.upper()} SET ({len(eval_files)} IMAGES)")
    print("=" * 55)
    print(f"Accuracy:            {acc * 100:.2f}%")
    print(f"Weighted Precision:  {prec:.4f}")
    print(f"Weighted Recall:     {rec:.4f}")
    print(f"Weighted F1-Score:   {f1:.4f}")
    print("\nClassification Report:")
    print(classification_report(all_labels, all_preds, target_names=classes))
    print("Confusion Matrix:")
    print(cm)
    print("=" * 55 + "\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", type=str, default="test", choices=["val", "test"])
    args = parser.parse_args()
    evaluate_split(args.split)
