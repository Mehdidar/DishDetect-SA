"""
evaluate_external.py
--------------------
Evaluates the trained MobileNet V3 Large (or any architecture) directly
on an external test folder WITHOUT needing split_info.json.

The test folder must have this structure:
    test_dir/
        veg/
            dish_name/
                image1.jpg
                image2.jpg
        non_veg/
            dish_name/
                image1.jpg

The class (veg / non_veg) is inferred from the top-level subfolder name.

Usage:
    .venv/bin/python scripts/evaluate_external.py \
        --model    mobilenet_v3_large_yolo_seg \
        --test_dir results/yolo_seg_test \
        --img_size 288

    # Or evaluate on raw (non-cropped) test set:
    .venv/bin/python scripts/evaluate_external.py \
        --model    mobilenet_v3_large_yolo_seg \
        --test_dir test_dataset_dishes \
        --img_size 288
"""
import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import csv
import json
import argparse
import torch
import numpy as np
from pathlib import Path
from torch.utils.data import DataLoader
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, classification_report, confusion_matrix
)
from tqdm import tqdm

from core.models import FoodClassifier
from core.utils import get_device, FoodDataset, get_transforms, plot_confusion_matrix

VALID_EXTS = {'.jpg', '.jpeg', '.png', '.bmp', '.webp'}


def load_external_test_set(test_dir: Path, class_mapping: dict):
    """
    Walk test_dir/class_name/[dish_name/]*.jpg and return
    (file_paths, labels, per_dish_info).

    class_mapping: {'non_veg': 0, 'veg': 1}  (or reversed — built from checkpoint)
    """
    file_paths = []
    labels     = []
    dish_info  = {}   # dish_name → {'class': str, 'count': int}

    class_dirs = sorted([d for d in test_dir.iterdir() if d.is_dir()])
    for class_dir in class_dirs:
        cls_name = class_dir.name
        if cls_name not in class_mapping:
            print(f"  ⚠️  Unknown class folder '{cls_name}' — skipping")
            continue
        cls_idx = class_mapping[cls_name]

        # Check for dish sub-folders
        sub_dirs = [d for d in class_dir.iterdir() if d.is_dir()]
        if sub_dirs:
            for dish_dir in sorted(sub_dirs):
                imgs = [f for f in dish_dir.iterdir()
                        if f.suffix.lower() in VALID_EXTS]
                dish_name = f"{cls_name}/{dish_dir.name}"
                dish_info[dish_name] = {'class': cls_name, 'count': len(imgs)}
                for img in imgs:
                    file_paths.append(str(img))
                    labels.append(cls_idx)
        else:
            # Flat: class_name/*.jpg
            imgs = [f for f in class_dir.iterdir()
                    if f.suffix.lower() in VALID_EXTS]
            dish_info[cls_name] = {'class': cls_name, 'count': len(imgs)}
            for img in imgs:
                file_paths.append(str(img))
                labels.append(cls_idx)

    return file_paths, labels, dish_info


def evaluate_external(args):
    device = get_device()

    # ── Load class mapping from training results ──────────────────────────
    results_dir = f'results_{args.model}'
    class_mapping_path = os.path.join(results_dir, 'class_mapping.json')

    if not os.path.exists(class_mapping_path):
        # Try building from checkpoint name heuristic
        print(f"⚠️  class_mapping.json not found at {class_mapping_path}")
        print("   Using default: {'non_veg': 0, 'veg': 1}")
        idx_to_class = {'0': 'non_veg', '1': 'veg'}
    else:
        with open(class_mapping_path) as f:
            idx_to_class = json.load(f)   # {'0': 'non_veg', '1': 'veg'}

    class_to_idx = {v: int(k) for k, v in idx_to_class.items()}
    num_classes  = len(idx_to_class)
    classes      = [idx_to_class[str(i)] for i in range(num_classes)]

    # ── Load model ────────────────────────────────────────────────────────
    # Map model name to architecture key accepted by FoodClassifier
    arch_map = {
        'mobilenet_v3_large_yolo_seg': 'mobilenet_v3_large_yolo_seg',
        'mobilenet_v3_large':          'mobilenet_v3_large',
        'mobilenet_v3_large_yolo':     'mobilenet_v3_large_yolo',
        'efficientnet_b2':             'efficientnet_b2',
        'efficientnet_b2_yolo_seg':    'efficientnet_b2_yolo_seg',
    }
    arch = arch_map.get(args.model, args.model)
    checkpoint_path = f'checkpoints/{args.model}_best.pth'

    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(
            f"Checkpoint not found: {checkpoint_path}\n"
            f"Please train first with --model {args.model}"
        )

    model = FoodClassifier(num_classes=num_classes, freeze_backbone=False, architecture=arch)
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    model = model.to(device)
    model.eval()
    print(f"Loaded checkpoint: {checkpoint_path}")

    # ── Load external test set ────────────────────────────────────────────
    test_dir = Path(args.test_dir)
    if not test_dir.exists():
        raise FileNotFoundError(f"Test directory not found: {test_dir}")

    file_paths, labels, dish_info = load_external_test_set(test_dir, class_to_idx)
    print(f"\nExternal test set: {len(file_paths)} images")
    print(f"  Classes found: {list(class_to_idx.keys())}")
    print(f"  Dishes found:  {len(dish_info)}")
    for dish, info in dish_info.items():
        print(f"    {dish}: {info['count']} images")

    if len(file_paths) == 0:
        raise ValueError("No images found in the test directory!")

    # ── Run inference ─────────────────────────────────────────────────────
    _, eval_transform = get_transforms(args.img_size)
    eval_dataset = FoodDataset(file_paths, labels, transform=eval_transform)
    eval_loader  = DataLoader(
        eval_dataset, batch_size=args.batch_size,
        shuffle=False, num_workers=args.num_workers
    )

    all_preds  = []
    all_labels = []
    all_confs  = []

    with torch.no_grad():
        for inputs, lbls in tqdm(eval_loader, desc="Evaluating external test set"):
            inputs = inputs.to(device)
            outputs = model(inputs)
            probs = torch.softmax(outputs, dim=1)
            confs, preds = torch.max(probs, 1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(lbls.numpy())
            all_confs.extend(confs.cpu().numpy())

    # ── Metrics ───────────────────────────────────────────────────────────
    acc   = accuracy_score(all_labels, all_preds)
    prec  = precision_score(all_labels, all_preds, average='weighted', zero_division=0)
    rec   = recall_score(all_labels, all_preds, average='weighted', zero_division=0)
    f1    = f1_score(all_labels, all_preds, average='weighted', zero_division=0)
    cm    = confusion_matrix(all_labels, all_preds)
    report_str  = classification_report(all_labels, all_preds, target_names=classes, zero_division=0)
    report_dict = classification_report(all_labels, all_preds, target_names=classes,
                                        output_dict=True, zero_division=0)

    total_images = len(all_labels)
    correct   = sum(p == l for p, l in zip(all_preds, all_labels))
    incorrect = total_images - correct

    # Per-class TP/TN/FP/FN
    pos_class = class_to_idx.get('veg', 1)
    neg_class = 1 - pos_class
    tp = int(cm[pos_class][pos_class]) if cm.shape[0] > pos_class else 0
    tn = int(cm[neg_class][neg_class]) if cm.shape[0] > neg_class else 0
    fp = int(cm[neg_class][pos_class]) if cm.shape[0] > pos_class else 0
    fn = int(cm[pos_class][neg_class]) if cm.shape[0] > neg_class else 0

    # ── Print ─────────────────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print(f"EXTERNAL TEST SET EVALUATION RESULTS")
    print(f"Model:      {args.model}")
    print(f"Test dir:   {args.test_dir}")
    print(f"{'='*60}")
    print(f"Total Images:          {total_images}")
    print(f"Correct Predictions:   {correct}")
    print(f"Incorrect:             {incorrect}")
    print(f"Accuracy:              {acc:.4f}  ({acc*100:.2f}%)")
    print(f"Precision (weighted):  {prec:.4f}")
    print(f"Recall (weighted):     {rec:.4f}")
    print(f"F1-Score (weighted):   {f1:.4f}")
    print(f"\nConfusion Matrix (positive class = 'veg'):")
    print(f"  True Positives  (TP): {tp}")
    print(f"  True Negatives  (TN): {tn}")
    print(f"  False Positives (FP): {fp}  (Non-Veg predicted as Veg)")
    print(f"  False Negatives (FN): {fn}  (Veg predicted as Non-Veg)")
    print(f"\nClassification Report:\n{report_str}")

    # ── Save results ──────────────────────────────────────────────────────
    out_dir = os.path.join('results', f'test_external_{args.model}')
    os.makedirs(out_dir, exist_ok=True)

    # classification_report.txt
    with open(os.path.join(out_dir, 'classification_report.txt'), 'w') as f:
        f.write(f"EXTERNAL TEST SET EVALUATION\n")
        f.write(f"Model:      {args.model}\n")
        f.write(f"Test dir:   {args.test_dir}\n")
        f.write(f"Total:      {total_images}\n")
        f.write(f"Correct:    {correct}\n")
        f.write(f"Incorrect:  {incorrect}\n")
        f.write(f"Accuracy:   {acc:.4f}  ({acc*100:.2f}%)\n")
        f.write(f"Precision:  {prec:.4f}\n")
        f.write(f"Recall:     {rec:.4f}\n")
        f.write(f"F1-Score:   {f1:.4f}\n\n")
        f.write(report_str)

    # metrics.json
    metrics = {
        'model':          args.model,
        'test_dir':       args.test_dir,
        'total_images':   int(total_images),
        'correct':        int(correct),
        'incorrect':      int(incorrect),
        'accuracy':       float(acc),
        'precision_weighted': float(prec),
        'recall_weighted': float(rec),
        'f1_weighted':    float(f1),
        'positive_class': 'veg',
        'tp': int(tp), 'tn': int(tn), 'fp': int(fp), 'fn': int(fn),
        'per_class': {
            k: {m: float(v) for m, v in vv.items()} if isinstance(vv, dict) else float(vv)
            for k, vv in report_dict.items()
        },
        'dish_breakdown': dish_info,
    }
    with open(os.path.join(out_dir, 'metrics.json'), 'w') as f:
        json.dump(metrics, f, indent=4)

    # confusion_matrix.png
    plot_confusion_matrix(cm, classes, os.path.join(out_dir, 'confusion_matrix.png'))

    # predictions.csv
    with open(os.path.join(out_dir, 'predictions.csv'), 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['image_path', 'true_label', 'predicted_label', 'correct', 'confidence'])
        for fpath, true_lbl, pred_lbl, conf in zip(file_paths, all_labels, all_preds, all_confs):
            writer.writerow([
                fpath,
                idx_to_class[str(true_lbl)],
                idx_to_class[str(pred_lbl)],
                int(true_lbl == pred_lbl),
                f'{conf:.4f}',
            ])

    print(f"\n✅ Results saved to: {out_dir}/")
    print(f"   - classification_report.txt")
    print(f"   - metrics.json")
    print(f"   - confusion_matrix.png")
    print(f"   - predictions.csv")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="Evaluate food classifier on a fully external held-out test folder"
    )
    parser.add_argument('--model',       type=str, default='mobilenet_v3_large_yolo_seg',
                        help='Model name (used to find checkpoint and results dir)')
    parser.add_argument('--test_dir',    type=str, required=True,
                        help='Path to external test folder (veg/ non_veg/ subfolders)')
    parser.add_argument('--img_size',    type=int, default=288)
    parser.add_argument('--batch_size',  type=int, default=32)
    parser.add_argument('--num_workers', type=int, default=0)
    args = parser.parse_args()
    evaluate_external(args)
