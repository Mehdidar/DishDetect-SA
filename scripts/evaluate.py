import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
"""
evaluate.py
-----------
Evaluates the MobileNet V3 Large (or EfficientNet B2) model on a specified split.

Usage:
    # Evaluate on validation set (legacy default):
    .venv/bin/python evaluate.py --model mobilenet_v3_large --split val

    # Final unbiased evaluation on held-out test set:
    .venv/bin/python evaluate.py --model mobilenet_v3_large --split test
"""

import os
import csv
import json
import argparse
import torch
import shutil
import numpy as np
from torch.utils.data import DataLoader
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    classification_report, confusion_matrix
)
from tqdm import tqdm
from core.models import FoodClassifier
from core.utils import get_device, FoodDataset, get_transforms, plot_confusion_matrix


def evaluate(args):
    device = get_device()

    results_dir = f'results_{args.model}'
    split_info_path = os.path.join(results_dir, 'split_info.json')
    class_mapping_path = os.path.join(results_dir, 'class_mapping.json')

    # --- Load split info ---
    if not os.path.exists(split_info_path):
        raise FileNotFoundError(f"{split_info_path} not found. Please run split_dataset.py and train first.")
    with open(split_info_path) as f:
        split_info = json.load(f)

    if not os.path.exists(class_mapping_path):
        raise FileNotFoundError(f"{class_mapping_path} not found. Please run training first.")
    with open(class_mapping_path) as f:
        mapping = json.load(f)  # e.g. {'0': 'non_veg', '1': 'veg'}

    # --- Select split ---
    split_key = f'{args.split}_files'
    if split_key not in split_info:
        raise ValueError(
            f"Split '{args.split}' not found in split_info.json. "
            f"Available keys: {list(split_info.keys())}\n"
            f"If you want a test set, first run: .venv/bin/python split_dataset.py ..."
        )

    eval_files = split_info[split_key]
    print(f"\nEvaluating on: {args.split.upper()} set ({len(eval_files)} images)")

    if args.split == 'test':
        print("⚠️  This is the FINAL held-out test evaluation. Run this only ONCE after all training is complete.")

    # --- Extract ground truth labels from file paths ---
    reverse_mapping = {v: int(k) for k, v in mapping.items()}
    eval_labels = []
    for fpath in eval_files:
        path_parts = fpath.split(os.sep)
        cls_name = None
        for part in reversed(path_parts[:-1]):
            if part in reverse_mapping:
                cls_name = part
                break
        if cls_name is None:
            raise ValueError(f"Cannot determine class from path: {fpath}")
        eval_labels.append(reverse_mapping[cls_name])

    # --- Load model ---
    checkpoint_path = f'checkpoints/{args.model}_best.pth'
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint {checkpoint_path} not found. Please train first.")

    num_classes = len(mapping)
    model = FoodClassifier(num_classes=num_classes, freeze_backbone=False, architecture=args.model)
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    model = model.to(device)
    model.eval()
    print(f"Loaded checkpoint: {checkpoint_path}")

    # --- Run inference ---
    _, eval_transform = get_transforms(args.img_size)
    eval_dataset = FoodDataset(eval_files, eval_labels, transform=eval_transform)
    eval_loader = DataLoader(eval_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

    all_preds = []
    all_labels_out = []
    all_confidences = []

    with torch.no_grad():
        for inputs, labels in tqdm(eval_loader, desc=f"Evaluating ({args.split})"):
            inputs = inputs.to(device)
            outputs = model(inputs)
            probs = torch.softmax(outputs, dim=1)
            confs, preds = torch.max(probs, 1)
            all_preds.extend(preds.cpu().numpy())
            all_labels_out.extend(labels.numpy())
            all_confidences.extend(confs.cpu().numpy())

    # --- Compute metrics ---
    classes = [mapping[str(i)] for i in range(len(mapping))]
    acc   = accuracy_score(all_labels_out, all_preds)
    prec  = precision_score(all_labels_out, all_preds, average='weighted', zero_division=0)
    rec   = recall_score(all_labels_out, all_preds, average='weighted', zero_division=0)
    f1    = f1_score(all_labels_out, all_preds, average='weighted', zero_division=0)
    cm    = confusion_matrix(all_labels_out, all_preds)
    report_str  = classification_report(all_labels_out, all_preds, target_names=classes)
    report_dict = classification_report(all_labels_out, all_preds, target_names=classes, output_dict=True)

    total_images = len(all_labels_out)
    correct = sum(p == l for p, l in zip(all_preds, all_labels_out))
    incorrect = total_images - correct

    # --- Per-class TP/TN/FP/FN (treating 'vegan' or 'veg' as positive class) ---
    pos_class = reverse_mapping.get('vegan', reverse_mapping.get('veg', 1))
    neg_class = 1 - pos_class
    tp = int(cm[pos_class][pos_class]) if cm.shape[0] > pos_class else 0
    tn = int(cm[neg_class][neg_class]) if cm.shape[0] > neg_class else 0
    fp = int(cm[neg_class][pos_class]) if cm.shape[0] > pos_class else 0
    fn = int(cm[pos_class][neg_class]) if cm.shape[0] > neg_class else 0

    # --- Print results ---
    print(f"\n{'='*55}")
    print(f"EVALUATION RESULTS — {args.split.upper()} SET")
    print(f"{'='*55}")
    print(f"Total Images:        {total_images}")
    print(f"Correct Predictions: {correct}")
    print(f"Incorrect:           {incorrect}")
    print(f"Accuracy:            {acc:.4f} ({acc*100:.2f}%)")
    print(f"Precision (weighted):{prec:.4f}")
    print(f"Recall (weighted):   {rec:.4f}")
    print(f"F1-Score (weighted): {f1:.4f}")
    print(f"\nConfusion Matrix (Positive class = '{classes[pos_class]}'):")
    print(f"  True Positives  (TP): {tp}")
    print(f"  True Negatives  (TN): {tn}")
    print(f"  False Positives (FP): {fp}  (Non-Vegan predicted as Vegan)")
    print(f"  False Negatives (FN): {fn}  (Vegan predicted as Non-Vegan)")
    print(f"\nClassification Report:\n{report_str}")

    if args.split == 'test':
        print("⚠️  NOTE: Test accuracy is the PRIMARY generalization metric.")
        print("   It is unbiased because the test set was NEVER used for training or checkpoint selection.")

    # --- Save outputs ---
    if args.split == 'test':
        out_dir = os.path.join(results_dir, 'test')
    else:
        out_dir = results_dir
    os.makedirs(out_dir, exist_ok=True)

    # classification_report.txt
    with open(os.path.join(out_dir, 'classification_report.txt'), 'w') as f:
        f.write(f"Evaluation Split: {args.split.upper()}\n")
        f.write(f"Total Images: {total_images}\n")
        f.write(f"Correct:      {correct}\n")
        f.write(f"Incorrect:    {incorrect}\n")
        f.write(f"Accuracy:     {acc:.4f}\n")
        f.write(f"Precision:    {prec:.4f}\n")
        f.write(f"Recall:       {rec:.4f}\n")
        f.write(f"F1-Score:     {f1:.4f}\n\n")
        f.write(report_str)

    # metrics.json — ensure all values are Python-native for JSON serialization
    metrics = {
        'split': args.split,
        'model': args.model,
        'total_images': int(total_images),
        'correct': int(correct),
        'incorrect': int(incorrect),
        'accuracy': float(acc),
        'precision_weighted': float(prec),
        'recall_weighted': float(rec),
        'f1_weighted': float(f1),
        'positive_class': 'veg',
        'tp': int(tp), 'tn': int(tn), 'fp': int(fp), 'fn': int(fn),
        'per_class': {k: {m: float(v) for m, v in vv.items()} if isinstance(vv, dict) else float(vv)
                      for k, vv in report_dict.items()},
    }
    with open(os.path.join(out_dir, 'metrics.json'), 'w') as f:
        json.dump(metrics, f, indent=4)

    # confusion_matrix.png
    plot_confusion_matrix(cm, classes, os.path.join(out_dir, 'confusion_matrix.png'))

    # predictions.csv
    idx_to_class = {v: k for k, v in reverse_mapping.items()}
    with open(os.path.join(out_dir, 'predictions.csv'), 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['image_path', 'true_label', 'predicted_label', 'correct', 'confidence'])
        for fpath, true_lbl, pred_lbl, conf in zip(eval_files, all_labels_out, all_preds, all_confidences):
            writer.writerow([
                fpath,
                mapping[str(true_lbl)],
                mapping[str(pred_lbl)],
                int(true_lbl == pred_lbl),
                f'{conf:.4f}',
            ])

    # --- Save predicted images ---
    if args.save_images:
        pred_img_dir = os.path.join(out_dir, 'predicted_images')
        for cls in classes:
            os.makedirs(os.path.join(pred_img_dir, cls), exist_ok=True)
        for fpath, pred_idx in zip(eval_files, all_preds):
            pred_cls = mapping[str(pred_idx)]
            dish_name = os.path.basename(os.path.dirname(fpath))
            dest_dir = os.path.join(pred_img_dir, pred_cls, dish_name)
            os.makedirs(dest_dir, exist_ok=True)
            shutil.copy(fpath, os.path.join(dest_dir, os.path.basename(fpath)))

    # --- Final Evaluation Report (test split only) ---
    if args.split == 'test':
        # Gather train/val info from split_info
        n_train = len(split_info.get('train_files', []))
        n_val   = len(split_info.get('val_files', []))
        n_test  = len(split_info.get('test_files', []))
        total   = n_train + n_val + n_test

        report_lines = [
            "=" * 60,
            "FINAL EVALUATION REPORT",
            f"Model: {args.model}",
            "=" * 60,
            "",
            "DATASET",
            f"  Total images:      {total}",
            f"  Train images:      {n_train} ({n_train/total*100:.1f}%)",
            f"  Validation images: {n_val}  ({n_val/total*100:.1f}%)",
            f"  Test images:       {n_test}  ({n_test/total*100:.1f}%)",
            "",
            "LEAKAGE STATUS",
            "  Duplicate groups checked at split time via split_dataset.py",
            "  Refer to leakage check output for PASS/FAIL status.",
            "",
            "MODEL CONFIGURATION",
            f"  Architecture:      MobileNetV3-Large (torchvision)",
            f"  Image size:        {args.img_size}x{args.img_size}",
            f"  Pretrained:        ImageNet",
            f"  Transfer learning: Yes (2-stage — frozen backbone then fine-tuning)",
            f"  Class weights:     Yes (CrossEntropyLoss with inverse-frequency weights)",
            f"  Random seed:       {split_info.get('seed', 42)}",
            "",
            "METRICS",
            f"  Test Accuracy:            {acc:.4f} ({acc*100:.2f}%)",
            f"  Test Precision (weighted):{prec:.4f}",
            f"  Test Recall (weighted):   {rec:.4f}",
            f"  Test F1-Score (weighted): {f1:.4f}",
            f"  Correct predictions:      {correct}",
            f"  Incorrect predictions:    {incorrect}",
            "",
            "CONFUSION MATRIX (positive class = 'veg')",
            f"  True Positives  (TP): {tp}",
            f"  True Negatives  (TN): {tn}",
            f"  False Positives (FP): {fp}",
            f"  False Negatives (FN): {fn}",
            "",
            "=" * 60,
            "IMPORTANT STATEMENT ON GENERALIZATION",
            "=" * 60,
            "",
            "Test accuracy is the PRIMARY final performance metric because",
            "the test set was NOT used for model training or model selection.",
            "",
            "The validation set (~15%) was used only for:",
            "  - Monitoring convergence",
            "  - Selecting the best checkpoint (early stopping)",
            "  - NOT for gradient updates",
            "",
            "The training set (~70%) was used only for weight updates.",
            "",
            "The ~98% full-dataset accuracy (train+val+test) reported by",
            "total_classification_report.py is NOT a valid generalization",
            "metric because it includes images the model was trained on.",
            "=" * 60,
        ]

        report_path = os.path.join('results', 'final_evaluation_report.txt')
        os.makedirs('results', exist_ok=True)
        with open(report_path, 'w') as f:
            f.write('\n'.join(report_lines))
        print(f"\n📄 Final evaluation report saved: {report_path}")

    print(f"\n✅ Results saved to: {out_dir}/")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate Food Classifier on val or test split")
    parser.add_argument("--model",       type=str,  default='mobilenet_v3_large',
                        choices=[
                            'efficientnet_b2', 'efficientnet_b2_yolo_seg', 'efficientnet_b2_vegan', 'efficientnet_b2_vegan_veg',
                            'mobilenet_v3_large', 'mobilenet_v3_large_yolo', 'mobilenet_v3_large_yolo_seg', 'mobilenet_v3_large_vegan', 'mobilenet_v3_large_vegan_veg', 'mobilenet_v3_large_vegan_veg_yolo_seg'
                        ])
    parser.add_argument("--split",       type=str,  default='val',
                        choices=['val', 'test'],
                        help="Which split to evaluate on. Use 'test' for final unbiased evaluation.")
    parser.add_argument("--batch_size",  type=int,  default=32)
    parser.add_argument("--num_workers", type=int,  default=0)
    parser.add_argument("--img_size",    type=int,  default=224,
                        help="Image size (224 for MobileNetV3, 288 for EfficientNetB2)")
    parser.add_argument("--save_images", action='store_true',
                        help="Copy predicted images into output folder (can be large)")
    args = parser.parse_args()
    evaluate(args)
