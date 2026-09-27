import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
"""
full_dataset_diagnostic.py
--------------------------
DIAGNOSTIC TOOL ONLY — NOT A FINAL ACCURACY METRIC.

This script evaluates the model on the entire dataset (Train + Validation + Test).
It is intended for diagnostic purposes only, to understand how well the model
fits the data it was trained on versus held-out data.

⚠️  WARNING: The "Full Dataset Accuracy" reported here (~98%) includes images the
model was trained on. It is NOT a valid estimate of real-world generalization.

The ONLY valid final generalization metric is the TEST SET accuracy, obtained by:
    .venv/bin/python evaluate.py --model mobilenet_v3_large --split test

Usage:
    .venv/bin/python full_dataset_diagnostic.py
"""

import os
import json
import torch
from torchvision import transforms
from torch.utils.data import DataLoader, Dataset
from PIL import Image
from tqdm import tqdm
from sklearn.metrics import accuracy_score, classification_report
from core.models import FoodClassifier


DISCLAIMER = """
╔══════════════════════════════════════════════════════════╗
║          DIAGNOSTIC EVALUATION — NOT FINAL ACCURACY     ║
║                                                          ║
║  Full-dataset accuracy includes TRAINING images.        ║
║  It reflects memorization, NOT generalization.           ║
║                                                          ║
║  To get the real final accuracy, run:                   ║
║  python evaluate.py --model mobilenet_v3_large \\         ║
║                     --split test                         ║
╚══════════════════════════════════════════════════════════╝
"""


class InferenceDataset(Dataset):
    def __init__(self, file_paths, transform):
        self.file_paths = file_paths
        self.transform = transform

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, idx):
        path = self.file_paths[idx]
        image = Image.open(path).convert('RGB')
        return self.transform(image), path


def run_inference(model_name, img_size, all_files, mapping, reverse_mapping, device):
    transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    dataset = InferenceDataset(all_files, transform)
    loader = DataLoader(dataset, batch_size=32, shuffle=False, num_workers=0)

    model = FoodClassifier(num_classes=2, freeze_backbone=False, architecture=model_name)
    checkpoint = f'checkpoints/{model_name}_best.pth'
    if not os.path.exists(checkpoint):
        print(f"  ⚠️  Checkpoint not found: {checkpoint} — skipping {model_name}")
        return

    model.load_state_dict(torch.load(checkpoint, map_location=device))
    model.to(device)
    model.eval()

    all_preds, all_labels = [], []

    with torch.no_grad():
        for inputs, paths in tqdm(loader, desc=f"  Evaluating {model_name}"):
            inputs = inputs.to(device)
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)

            for p, fpath in zip(preds.cpu().numpy(), paths):
                path_parts = fpath.split(os.sep)
                gt_cls = None
                for part in reversed(path_parts[:-1]):
                    if part in reverse_mapping:
                        gt_cls = reverse_mapping[part]
                        break
                if gt_cls is not None:
                    all_labels.append(gt_cls)
                    all_preds.append(p)

    classes = [mapping[str(i)] for i in range(len(mapping))]
    acc = accuracy_score(all_labels, all_preds)
    report = classification_report(all_labels, all_preds, target_names=classes)
    return acc, report, len(all_labels)


def main():
    print(DISCLAIMER)
    device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')

    # --- Load split info ---
    split_info_path = 'results_mobilenet_v3_large/split_info.json'
    class_mapping_path = 'results_mobilenet_v3_large/class_mapping.json'

    if not os.path.exists(split_info_path):
        raise FileNotFoundError(f"{split_info_path} not found. Run split_dataset.py first.")

    with open(split_info_path) as f:
        split_info = json.load(f)
    with open(class_mapping_path) as f:
        mapping = json.load(f)

    reverse_mapping = {v: int(k) for k, v in mapping.items()}

    train_files = split_info.get('train_files', [])
    val_files   = split_info.get('val_files', [])
    test_files  = split_info.get('test_files', [])
    all_files   = train_files + val_files + test_files

    print(f"Split breakdown:")
    print(f"  TRAIN:      {len(train_files)} images")
    print(f"  VALIDATION: {len(val_files)} images")
    print(f"  TEST:       {len(test_files)} images")
    print(f"  TOTAL:      {len(all_files)} images")

    model_configs = [
        ('mobilenet_v3_large', 224),
    ]

    for model_name, img_size in model_configs:
        print(f"\n{'='*60}")
        print(f"DIAGNOSTIC REPORT — {model_name}")
        print(f"{'='*60}")

        # Per-split accuracy
        for split_name, files in [('TRAIN', train_files), ('VALIDATION', val_files), ('TEST', test_files)]:
            if not files:
                continue
            result = run_inference(model_name, img_size, files, mapping, reverse_mapping, device)
            if result:
                acc, _, n = result
                label = "(seen during training)" if split_name == 'TRAIN' else \
                        "(used for checkpoint selection)" if split_name == 'VALIDATION' else \
                        "(held-out — PRIMARY metric)"
                print(f"\n  {split_name} Accuracy ({n} images): {acc*100:.2f}%  {label}")

        # Full dataset accuracy
        print(f"\n  {'─'*55}")
        result_all = run_inference(model_name, img_size, all_files, mapping, reverse_mapping, device)
        if result_all:
            acc_all, report_all, n_all = result_all
            print(f"\n  FULL DATASET 'Accuracy' ({n_all} images): {acc_all*100:.2f}%")
            print(f"  ⚠️  This includes training images. It is NOT a generalization metric.")
            print(f"\n  Full Dataset Classification Report:")
            print(report_all)

    print(DISCLAIMER)


if __name__ == '__main__':
    main()
