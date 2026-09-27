import os
import json
import torch
from torch.utils.data import Dataset
from PIL import Image
import imagehash
from sklearn.model_selection import GroupShuffleSplit
import numpy as np
import matplotlib.pyplot as plt
import torchvision.transforms as transforms
from collections import defaultdict
from tqdm import tqdm

def get_device():
    print(f"PyTorch version: {torch.__version__}")
    mps_avail = torch.backends.mps.is_available()
    cuda_avail = torch.cuda.is_available()
    print(f"MPS availability: {mps_avail}")
    print(f"CUDA availability: {cuda_avail}")
    
    if mps_avail:
        device = torch.device("mps")
    elif cuda_avail:
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
        
    print(f"Selected device: {device}")
    return device

class FoodDataset(Dataset):
    def __init__(self, filepaths, labels, transform=None):
        self.filepaths = filepaths
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.filepaths)

    def __getitem__(self, idx):
        img_path = self.filepaths[idx]
        label = self.labels[idx]
        try:
            image = Image.open(img_path).convert('RGB')
            if self.transform:
                image = self.transform(image)
            return image, label
        except Exception as e:
            print(f"Error loading {img_path}: {e}")
            return torch.zeros(3, 288, 288), label

def get_transforms(img_size=288):
    train_transform = transforms.Compose([
        transforms.Resize((img_size + 32, img_size + 32)),
        transforms.RandomCrop((img_size, img_size)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(p=0.2),
        transforms.RandAugment(),
        transforms.RandomApply([transforms.GaussianBlur(kernel_size=3)], p=0.2),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    val_transform = transforms.Compose([
        transforms.Resize((img_size + 32, img_size + 32)),
        transforms.CenterCrop((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    return train_transform, val_transform

def prepare_data(dataset_path, val_size=0.2, seed=42, hash_threshold=5, results_dir='results'):
    if not os.path.exists(dataset_path):
        raise ValueError(f"Dataset path {dataset_path} does not exist.")
        
    classes = [d for d in os.listdir(dataset_path) if os.path.isdir(os.path.join(dataset_path, d))]
    classes.sort()
    
    if not classes:
        raise ValueError(f"No class folders found in {dataset_path}.")
        
    class_to_idx = {cls_name: i for i, cls_name in enumerate(classes)}
    
    os.makedirs(results_dir, exist_ok=True)
    with open(os.path.join(results_dir, 'class_mapping.json'), 'w') as f:
        mapping = {str(v): k for k, v in class_to_idx.items()}
        json.dump(mapping, f, indent=4)
        
    filepaths = []
    labels = []
    hashes = []
    
    valid_exts = {'.jpg', '.jpeg', '.png', '.bmp', '.webp'}
    
    print("Scanning dataset and computing hashes to prevent data leakage...")
    for cls_name in classes:
        cls_dir = os.path.join(dataset_path, cls_name)
        if not os.path.exists(cls_dir): continue
        
        for root, _, files in os.walk(cls_dir):
            for file in tqdm(files, desc=f"Hashing {cls_name}"):
                if os.path.splitext(file)[1].lower() not in valid_exts: continue
                f = os.path.join(root, file)
                try:
                    with Image.open(f) as img:
                        if img.mode != 'RGB':
                            img = img.convert('RGB')
                        h = imagehash.phash(img)
                    filepaths.append(f)
                    labels.append(class_to_idx[cls_name])
                    hashes.append(h)
                except Exception:
                    pass
                    
    filepaths = np.array(filepaths)
    labels = np.array(labels)
    
    if len(filepaths) == 0:
        raise ValueError("No valid images found in the dataset directory.")
        
    # Group near-duplicates
    groups = np.zeros(len(filepaths), dtype=int)
    group_id = 0
    assigned = np.zeros(len(filepaths), dtype=bool)
    
    for i in range(len(filepaths)):
        if not assigned[i]:
            groups[i] = group_id
            assigned[i] = True
            for j in range(i+1, len(filepaths)):
                if not assigned[j] and hashes[i] - hashes[j] <= hash_threshold:
                    groups[j] = group_id
                    assigned[j] = True
            group_id += 1
            
    print(f"Grouped {len(filepaths)} images into {group_id} unique groups based on near-duplicate threshold {hash_threshold}.")
    
    total = len(labels)
    num_classes = len(classes)
    class_counts = [np.sum(labels == i) for i in range(num_classes)]
    
    weights = []
    for count in class_counts:
        weight = total / (num_classes * count) if count > 0 else 1.0
        weights.append(weight)
        
    class_weights = torch.tensor(weights, dtype=torch.float)    
    gss = GroupShuffleSplit(n_splits=1, test_size=val_size, random_state=seed)
    train_idx, val_idx = next(gss.split(filepaths, labels, groups=groups))
    
    train_files, val_files = filepaths[train_idx], filepaths[val_idx]
    train_labels, val_labels = labels[train_idx], labels[val_idx]
    
    split_info = {
        'train_files': train_files.tolist(),
        'val_files': val_files.tolist()
    }
    with open(os.path.join(results_dir, 'split_info.json'), 'w') as f:
        json.dump(split_info, f, indent=4)
        
    print(f"Train size: {len(train_files)}, Val size: {len(val_files)}")
    print(f"Split info saved to '{results_dir}/split_info.json'")
    return train_files, train_labels, val_files, val_labels, class_weights


def prepare_data_three_way(dataset_path, train_size=0.70, val_size=0.15, seed=42, hash_threshold=5, results_dir='results'):
    """
    Proper 3-way stratified group split: Train / Validation / Test.
    Near-duplicate image groups (by perceptual hash) are kept intact within a single split.
    No image group appears in more than one split.
    
    Args:
        dataset_path: Path to dataset root with class subdirectories
        train_size: Fraction for training (default 0.70)
        val_size: Fraction for validation (default 0.15) — test_size is derived as 1-train-val
        seed: Random seed for reproducibility
        hash_threshold: Max Hamming distance to consider two images near-duplicates
        results_dir: Directory to save split_info.json, class_mapping.json, and split CSVs
    """
    import csv
    import random
    random.seed(seed)
    np.random.seed(seed)

    test_size = round(1.0 - train_size - val_size, 10)
    assert test_size > 0, "train_size + val_size must be < 1.0"

    if not os.path.exists(dataset_path):
        raise ValueError(f"Dataset path {dataset_path} does not exist.")

    classes = sorted([d for d in os.listdir(dataset_path) if os.path.isdir(os.path.join(dataset_path, d))])
    if not classes:
        raise ValueError(f"No class folders found in {dataset_path}.")

    class_to_idx = {cls_name: i for i, cls_name in enumerate(classes)}

    os.makedirs(results_dir, exist_ok=True)
    with open(os.path.join(results_dir, 'class_mapping.json'), 'w') as f:
        json.dump({str(v): k for k, v in class_to_idx.items()}, f, indent=4)

    filepaths, labels, hashes = [], [], []
    valid_exts = {'.jpg', '.jpeg', '.png', '.bmp', '.webp'}

    print("Scanning dataset and computing perceptual hashes...")
    for cls_name in classes:
        cls_dir = os.path.join(dataset_path, cls_name)
        if not os.path.exists(cls_dir):
            continue
        for root, _, files in os.walk(cls_dir):
            for file in tqdm(files, desc=f"Hashing {cls_name}"):
                if os.path.splitext(file)[1].lower() not in valid_exts:
                    continue
                fpath = os.path.join(root, file)
                try:
                    with Image.open(fpath) as img:
                        img_rgb = img.convert('RGB')
                        h = imagehash.phash(img_rgb)
                    filepaths.append(fpath)
                    labels.append(class_to_idx[cls_name])
                    hashes.append(h)
                except Exception:
                    pass

    filepaths = np.array(filepaths)
    labels = np.array(labels)

    if len(filepaths) == 0:
        raise ValueError("No valid images found.")

    # --- Group near-duplicates ---
    groups = np.full(len(filepaths), -1, dtype=int)
    group_id = 0
    for i in range(len(filepaths)):
        if groups[i] == -1:
            groups[i] = group_id
            for j in range(i + 1, len(filepaths)):
                if groups[j] == -1 and (hashes[i] - hashes[j]) <= hash_threshold:
                    groups[j] = group_id
            group_id += 1

    print(f"Grouped {len(filepaths)} images into {group_id} unique hash groups (threshold={hash_threshold}).")

    # --- Stratified 3-way split at group level ---
    # For each class, get the unique groups belonging to that class
    # Then split those groups into train/val/test independently for stratification
    train_idx, val_idx, test_idx = [], [], []

    for cls_idx in range(len(classes)):
        cls_mask = labels == cls_idx
        cls_groups = np.unique(groups[cls_mask])
        np.random.shuffle(cls_groups)

        n = len(cls_groups)
        n_test = max(1, round(n * test_size))
        n_val  = max(1, round(n * val_size))
        n_train = n - n_val - n_test

        if n_train <= 0:
            raise ValueError(f"Not enough groups in class '{classes[cls_idx]}' to form a 3-way split.")

        test_groups  = set(cls_groups[:n_test])
        val_groups   = set(cls_groups[n_test:n_test + n_val])
        train_groups = set(cls_groups[n_test + n_val:])

        for i, (g, lbl) in enumerate(zip(groups, labels)):
            if lbl != cls_idx:
                continue
            if g in train_groups:
                train_idx.append(i)
            elif g in val_groups:
                val_idx.append(i)
            elif g in test_groups:
                test_idx.append(i)

    train_files  = filepaths[train_idx].tolist()
    val_files    = filepaths[val_idx].tolist()
    test_files   = filepaths[test_idx].tolist()
    train_labels_arr = labels[train_idx].tolist()
    val_labels_arr   = labels[val_idx].tolist()
    test_labels_arr  = labels[test_idx].tolist()
    train_groups_arr = groups[train_idx].tolist()
    val_groups_arr   = groups[val_idx].tolist()
    test_groups_arr  = groups[test_idx].tolist()

    # --- Save split_info.json ---
    split_info = {
        'train_files': train_files,
        'val_files':   val_files,
        'test_files':  test_files,
        'seed': seed,
        'hash_threshold': hash_threshold,
        'train_size': train_size,
        'val_size': val_size,
        'test_size': test_size,
    }
    with open(os.path.join(results_dir, 'split_info.json'), 'w') as f:
        json.dump(split_info, f, indent=4)

    # --- Save split CSVs ---
    splits_dir = os.path.join(results_dir, 'splits')
    os.makedirs(splits_dir, exist_ok=True)
    idx_to_class = {v: k for k, v in class_to_idx.items()}

    for split_name, fpaths, lbls, grps in [
        ('train', train_files, train_labels_arr, train_groups_arr),
        ('val',   val_files,   val_labels_arr,   val_groups_arr),
        ('test',  test_files,  test_labels_arr,  test_groups_arr),
    ]:
        csv_path = os.path.join(splits_dir, f'{split_name}.csv')
        with open(csv_path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['image_path', 'class', 'group_id'])
            for fp, lbl, grp in zip(fpaths, lbls, grps):
                writer.writerow([fp, idx_to_class[lbl], grp])

    # --- Compute class weights on train only ---
    total = len(train_labels_arr)
    class_counts = [train_labels_arr.count(i) for i in range(len(classes))]
    weights = [total / (len(classes) * c) if c > 0 else 1.0 for c in class_counts]
    class_weights = torch.tensor(weights, dtype=torch.float)

    print(f"\nSplit info saved to '{results_dir}/split_info.json'")
    print(f"Split CSVs saved to '{splits_dir}/'")

    return (
        np.array(train_files), np.array(train_labels_arr),
        np.array(val_files),   np.array(val_labels_arr),
        np.array(test_files),  np.array(test_labels_arr),
        class_weights
    )

class EarlyStopping:
    def __init__(self, patience=5, min_delta=0, path='checkpoints/best_food_classifier.pth'):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = None
        self.early_stop = False
        self.path = path
        os.makedirs(os.path.dirname(self.path), exist_ok=True)

    def __call__(self, val_loss, model):
        if self.best_loss is None:
            self.best_loss = val_loss
            self.save_checkpoint(model)
        elif val_loss > self.best_loss - self.min_delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_loss = val_loss
            self.save_checkpoint(model)
            self.counter = 0

    def save_checkpoint(self, model):
        torch.save(model.state_dict(), self.path)

def plot_training_curves(history, save_path='results/training_curves.png'):
    epochs = range(1, len(history['train_loss']) + 1)
    
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    plt.plot(epochs, history['train_loss'], label='Train Loss')
    plt.plot(epochs, history['val_loss'], label='Val Loss')
    plt.title('Loss Curves')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    
    plt.subplot(1, 2, 2)
    plt.plot(epochs, history['train_acc'], label='Train Acc')
    plt.plot(epochs, history['val_acc'], label='Val Acc')
    plt.title('Accuracy Curves')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

def plot_confusion_matrix(cm, classes, save_path='results/confusion_matrix.png'):
    fig, ax = plt.subplots(figsize=(6, 5))
    cax = ax.matshow(cm, cmap=plt.cm.Blues)
    fig.colorbar(cax)

    ax.set_xticks(np.arange(len(classes)))
    ax.set_yticks(np.arange(len(classes)))
    ax.set_xticklabels(classes)
    ax.set_yticklabels(classes)
    ax.xaxis.set_ticks_position('bottom')
    
    for i in range(len(classes)):
        for j in range(len(classes)):
            val = cm[i, j]
            color = "white" if val > (cm.max() / 2.) else "black"
            ax.text(j, i, str(val), va='center', ha='center', color=color)

    plt.title('Confusion Matrix')
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
