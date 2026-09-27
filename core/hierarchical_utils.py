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
from .utils import get_device, get_transforms, FoodDataset

def prepare_hierarchical_data(dataset_path, val_size=0.2, seed=42, hash_threshold=5, results_dir='results'):
    if not os.path.exists(dataset_path):
        raise ValueError(f"Dataset path {dataset_path} does not exist.")
        
    # Discover classes hierarchically: dataset_path/veg/dish_name
    top_levels = [d for d in os.listdir(dataset_path) if os.path.isdir(os.path.join(dataset_path, d))]
    
    classes = []
    dish_to_top = {}
    for top in top_levels:
        top_dir = os.path.join(dataset_path, top)
        dishes = [d for d in os.listdir(top_dir) if os.path.isdir(os.path.join(top_dir, d))]
        for dish in dishes:
            classes.append(dish)
            dish_to_top[dish] = top
            
    classes.sort()
    
    if not classes:
        raise ValueError(f"No class folders found in {dataset_path}.")
        
    class_to_idx = {cls_name: i for i, cls_name in enumerate(classes)}
    
    os.makedirs(results_dir, exist_ok=True)
    with open(os.path.join(results_dir, 'class_mapping.json'), 'w') as f:
        mapping = {str(v): k for k, v in class_to_idx.items()}
        json.dump(mapping, f, indent=4)
        
    with open(os.path.join(results_dir, 'dish_to_category.json'), 'w') as f:
        json.dump(dish_to_top, f, indent=4)
        
    filepaths = []
    labels = []
    hashes = []
    
    valid_exts = {'.jpg', '.jpeg', '.png', '.bmp', '.webp'}
    
    print("Scanning hierarchical dataset and computing hashes to prevent data leakage...")
    for top in top_levels:
        top_dir = os.path.join(dataset_path, top)
        dishes = [d for d in os.listdir(top_dir) if os.path.isdir(os.path.join(top_dir, d))]
        for dish in dishes:
            dish_dir = os.path.join(top_dir, dish)
            label_idx = class_to_idx[dish]
            for root, _, files in os.walk(dish_dir):
                for file in files:
                    ext = os.path.splitext(file)[1].lower()
                    if ext in valid_exts:
                        full_path = os.path.join(root, file)
                        try:
                            with Image.open(full_path) as img:
                                img = img.convert('RGB')
                                if img.size[0] < 32 or img.size[1] < 32:
                                    continue
                                h = str(imagehash.phash(img))
                                hashes.append(h)
                                filepaths.append(full_path)
                                labels.append(label_idx)
                        except Exception as e:
                            print(f"Skipping {full_path}: {e}")

    hashes = np.array(hashes)
    filepaths = np.array(filepaths)
    labels = np.array(labels)
    
    print("Grouping similar images...")
    groups = np.zeros(len(hashes), dtype=int)
    current_group = 0
    visited = set()
    
    for i in tqdm(range(len(hashes))):
        if i in visited:
            continue
        groups[i] = current_group
        visited.add(i)
        
        h1 = imagehash.hex_to_hash(hashes[i])
        for j in range(i + 1, len(hashes)):
            if j not in visited:
                h2 = imagehash.hex_to_hash(hashes[j])
                if h1 - h2 <= hash_threshold:
                    groups[j] = current_group
                    visited.add(j)
        current_group += 1
        
    gss = GroupShuffleSplit(n_splits=1, test_size=val_size, random_state=seed)
    train_idx, val_idx = next(gss.split(filepaths, labels, groups=groups))
    
    train_files, val_files = filepaths[train_idx], filepaths[val_idx]
    train_labels, val_labels = labels[train_idx], labels[val_idx]
    
    print(f"\nDataset Split Statistics:")
    print(f"Total unique groups (images without augmentations): {current_group}")
    print(f"Train images: {len(train_files)}")
    print(f"Validation images: {len(val_files)}")
    print(f"Total classes: {len(classes)}")
    
    split_info = {
        'train_files': train_files.tolist(),
        'val_files': val_files.tolist(),
        'train_labels': train_labels.tolist(),
        'val_labels': val_labels.tolist()
    }
    with open(os.path.join(results_dir, 'split_info.json'), 'w') as f:
        json.dump(split_info, f, indent=4)
        
    return train_files, val_files, train_labels, val_labels, classes
