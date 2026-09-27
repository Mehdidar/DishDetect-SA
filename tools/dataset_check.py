import os
import glob
import json
import argparse
from collections import defaultdict
from PIL import Image
import imagehash
from tqdm import tqdm
import torch

def check_dataset(dataset_path, hash_threshold=5):
    print(f"Checking dataset at: {dataset_path}")
    if not os.path.exists(dataset_path):
        print(f"Error: Dataset path '{dataset_path}' does not exist.")
        return {}
        
    classes = [d for d in os.listdir(dataset_path) if os.path.isdir(os.path.join(dataset_path, d))]
    classes.sort()
    
    # Print Hardware Information
    print(f"PyTorch version: {torch.__version__}")
    print(f"MPS availability: {torch.backends.mps.is_available()}")
    print(f"CUDA availability: {torch.cuda.is_available()}")
    
    report = {
        'total_images': 0,
        'class_counts': {c: 0 for c in classes},
        'corrupted_images': [],
        'unsupported_formats': [],
        'small_images': [], # smaller than 128x128
        'near_duplicate_groups': [],
        'class_imbalance_ratio': 0.0,
        'empty_folders': []
    }
    


    supported_exts = {'.jpg', '.jpeg', '.png', '.bmp', '.webp'}
    image_data = [] # stores (path, phash, cls_name)
    
    for cls_name in classes:
        cls_dir = os.path.join(dataset_path, cls_name)
        if not os.path.exists(cls_dir):
            report['empty_folders'].append(cls_name)
            continue
            
        files = []
        for root, _, fnames in os.walk(cls_dir):
            for fname in fnames:
                files.append(os.path.join(root, fname))
                
        if len(files) == 0:
            report['empty_folders'].append(cls_name)
            
        for f in tqdm(files, desc=f"Scanning {cls_name}"):
            if not os.path.isfile(f):
                continue
                
            ext = os.path.splitext(f)[1].lower()
            if ext not in supported_exts:
                report['unsupported_formats'].append(f)
                continue
                
            try:
                with Image.open(f) as img:
                    img.verify()
                
                with Image.open(f) as img:
                    if img.mode != 'RGB':
                        img = img.convert('RGB')
                        
                    width, height = img.size
                    if width < 128 or height < 128:
                        report['small_images'].append(f)
                        
                    h = imagehash.phash(img)
                    image_data.append((f, h, cls_name))
                    
                    report['total_images'] += 1
                    report['class_counts'][cls_name] += 1
            except Exception:
                report['corrupted_images'].append(f)

    if report['class_counts']:
        counts = list(report['class_counts'].values())
        max_c = max(counts)
        min_c = min(counts)
        if min_c > 0:
            report['class_imbalance_ratio'] = max_c / min_c
        
    # Find duplicates
    assigned = [False] * len(image_data)
    for i in tqdm(range(len(image_data)), desc="Detecting duplicates"):
        if assigned[i]: continue
        current_group = [image_data[i][0]]
        assigned[i] = True
        
        for j in range(i+1, len(image_data)):
            if not assigned[j] and image_data[i][1] - image_data[j][1] <= hash_threshold:
                current_group.append(image_data[j][0])
                assigned[j] = True
                
        if len(current_group) > 1:
            report['near_duplicate_groups'].append(current_group)
            
    print("\nDataset Check Summary:")
    print(f"Total valid images: {report['total_images']}")
    for cls_name, count in report['class_counts'].items():
        print(f"{cls_name} images: {count}")
    print(f"Corrupted images: {len(report['corrupted_images'])}")
    print(f"Unsupported formats: {len(report['unsupported_formats'])}")
    print(f"Small images (<128px): {len(report['small_images'])}")
    print(f"Near-duplicate groups found: {len(report['near_duplicate_groups'])} (Threshold={hash_threshold})")
    
    os.makedirs('results', exist_ok=True)
    with open('results/dataset_report.json', 'w') as f:
        json.dump(report, f, indent=4)
        
    print("\nDetailed report saved to 'results/dataset_report.json'")
    return report

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Check dataset integrity and statistics.")
    parser.add_argument("--dataset", type=str, required=True, help="Path to the dataset directory")
    parser.add_argument("--hash_threshold", type=int, default=5, help="Hamming distance threshold for near-duplicates (default: 5)")
    args = parser.parse_args()
    check_dataset(args.dataset, args.hash_threshold)
