import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import os
import json
import torch
from torchvision import transforms
from torch.utils.data import DataLoader, Dataset
from PIL import Image
from tqdm import tqdm
from sklearn.metrics import classification_report
from core.models import FoodClassifier

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

def get_transforms(img_size):
    return transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

def main():
    device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
    
    with open('results_mobilenet_v3_large/split_info.json', 'r') as f:
        split_info = json.load(f)
        all_files = split_info.get('train_files', []) + split_info.get('val_files', [])
        
    print(f"Found {len(all_files)} total valid images in the dataset.")
    
    models = [
        ('mobilenet_v3_large', 224),
        ('efficientnet_b2', 288)
    ]
    
    for model_name, img_size in models:
        print(f"\n======================================")
        print(f"Generating Total Report for {model_name}")
        print(f"======================================")
        
        transform = get_transforms(img_size)
        dataset = InferenceDataset(all_files, transform)
        loader = DataLoader(dataset, batch_size=32, shuffle=False, num_workers=4)
        
        with open(f'results_{model_name}/class_mapping.json', 'r') as f:
            mapping = json.load(f)
            
        reverse_mapping = {v: int(k) for k, v in mapping.items()}
            
        model = FoodClassifier(num_classes=2, freeze_backbone=False, architecture=model_name)
        model.load_state_dict(torch.load(f'checkpoints/{model_name}_best.pth', map_location=device))
        model.to(device)
        model.eval()
        
        all_preds = []
        all_labels = []
        
        with torch.no_grad():
            for inputs, paths in tqdm(loader, desc=f"Evaluating"):
                inputs = inputs.to(device)
                outputs = model(inputs)
                _, preds = torch.max(outputs, 1)
                
                for p, fpath in zip(preds.cpu().numpy(), paths):
                    # Extract ground truth label from the file path
                    path_parts = fpath.split(os.sep)
                    gt_cls = None
                    for part in reversed(path_parts[:-1]):
                        if part in reverse_mapping:
                            gt_cls = reverse_mapping[part]
                            break
                            
                    if gt_cls is not None:
                        all_labels.append(gt_cls)
                        all_preds.append(p)
                        
        from sklearn.metrics import classification_report, confusion_matrix
        from core.utils import plot_confusion_matrix
        
        classes = [mapping[str(i)] for i in range(len(mapping))]
        report = classification_report(all_labels, all_preds, target_names=classes)
        cm = confusion_matrix(all_labels, all_preds)
        
        print(f"\n[Classification Report for {model_name} on ENTIRE Dataset]")
        print(report)
        
        report_path = f'results_{model_name}/total_classification_report.txt'
        with open(report_path, 'w') as f:
            f.write(f"Total Dataset Classification Report\n")
            f.write(f"Total Images Evaluated: {len(all_labels)}\n\n")
            f.write(report)
            
        plot_confusion_matrix(cm, classes, os.path.join(f'results_{model_name}', 'total_confusion_matrix.png'))
            
        print(f"Saved report to: {report_path}\n")

if __name__ == '__main__':
    main()
