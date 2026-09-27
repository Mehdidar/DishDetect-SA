import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import os
import argparse
import json
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
from core.models import FoodClassifier
from core.utils import get_device, FoodDataset, get_transforms, EarlyStopping, plot_training_curves
from core.hierarchical_utils import prepare_hierarchical_data

def train_one_epoch(model, dataloader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    
    for inputs, labels in tqdm(dataloader, desc="Training", leave=False):
        inputs, labels = inputs.to(device), labels.to(device)
        
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item() * inputs.size(0)
        _, preds = torch.max(outputs, 1)
        correct += torch.sum(preds == labels.data).item()
        total += labels.size(0)
        
    epoch_loss = running_loss / total
    epoch_acc = correct / total
    return epoch_loss, epoch_acc

def validate_one_epoch(model, dataloader, criterion, device):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for inputs, labels in tqdm(dataloader, desc="Validating", leave=False):
            inputs, labels = inputs.to(device), labels.to(device)
            
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            
            running_loss += loss.item() * inputs.size(0)
            _, preds = torch.max(outputs, 1)
            correct += torch.sum(preds == labels.data).item()
            total += labels.size(0)
            
    epoch_loss = running_loss / total
    epoch_acc = correct / total
    return epoch_loss, epoch_acc

def train(args):
    device = get_device()
    
    results_dir = f'results_{args.model}'

    # Hierarchical Dish-wise data
    train_files, val_files, train_labels, val_labels, classes = prepare_hierarchical_data(
        args.dataset, val_size=0.2, seed=args.seed, hash_threshold=args.hash_threshold, results_dir=results_dir
    )
    
    # Calculate class weights for hierarchical dataset
    import numpy as np
    labels_list = train_labels.tolist()
    class_counts = [labels_list.count(c) for c in range(len(classes))]
    total_samples = len(train_labels)
    class_weights_np = total_samples / (len(classes) * np.array(class_counts))
    class_weights = torch.FloatTensor(class_weights_np)
    
    train_transform, val_transform = get_transforms(args.img_size)
    
    train_dataset = FoodDataset(train_files, train_labels, transform=train_transform)
    val_dataset = FoodDataset(val_files, val_labels, transform=val_transform)
    
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
    
    num_classes = len(class_weights)
    model = FoodClassifier(num_classes=num_classes, freeze_backbone=True, architecture=args.model)
    model = model.to(device)
    
    class_weights = class_weights.to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights, label_smoothing=0.1)
    
    history = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': []}
    early_stopping = EarlyStopping(patience=args.patience, path=f'checkpoints/{args.model}_best.pth')
    
    print("\n--- STAGE 1: Training Classifier ---")
    optimizer_stage1 = torch.optim.AdamW(model.model.classifier.parameters(), lr=args.lr, weight_decay=0.05)
    scheduler_stage1 = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer_stage1, mode='min', patience=2, factor=0.5)
    
    for epoch in range(1, args.stage1_epochs + 1):
        print(f"Epoch {epoch}/{args.stage1_epochs}")
        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer_stage1, device)
        val_loss, val_acc = validate_one_epoch(model, val_loader, criterion, device)
        
        scheduler_stage1.step(val_loss)
        
        print(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f} | Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")
        
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)
        
        early_stopping(val_loss, model)
        if early_stopping.early_stop:
            print("Early stopping triggered in Stage 1!")
            break
            
    model.load_state_dict(torch.load(f'checkpoints/{args.model}_best.pth'))
    
    print("\n--- STAGE 2: Fine-tuning Backbone ---")
    model.unfreeze_backbone()
    early_stopping.early_stop = False
    early_stopping.counter = 0
    
    optimizer_stage2 = torch.optim.AdamW(model.parameters(), lr=args.lr * 0.01, weight_decay=0.05)
    scheduler_stage2 = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer_stage2, T_max=args.stage2_epochs, eta_min=1e-6)
    
    for epoch in range(1, args.stage2_epochs + 1):
        print(f"Epoch {epoch}/{args.stage2_epochs}")
        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer_stage2, device)
        val_loss, val_acc = validate_one_epoch(model, val_loader, criterion, device)
        
        scheduler_stage2.step()
        
        print(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f} | Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")
        
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)
        
        early_stopping(val_loss, model)
        if early_stopping.early_stop:
            print("Early stopping triggered in Stage 2!")
            break
            
    print("\nTraining completed. Saving metrics and plots...")
    os.makedirs(results_dir, exist_ok=True)
    with open(os.path.join(results_dir, 'metrics.json'), 'w') as f:
        json.dump(history, f)
    
    plot_training_curves(history, os.path.join(results_dir, 'training_curves.png'))
    print("Done!")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Food Classifier")
    parser.add_argument("--dataset", type=str, required=True, help="Path to the dataset directory")
    parser.add_argument("--model", type=str, default='mobilenet_v3_large',
                        choices=['efficientnet_b2', 'mobilenet_v3_large', 'mobilenet_v3_large_yolo', 'mobilenet_v3_large_yolo_seg'],
                        help="Model architecture")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size")
    parser.add_argument("--num_workers", type=int, default=0, help="Number of data loading workers")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate for stage 1")
    parser.add_argument("--stage1_epochs", type=int, default=15, help="Max epochs for stage 1")
    parser.add_argument("--stage2_epochs", type=int, default=25, help="Max epochs for stage 2")
    parser.add_argument("--patience", type=int, default=7, help="Early stopping patience")
    parser.add_argument("--img_size", type=int, default=288, help="Image size")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--hash_threshold", type=int, default=5, help="Near-duplicate threshold")
    parser.add_argument("--use_three_way_split", action="store_true", help="Use 70/15/15 train/val/test split instead of 80/20 train/val")
    
    args = parser.parse_args()
    
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
        
    train(args)
