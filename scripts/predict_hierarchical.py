import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import argparse
import json
import torch
from torchvision import transforms
from PIL import Image
from core.models import FoodClassifier
from core.utils import get_device
import shutil

def get_inference_transforms(img_size=288):
    return transforms.Compose([
        transforms.Resize((img_size + 32, img_size + 32)),
        transforms.CenterCrop((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

def predict_hierarchical(args):
    device = get_device()
    
    with open(args.class_mapping, 'r') as f:
        class_mapping = json.load(f)
    
    with open(args.dish_category, 'r') as f:
        dish_to_category = json.load(f)
        
    num_classes = len(class_mapping)
    model = FoodClassifier(num_classes=num_classes, architecture=args.model)
    
    if not os.path.exists(args.weights):
        raise ValueError(f"Weights file {args.weights} not found.")
        
    model.load_state_dict(torch.load(args.weights, map_location=device))
    model = model.to(device)
    model.eval()
    
    transform = get_inference_transforms(args.img_size)
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    valid_exts = {'.jpg', '.jpeg', '.png', '.bmp', '.webp'}
    test_images = []
    for root, _, files in os.walk(args.input_dir):
        for file in files:
            ext = os.path.splitext(file)[1].lower()
            if ext in valid_exts:
                test_images.append(os.path.join(root, file))
                
    print(f"Found {len(test_images)} images to predict.")
    
    with torch.no_grad():
        for img_path in test_images:
            try:
                img = Image.open(img_path).convert('RGB')
                input_tensor = transform(img).unsqueeze(0).to(device)
                outputs = model(input_tensor)
                _, preds = torch.max(outputs, 1)
                
                pred_idx = str(preds.item())
                dish_name = class_mapping[pred_idx]
                category = dish_to_category[dish_name]
                
                # output structure: results/veg/dish_name/
                target_dir = os.path.join(args.output_dir, category, dish_name)
                os.makedirs(target_dir, exist_ok=True)
                
                target_path = os.path.join(target_dir, os.path.basename(img_path))
                shutil.copy2(img_path, target_path)
                
            except Exception as e:
                print(f"Error predicting {img_path}: {e}")
                
    print(f"Predictions saved hierarchically in {args.output_dir}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Predict images hierarchically (category -> dish)')
    parser.add_argument('--input_dir', type=str, required=True, help='Directory of test images')
    parser.add_argument('--output_dir', type=str, default='results/predictions_hierarchical', help='Output folder')
    parser.add_argument('--weights', type=str, required=True, help='Path to model weights')
    parser.add_argument('--class_mapping', type=str, required=True, help='Path to class_mapping.json')
    parser.add_argument('--dish_category', type=str, required=True, help='Path to dish_to_category.json')
    parser.add_argument('--model', type=str, default='mobilenet_v3_large', help='Model architecture')
    parser.add_argument('--img_size', type=int, default=288, help='Image resolution')
    
    args = parser.parse_args()
    predict_hierarchical(args)
