import os
import argparse
import json
import torch
import torch.nn.functional as F
from PIL import Image
from core.models import FoodClassifier
from core.utils import get_device, get_transforms

def predict(args):
    device = get_device()
    
    # Load class mapping
    if not os.path.exists('results/class_mapping.json'):
        raise FileNotFoundError("class_mapping.json not found in results/. Please run training first.")
        
    with open('results/class_mapping.json', 'r') as f:
        mapping = json.load(f)
        
    num_classes = len(mapping)
    model = FoodClassifier(num_classes=num_classes, freeze_backbone=False)
    if not os.path.exists('checkpoints/best_food_classifier.pth'):
        raise FileNotFoundError("Model checkpoint not found. Please train first.")
    model.load_state_dict(torch.load('checkpoints/best_food_classifier.pth', map_location=device))
    model = model.to(device)
    model.eval()
    
    # Load and transform image
    _, val_transform = get_transforms(args.img_size)
    try:
        image = Image.open(args.image).convert('RGB')
    except Exception as e:
        print(f"Error loading image: {e}")
        return
        
    input_tensor = val_transform(image).unsqueeze(0).to(device)
    
    # Predict
    with torch.no_grad():
        outputs = model(input_tensor)
        probs = F.softmax(outputs, dim=1)
        conf, pred_idx = torch.max(probs, 1)
        
    conf_val = conf.item()
    pred_class = mapping[str(pred_idx.item())]
    
    if conf_val < args.threshold:
        print(f"Prediction: UNCERTAIN")
    else:
        print(f"Prediction: {pred_class.upper()}")
        
    print(f"Confidence: {conf_val * 100:.2f}%")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Predict single image")
    parser.add_argument("--image", type=str, required=True, help="Path to the image")
    parser.add_argument("--threshold", type=float, default=0.7, help="Confidence threshold")
    parser.add_argument("--img_size", type=int, default=288, help="Image size")
    
    args = parser.parse_args()
    predict(args)
