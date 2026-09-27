import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import os
import json
import time
import torch
from core.models import FoodClassifier

def count_parameters(model):
    return sum(p.numel() for p in model.parameters())

def get_file_size_mb(path):
    if os.path.exists(path):
        return os.path.getsize(path) / (1024 * 1024)
    return 0

def measure_inference(model, device, img_size):
    # Dummy input
    dummy_input = torch.randn(1, 3, img_size, img_size).to(device)
    
    # Warmup
    for _ in range(10):
        with torch.no_grad():
            model(dummy_input)
            
    if torch.backends.mps.is_available():
        torch.mps.synchronize()
        
    start_time = time.time()
    iters = 100
    for _ in range(iters):
        with torch.no_grad():
            model(dummy_input)
    
    if torch.backends.mps.is_available():
        torch.mps.synchronize()
        
    end_time = time.time()
    
    avg_time_ms = ((end_time - start_time) / iters) * 1000
    return avg_time_ms

def main():
    device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
    
    models = [
        ('mobilenet_v3_large', 224),
        ('efficientnet_b2', 288)
    ]
    
    metrics = {}
    
    for name, img_size in models:
        model = FoodClassifier(num_classes=2, freeze_backbone=False, architecture=name)
        ckpt_path = f'checkpoints/{name}_best.pth'
        
        if os.path.exists(ckpt_path):
            model.load_state_dict(torch.load(ckpt_path, map_location=device))
        model.to(device)
        model.eval()
        
        params = count_parameters(model)
        size_mb = get_file_size_mb(ckpt_path)
        
        # Measure MPS memory if possible
        mem_before = 0
        if torch.backends.mps.is_available():
            try:
                mem_before = torch.mps.current_allocated_memory() / (1024 * 1024)
            except:
                pass
                
        inf_time = measure_inference(model, device, img_size)
        
        mem_after = 0
        if torch.backends.mps.is_available():
            try:
                mem_after = torch.mps.current_allocated_memory() / (1024 * 1024)
            except:
                pass
                
        mem_usage = mem_after - mem_before if mem_after > mem_before else 0
        
        metrics[name] = {
            'parameters': params,
            'size_mb': size_mb,
            'inference_time_ms': inf_time,
            'gpu_memory_mb': mem_usage if mem_usage > 0 else "N/A"
        }
        
    print(json.dumps(metrics, indent=4))

if __name__ == "__main__":
    main()
