import csv
import os
import shutil
from pathlib import Path

def get_true_label(filepath):
    dish = filepath.split('/')[-2].lower()
    if 'mandi' in dish: return 'non_veg'
    if 'shish' in dish: return 'non_veg'
    if 'meat' in dish: return 'non_veg'
    if 'grill' in dish: return 'non_veg'
    if 'shawarma' in dish: return 'non_veg'
    if 'okra' in dish: return 'non_veg'
    if 'cheese' in dish: return 'veg'
    if 'hummus' in dish: return 'veg'
    if 'tamees' in dish: return 'veg'
    if 'zaatar' in dish: return 'veg'
    return 'unknown'

csv_path = 'results/restaurant_testing_predictions.csv'
base_out_dir = 'results/arabic_prediction_images'
correct_dir = os.path.join(base_out_dir, 'correct')
incorrect_dir = os.path.join(base_out_dir, 'incorrect')

os.makedirs(correct_dir, exist_ok=True)
os.makedirs(incorrect_dir, exist_ok=True)

copied_count = 0
with open(csv_path, 'r', encoding='utf-8') as f:
    reader = csv.DictReader(f)
    for row in reader:
        filepath = row['file']
        prediction = row['prediction']
        true_label = get_true_label(filepath)
        
        # Determine if correct
        is_correct = (prediction == true_label)
        target_dir = correct_dir if is_correct else incorrect_dir
        
        # Create a unique filename so "images.jpg" doesn't overwrite
        dish_name = Path(filepath).parts[-2].replace(' ', '_')
        original_filename = Path(filepath).name
        
        # Add prediction info to the filename for easy viewing
        new_filename = f"{dish_name}_true-{true_label}_pred-{prediction}_{original_filename}"
        target_path = os.path.join(target_dir, new_filename)
        
        if os.path.exists(filepath):
            shutil.copy(filepath, target_path)
            copied_count += 1
        else:
            print(f"Warning: File not found {filepath}")

print(f"Done! Copied {copied_count} images to {base_out_dir} into 'correct' and 'incorrect' folders.")
