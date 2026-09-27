import os
import csv
import shutil

# Paths
csv_path = 'results_mobilenet_v3_large/splits/test.csv'
output_dir = 'datasets/test_dataset_dishes'

def main():
    if not os.path.exists(csv_path):
        print(f"Error: Could not find {csv_path}")
        return

    os.makedirs(output_dir, exist_ok=True)
    copied_count = 0

    with open(csv_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            img_path = row['image_path']
            
            if not os.path.exists(img_path):
                print(f"Warning: File not found: {img_path}")
                continue
                
            # Extract class, dish name and file name from the path
            # Example path: results/categorized_dataset/non_veg/grilled_kofta/9_ed3d5f.jpg
            parts = img_path.split('/')
            class_name = parts[-3]
            dish_name = parts[-2]
            file_name = parts[-1]
            
            # Create destination folder (e.g. test_dataset_dishes/non_veg/grilled_kofta)
            dest_folder = os.path.join(output_dir, class_name, dish_name)
            os.makedirs(dest_folder, exist_ok=True)
            
            # Copy the file
            dest_path = os.path.join(dest_folder, file_name)
            shutil.copy2(img_path, dest_path)
            copied_count += 1
            
    print(f"Successfully copied {copied_count} test images to '{output_dir}' organized by dish.")

if __name__ == '__main__':
    main()
