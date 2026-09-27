import os
import shutil
from tqdm import tqdm

def organize_dataset():
    source_dir = "/Users/mehdidar/Desktop/25_dishes"
    dest_dir = "/Users/mehdidar/Desktop/food_classifier/results/categorized_dataset"
    
    # Define mapping based on standard ingredients
    dish_mapping = {
        "cheese_pie": "veg",
        "grilled_kofta": "non_veg",
        "hummus": "veg",
        "kofta_stew": "non_veg",
        "mandi": "non_veg",
        "meat-madfoon": "non_veg",
        "meat_maqluba": "non_veg",
        "meat_pie": "non_veg",
        "mixed_grill": "non_veg",
        "okra_tagine": "veg", 
        "onion_soup": "veg", 
        "pistachio_sweet": "veg",
        "pistachio_tamriya": "veg",
        "pomegranate_fattoush": "veg",
        "rice_pudding": "veg",
        "saffron_cake": "veg",
        "sesame_brittle": "veg",
        "shawarma": "non_veg",
        "shish tawook": "non_veg",
        "shrimp_sayadiyah": "non_veg", 
        "spinach_fatayer": "veg",
        "spinach_stew": "veg", 
        "stuffed_pigeon": "non_veg",
        "tamees_bread": "veg",
        "tamriya": "veg",
        "za’atar_manakish": "veg"
    }

    if not os.path.exists(source_dir):
        print(f"Source directory {source_dir} not found.")
        return

    # Create destination directories
    veg_dir = os.path.join(dest_dir, "veg")
    non_veg_dir = os.path.join(dest_dir, "non_veg")
    os.makedirs(veg_dir, exist_ok=True)
    os.makedirs(non_veg_dir, exist_ok=True)

    print(f"Copying images to {dest_dir}...")
    
    copied = 0
    for category in os.listdir(source_dir):
        cat_path = os.path.join(source_dir, category)
        if not os.path.isdir(cat_path):
            continue
            
        target_class = dish_mapping.get(category)
        if not target_class:
            print(f"Warning: Unknown category '{category}', skipping.")
            continue
            
        target_dir = os.path.join(veg_dir if target_class == "veg" else non_veg_dir, category)
        os.makedirs(target_dir, exist_ok=True)
        
        # Copy files
        files = os.listdir(cat_path)
        for f in tqdm(files, desc=f"Copying {category} to {target_class}"):
            src_file = os.path.join(cat_path, f)
            if os.path.isfile(src_file):
                # Ensure unique filenames by prefixing with category
                dst_file = os.path.join(target_dir, f)
                shutil.copy2(src_file, dst_file)
                copied += 1

    print(f"Successfully copied {copied} images into veg/non_veg folders.")
    print(f"You can find the new dataset at: {dest_dir}")

if __name__ == "__main__":
    organize_dataset()
