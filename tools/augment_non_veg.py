import os
import random
from PIL import Image, ImageEnhance, ImageOps
from tqdm import tqdm

DATASET_DIR = "results/categorized_dataset"
VEG_DIR = os.path.join(DATASET_DIR, "veg")
NON_VEG_DIR = os.path.join(DATASET_DIR, "non_veg")

def get_image_files(directory):
    valid_exts = {'.jpg', '.jpeg', '.png', '.bmp', '.webp'}
    files = []
    for f in os.listdir(directory):
        if os.path.splitext(f)[1].lower() in valid_exts:
            files.append(os.path.join(directory, f))
    return files

def augment_image(img):
    """Applies a random augmentation to a PIL Image."""
    # 1. Random horizontal flip
    if random.random() > 0.5:
        img = ImageOps.mirror(img)
        
    # 2. Random rotation (-15 to 15 degrees)
    if random.random() > 0.5:
        angle = random.uniform(-15, 15)
        img = img.rotate(angle)
        
    # 3. Random brightness
    if random.random() > 0.5:
        enhancer = ImageEnhance.Brightness(img)
        img = enhancer.enhance(random.uniform(0.8, 1.2))
        
    return img

def main():
    print("Checking dataset class balance...")
    veg_files = get_image_files(VEG_DIR)
    non_veg_files = get_image_files(NON_VEG_DIR)
    
    veg_count = len(veg_files)
    non_veg_count = len(non_veg_files)
    
    print(f"Veg count: {veg_count}")
    print(f"Non-Veg count: {non_veg_count}")
    
    if non_veg_count >= veg_count:
        print("Non-veg is already balanced or has more images than veg. No augmentation needed.")
        return
        
    deficit = veg_count - non_veg_count
    print(f"Augmenting {deficit} Non-Veg images to balance the dataset...")
    
    # Randomly select images to augment
    images_to_augment = random.choices(non_veg_files, k=deficit)
    
    for i, img_path in enumerate(tqdm(images_to_augment, desc="Augmenting")):
        try:
            with Image.open(img_path) as img:
                if img.mode != 'RGB':
                    img = img.convert('RGB')
                    
                aug_img = augment_image(img)
                
                # Save augmented image
                basename = os.path.basename(img_path)
                aug_filename = f"aug_{i}_{basename}"
                aug_path = os.path.join(NON_VEG_DIR, aug_filename)
                
                aug_img.save(aug_path, format='JPEG', quality=95)
        except Exception as e:
            print(f"Error augmenting {img_path}: {e}")
            
    # Verify new counts
    new_non_veg_count = len(get_image_files(NON_VEG_DIR))
    print(f"\nDone! New Non-Veg count: {new_non_veg_count}")
    print(f"Dataset is now balanced: {veg_count} Veg vs {new_non_veg_count} Non-Veg")

if __name__ == "__main__":
    main()
