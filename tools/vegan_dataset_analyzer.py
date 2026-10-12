"""
vegan_dataset_analyzer.py
Audits the 25 core Arabic dishes in the dataset and generates
the exact image counts and distribution for:
1. Vegetarian vs Non-Vegetarian (Legacy)
2. Vegan vs Non-Vegan (New Target)
"""

import os
import json

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
DATASET_SRC = os.path.join(ROOT, "datasets", "categorized_dataset")

# Ingredient & Dietary Mapping for all 25 dishes
DISH_TAXONOMY = {
    # ── Formerly "Vegetarian" (15 dishes)
    "hummus": {
        "legacy_diet": "veg",
        "new_diet": "vegan",
        "key_ingredients": "Chickpeas, tahini (sesame), olive oil, lemon juice, garlic",
        "animal_byproduct": None,
        "rationale": "100% plant-based legumes and oil. Naturally vegan."
    },
    "pomegranate_fattoush": {
        "legacy_diet": "veg",
        "new_diet": "vegan",
        "key_ingredients": "Romaine, tomatoes, cucumbers, toasted pita, sumac, pomegranate molasses, olive oil",
        "animal_byproduct": None,
        "rationale": "100% plant-based salad with olive oil dressing. Naturally vegan."
    },
    "za’atar_manakish": {
        "legacy_diet": "veg",
        "new_diet": "vegan",
        "key_ingredients": "Flatbread dough (flour, yeast, water, olive oil), dried thyme, sesame, sumac, olive oil",
        "animal_byproduct": None,
        "rationale": "Authentic za'atar flatbread uses only olive oil and herbs; no dairy or eggs."
    },
    "spinach_fatayer": {
        "legacy_diet": "veg",
        "new_diet": "vegan",
        "key_ingredients": "Dough (flour, water, olive oil), spinach, onions, sumac, lemon juice, pine nuts",
        "animal_byproduct": None,
        "rationale": "Traditional Middle Eastern spinach pies contain no dairy or egg wash."
    },
    "tamees_bread": {
        "legacy_diet": "veg",
        "new_diet": "vegan",
        "key_ingredients": "Flour, water, yeast, salt, vegetable oil (tandoor-baked)",
        "animal_byproduct": None,
        "rationale": "Plain Saudi Tamees is simple leavened wheat bread. 100% vegan."
    },
    "okra_tagine": {
        "legacy_diet": "veg",
        "new_diet": "vegan",
        "key_ingredients": "Baby okra pods, tomato sauce, garlic, coriander, olive oil",
        "animal_byproduct": None,
        "rationale": "Bamya bil-Zayt (oil-based okra stew without meat) is 100% plant-based."
    },
    "spinach_stew": {
        "legacy_diet": "veg",
        "new_diet": "vegan",
        "key_ingredients": "Spinach, chickpeas, garlic, cilantro, onions, olive oil",
        "animal_byproduct": None,
        "rationale": "Sabanegh bil-Zayt is 100% plant-based."
    },
    "sesame_brittle": {
        "legacy_diet": "veg",
        "new_diet": "vegan",
        "key_ingredients": "Toasted sesame seeds, sugar syrup, glucose, lemon juice, rose water",
        "animal_byproduct": None,
        "rationale": "Traditional Simsimiya uses caramelized sugar syrup (no dairy, no honey)."
    },
    "onion_soup": {
        "legacy_diet": "veg",
        "new_diet": "vegan",
        "key_ingredients": "Onions, olive oil, vegetable broth, flour, cumin",
        "animal_byproduct": None,
        "rationale": "Plant-based Arabic Shorbat Basal without cream/butter."
    },
    "fatayer cheese": {
        "legacy_diet": "veg",
        "new_diet": "non_vegan",
        "key_ingredients": "Pastry dough, Akkawi/Halloumi/Nabulsi cheese, eggs/butter",
        "animal_byproduct": "Dairy (Cheese, butter, egg wash)",
        "rationale": "Core filling is dairy cheese."
    },
    "rice_pudding": {
        "legacy_diet": "veg",
        "new_diet": "non_vegan",
        "key_ingredients": "Short-grain rice, whole cow's milk, sugar, cream (ashta)",
        "animal_byproduct": "Dairy (Cow's milk, heavy cream)",
        "rationale": "Liquid base is whole dairy milk."
    },
    "saffron_cake": {
        "legacy_diet": "veg",
        "new_diet": "non_vegan",
        "key_ingredients": "Flour, eggs, whole milk, condensed milk, evaporated milk, saffron cream",
        "animal_byproduct": "Eggs & Dairy (Milk, condensed milk, cream)",
        "rationale": "High content of eggs, condensed milk, and dairy cream."
    },
    "tamriya": {
        "legacy_diet": "veg",
        "new_diet": "non_vegan",
        "key_ingredients": "Dates, toasted flour, clarified butter (samneh / ghee), cardamom",
        "animal_byproduct": "Dairy (Clarified butter / samneh / ghee)",
        "rationale": "Authentic Saudi Tamriya is pan-roasted in dairy samneh/ghee."
    },
    "pistachio_tamriya": {
        "legacy_diet": "veg",
        "new_diet": "non_vegan",
        "key_ingredients": "Dates, samneh / butter, ground pistachios",
        "animal_byproduct": "Dairy (Clarified butter / ghee)",
        "rationale": "Contains dairy butter/samneh in the date paste."
    },
    "pistachio_sweet": {
        "legacy_diet": "veg",
        "new_diet": "non_vegan",
        "key_ingredients": "Semolina, sweet cheese, ashta (clotted dairy cream), clarified butter (samneh)",
        "animal_byproduct": "Dairy (Sweet cheese, ashta cream, ghee)",
        "rationale": "Contains clotted dairy cream and ghee."
    },

    # ── Formerly "Non-Vegetarian" (10 dishes, all remain NON-VEGAN)
    "grilled_kofta": {
        "legacy_diet": "non_veg",
        "new_diet": "non_vegan",
        "key_ingredients": "Ground lamb/beef, onions, parsley, spices",
        "animal_byproduct": "Red Meat (Lamb/Beef)",
        "rationale": "Direct meat product."
    },
    "shawarma": {
        "legacy_diet": "non_veg",
        "new_diet": "non_vegan",
        "key_ingredients": "Spiced chicken/lamb/beef slices, garlic paste, pita",
        "animal_byproduct": "Meat / Poultry",
        "rationale": "Direct meat product."
    },
    "stuffed_pigeon": {
        "legacy_diet": "non_veg",
        "new_diet": "non_vegan",
        "key_ingredients": "Whole squab (pigeon), rice/freekeh, spices, giblets",
        "animal_byproduct": "Poultry (Pigeon)",
        "rationale": "Direct poultry product."
    },
    "kofta_stew": {
        "legacy_diet": "non_veg",
        "new_diet": "non_vegan",
        "key_ingredients": "Meatballs (beef/lamb), tomato sauce, potatoes",
        "animal_byproduct": "Red Meat (Beef/Lamb)",
        "rationale": "Direct meat product."
    },
    "meat-madfoon": {
        "legacy_diet": "non_veg",
        "new_diet": "non_vegan",
        "key_ingredients": "Slow-cooked foil-wrapped lamb, spiced rice",
        "animal_byproduct": "Red Meat (Lamb)",
        "rationale": "Direct meat product."
    },
    "shish tawook": {
        "legacy_diet": "non_veg",
        "new_diet": "non_vegan",
        "key_ingredients": "Marinated chicken skewers, yogurt, garlic",
        "animal_byproduct": "Poultry & Dairy (Chicken, yogurt marinade)",
        "rationale": "Direct chicken product."
    },
    "meat_pie": {
        "folder_alias": "meat fatayer",
        "legacy_diet": "non_veg",
        "new_diet": "non_vegan",
        "key_ingredients": "Pastry dough, spiced minced beef/lamb",
        "animal_byproduct": "Red Meat",
        "rationale": "Direct meat product."
    },
    "mixed_grill": {
        "folder_alias": "Mashwiaat",
        "legacy_diet": "non_veg",
        "new_diet": "non_vegan",
        "key_ingredients": "Assorted grilled meats (lamb kebab, tawook, chops)",
        "animal_byproduct": "Red Meat & Poultry",
        "rationale": "Direct meat product."
    },
    "shrimp_sayadiyah": {
        "legacy_diet": "non_veg",
        "new_diet": "non_vegan",
        "key_ingredients": "Caramelized onion rice, seasoned shrimp",
        "animal_byproduct": "Seafood (Shrimp / Crustaceans)",
        "rationale": "Direct seafood product."
    },
    "meat_maqluba": {
        "legacy_diet": "non_veg",
        "new_diet": "non_vegan",
        "key_ingredients": "Layered rice, lamb/chicken, fried vegetables",
        "animal_byproduct": "Red Meat / Poultry",
        "rationale": "Direct meat product."
    }
}

def count_images_in_dataset(base_path):
    dish_counts = {}
    valid_exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    
    for category in ["veg", "non_veg"]:
        cat_dir = os.path.join(base_path, category)
        if not os.path.exists(cat_dir):
            continue
        for dish in os.listdir(cat_dir):
            dish_dir = os.path.join(cat_dir, dish)
            if not os.path.isdir(dish_dir):
                continue
            count = 0
            for root, _, files in os.walk(dish_dir):
                for f in files:
                    if os.path.splitext(f)[1].lower() in valid_exts:
                        count += 1
            dish_counts[dish] = count
    return dish_counts

def main():
    print("=" * 70)
    print("VEGAN VS NON-VEGAN TAXONOMY & AUDIT FOR 25 ARABIC DISHES")
    print("=" * 70)
    
    counts = count_images_in_dataset(DATASET_SRC)
    
    vegan_dishes = [d for d, info in DISH_TAXONOMY.items() if info["new_diet"] == "vegan"]
    non_vegan_dishes = [d for d, info in DISH_TAXONOMY.items() if info["new_diet"] == "non_vegan"]
    
    total_vegan_images = 0
    total_non_vegan_images = 0
    
    print("\n1. VEGAN DISHES (100% Plant-Based — No Meat, Dairy, Eggs, or Ghee):")
    print(f"{'Dish Name':<25} | {'Images':<8} | {'Key Plant Ingredients'}")
    print("-" * 70)
    for d in sorted(vegan_dishes):
        cnt = counts.get(d, counts.get(d.replace(" ", "_"), counts.get(d.replace("_", " "), 0)))
        total_vegan_images += cnt
        print(f"{d:<25} | {cnt:<8} | {DISH_TAXONOMY[d]['key_ingredients'][:35]}...")
        
    print(f"\nTOTAL VEGAN DISHES: {len(vegan_dishes)} | TOTAL IMAGES: {total_vegan_images}")
    
    print("\n" + "=" * 70)
    print("2. NON-VEGAN DISHES (Contains Meat, Poultry, Seafood, Cheese, Milk, Eggs, or Ghee):")
    print(f"{'Dish Name':<25} | {'Images':<8} | {'Animal Byproduct / Reason'}")
    print("-" * 70)
    for d in sorted(non_vegan_dishes):
        alias = DISH_TAXONOMY[d].get("folder_alias", d)
        cnt = counts.get(alias, counts.get(d, counts.get(d.replace(" ", "_"), counts.get(d.replace("_", " "), 0))))
        total_non_vegan_images += cnt
        byproduct = DISH_TAXONOMY[d]['animal_byproduct']
        print(f"{d:<25} | {cnt:<8} | {byproduct}")
        
    print(f"\nTOTAL NON-VEGAN DISHES: {len(non_vegan_dishes)} | TOTAL IMAGES: {total_non_vegan_images}")
    
    print("\n" + "=" * 70)
    print("3. COMPARATIVE SHIFT SUMMARY")
    print("=" * 70)
    print(f"Legacy Vegetarian Model:   15 Veg Dishes vs 10 Non-Veg Dishes")
    print(f"New Vegan Model:           {len(vegan_dishes)} Vegan Dishes vs {len(non_vegan_dishes)} Non-Vegan Dishes")
    print(f"Vegan vs Non-Vegan Imbalance Ratio: {total_non_vegan_images / max(total_vegan_images, 1):.2f}:1")
    
    # Save mapping json
    out_json = os.path.join(ROOT, "core", "vegan_taxonomy.json")
    with open(out_json, "w") as f:
        json.dump(DISH_TAXONOMY, f, indent=4)
    print(f"\nSaved full taxonomy metadata to: {out_json}")

if __name__ == "__main__":
    main()
