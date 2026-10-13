# 🌱 Vegan vs. Vegetarian (Meat-Free) Food Classifier

A specialized deep learning vision package designed to solve the **Lacto-Ovo Vegetarian vs. Strict Vegan** discrimination task across Middle Eastern / Arabic cuisine.

**Meat, poultry, fish, and seafood dishes are strictly excluded.** This package evaluates only the meatless cohort of dishes to detect whether a prepared dish is:
1. **Vegan (100% Plant-Based)**: Free from all animal products and byproducts.
2. **Vegetarian (Lacto-Ovo)**: Meat-free, but contains dairy (cheese, milk, yogurt, cream), eggs, butter, or clarified butter (*samneh* / ghee).

---

## 📊 Dataset Breakdown (15 Meatless Dishes, 704 Images)

The dataset contains **704 images** across 15 dishes, with a balanced **1.40 : 1** class ratio:

### 1. Vegan (100% Plant-Based — 9 Dishes, 411 Images)
- **Hummus** (61 images) — Chickpeas, sesame tahini, olive oil, garlic.
- **Tamees Bread** (53 images) — Plain tandoor wheat flatbread.
- **Za’atar Manakish** (52 images) — Flatbread with wild thyme, sesame, olive oil.
- **Okra Tagine** (45 images) — *Bamya bil-Zayt* (okra in tomato/olive oil stew).
- **Sesame Brittle** (45 images) — *Simsimiya* (roasted sesame in sugar syrup).
- **Spinach Fatayer** (44 images) — Levantine spinach pastries in olive oil dough.
- **Onion Soup** (42 images) — Plant-based caramelized onion soup.
- **Pomegranate Fattoush** (37 images) — Fresh mixed vegetable salad with olive oil & pomegranate molasses.
- **Spinach Stew** (32 images) — *Sabanegh bil-Zayt* (meatless spinach & chickpea stew).

### 2. Vegetarian (Dairy / Eggs / Ghee — 6 Dishes, 293 Images)
- **Fatayer Cheese** (86 images) — Stuffed with Akkawi, Halloumi, or Nabulsi dairy cheese.
- **Saffron Cake** (59 images) — Sponge cake with eggs, soaked in saffron condensed milk & cream.
- **Rice Pudding** (57 images) — Whole cow's milk, cream (*ashta*), and sugar.
- **Pistachio Sweet** (57 images) — Sweet cheese pastries (*Halawet el Jibn*) or *ashta* cream confections.
- **Pistachio Tamriya** (21 images) — Date confections pan-roasted in clarified butter (*samneh* / ghee).
- **Tamriya** (13 images) — Authentic Saudi roasted date confections in *samneh*.

---

## 📈 Benchmark Performance & YOLOv8-Seg Ablation

Evaluated using a three-way 70/15/15 hash-stratified split (preventing near-duplicate leakage) and external real-world restaurant photos:

| Pipeline Variant | Val Acc | Test Acc | Restaurant Acc (24 imgs) | Key Impact |
| :--- | :---: | :---: | :---: | :--- |
| **Uncropped Baseline** | **88.79%** | **85.05%** | **83.33%** (20/24) | Baseline whole-plate model |
| **YOLOv8-Seg Cropped** | **86.54%** | **85.58%** | **95.83%** (23/24) | **+12.50% Jump**; solves Za'atar glare |

### Real-World Restaurant Breakdown (24 Unseen Photos):
- **Hummus (Vegan)**: 5/5 (100.0%) in both models
- **Okra Tagine (Vegan)**: 4/4 (100.0%) in both models
- **Tamees Bread (Vegan)**: 5/5 (100.0%) in both models
- **Cheese Fatayer (Vegetarian)**: 4/5 (80.0%) in both models
- **Za’atar Manaqish (Vegan)**: 2/5 (40.0%) Baseline ➔ **5/5 (100.0%) with YOLOv8-Seg** (+60% recovery!)

---

## 📁 Package Structure

```
Vegan_vs_Vegetarian_Classifier/
├── README.md               # Package guide & documentation
├── dataset/ -> symlink     # Points to datasets/vegan_vs_veg/ (704 images)
├── checkpoints/
│   └── mobilenet_v3_large_vegan_veg_best.pth  # Best trained model weights
├── core/
│   ├── models.py           # MobileNetV3-Large binary classifier
│   ├── utils.py            # Dataset loader, device detection, transforms
│   └── vegan_taxonomy.json # Complete ingredient & dietary metadata
├── scripts/
│   ├── predict.py          # Single-image CLI inference tool with dietary warnings
│   ├── evaluate.py         # Evaluation on val or test splits
│   └── evaluate_restaurant.py # Real-world restaurant test evaluation
└── results/
    ├── class_mapping.json  # {"0": "vegan", "1": "vegetarian"}
    ├── split_info.json     # 70/15/15 train/val/test file assignments
    ├── training_curves.png # Loss & accuracy convergence plots
    ├── test/               # Confusion matrix & test reports
    └── restaurant_evaluation/ # Real-world restaurant predictions CSV
```

---

## 🚀 Quickstart Usage

### 1. Predict on a Single Food Image (with YOLOv8-Seg Cropping)
```bash
# Recommended YOLOv8-Seg pipeline (95.83% accuracy):
.venv/bin/python Vegan_vs_Vegetarian_Classifier/scripts/predict.py \
    --image "datasets/Restaurant testing data set/hummus/Hummus-di-ceci.webp" --yolo_seg
```

### 2. Predict on Cheese Fatayer (Dairy Warning)
```bash
.venv/bin/python Vegan_vs_Vegetarian_Classifier/scripts/predict.py \
    --image "datasets/Restaurant testing data set/cheese fatayer/images.jpg" --yolo_seg
```

### 3. Evaluate Held-Out Test Set
```bash
.venv/bin/python Vegan_vs_Vegetarian_Classifier/scripts/evaluate.py --split test
```

### 4. Evaluate Real Restaurant Dishes (Side-by-Side Comparison)
```bash
.venv/bin/python Vegan_vs_Vegetarian_Classifier/scripts/evaluate_restaurant.py --all
```
