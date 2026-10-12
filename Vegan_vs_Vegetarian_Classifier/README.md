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

## 📈 Benchmark Performance

Evaluated using a three-way 70/15/15 hash-stratified split (preventing near-duplicate leakage):

| Split | Images | Accuracy | Weighted Precision | Weighted Recall | Weighted F1 |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Validation Set** | 107 | **88.79%** | 0.8879 | 0.8879 | **0.8879** |
| **Held-Out Test Set** | 107 | **87.85%** | 0.8875 | 0.8785 | **0.8793** |
| **Real Restaurant Test** | 24 | **83.33%** | — | — | — |

### Test Split Performance by Class:
- **Vegan Precision**: **95%** (Recall 84%, F1 0.89)
- **Vegetarian Recall**: **93%** (Precision 81%, F1 0.87)

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

### 1. Predict on a Single Food Image
```bash
.venv/bin/python Vegan_vs_Vegetarian_Classifier/scripts/predict.py --image "datasets/Restaurant testing data set/hummus/Hummus-di-ceci.webp"
```
**Output:**
```
=======================================================
VEGAN VS VEGETARIAN CLASSIFICATION RESULT
=======================================================
Image File:   Hummus-di-ceci.webp
Prediction:   VEGAN
Confidence:   80.68%
-------------------------------------------------------
Probability [Vegan]:       80.68%
Probability [Vegetarian]:  19.32%

🌱 Dietary Note: 100% plant-based (Zero dairy, eggs, or animal fats). Suitable for vegans.
=======================================================
```

### 2. Predict on Cheese Fatayer (Dairy Pastry)
```bash
.venv/bin/python Vegan_vs_Vegetarian_Classifier/scripts/predict.py --image "datasets/Restaurant testing data set/cheese fatayer/images.jpg"
```
**Output:**
```
=======================================================
VEGAN VS VEGETARIAN CLASSIFICATION RESULT
=======================================================
Image File:   images.jpg
Prediction:   VEGETARIAN
Confidence:   85.06%
-------------------------------------------------------
Probability [Vegan]:       14.94%
Probability [Vegetarian]:  85.06%

⚠️  Dietary Note: This dish contains animal byproducts (Dairy, Cheese, Eggs, or Ghee/Samneh). Not suitable for strict vegans.
=======================================================
```

### 3. Evaluate Held-Out Test Set
```bash
.venv/bin/python Vegan_vs_Vegetarian_Classifier/scripts/evaluate.py --split test
```

### 4. Evaluate Real Restaurant Dishes
```bash
.venv/bin/python Vegan_vs_Vegetarian_Classifier/scripts/evaluate_restaurant.py
```
