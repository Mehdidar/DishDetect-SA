# 🍽️ DishDetect SA — Veg/Non-Veg Food Intelligence

> **A multi-stage deep learning pipeline for Arabic & Saudi restaurant food image classification.**

DishDetect SA classifies food images as **Vegetarian** or **Non-Vegetarian** using a robust six-stage pipeline combining web scraping, zero-shot AI filtering, YOLO segmentation, and a fine-tuned MobileNetV3-Large classifier.

---

## 🚀 Key Results

| Benchmark | Accuracy |
|---|---|
| Standard Test Set (534 images) | **92.70%** |
| Saudi Restaurant Real-World Test (55 images) | **92.73%** |
| German Cross-Domain Test (2,191 images) | **95.53%** |

---

## 🧠 Pipeline Stages

| Stage | Tool | Purpose |
|---|---|---|
| 1 | Requests / Selenium / BeautifulSoup | Image data collection |
| 2 | OpenAI Zero-Shot Classification | Initial food image filtering |
| 3 | Pillow (PIL) | Image pre-processing |
| 4 | CLIP ViT-B/32 | Veg/Non-Veg first-pass classification |
| 5 | YOLOv8n-seg | Object detection & background removal |
| 6 | MobileNetV3-Large | Final fine-tuned classification |

---

## 📁 Project Structure

```
DishDetect SA/
├── core/               # Model factory & shared utilities
│   ├── models/         # MobileNetV3 & EfficientNet definitions
│   └── utils.py        # Dataset, transforms, helpers
├── scripts/            # Training, evaluation & prediction pipeline
├── tools/              # Dataset diagnostics & reporting
├── app/                # Tkinter GUI for live predictions
├── docs/               # Generated reports & documentation
├── datasets/           # Training / validation / test splits
├── checkpoints/        # Saved model weights
└── results/            # Evaluation outputs, charts, confusion matrices
```

---

## ⚙️ Setup

1. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

2. **Validate your dataset:**
   ```bash
   python tools/dataset_check.py --dataset /path/to/dataset
   ```

3. **Train the model:**
   ```bash
   python scripts/train.py --dataset /path/to/dataset
   ```
   Trains in two stages (frozen backbone → full fine-tuning) and saves the best checkpoint to `checkpoints/`.

4. **Evaluate:**
   ```bash
   python scripts/evaluate.py --dataset /path/to/dataset
   ```
   Saves classification report and confusion matrix to `results/`.

5. **Predict a single image:**
   ```bash
   python scripts/predict.py --image /path/to/image.jpg --threshold 0.7
   ```

6. **Launch the GUI:**
   ```bash
   python app/ui.py
   ```

---

## 📊 Generate Reports

```bash
# Master project report (all phases, metrics, charts)
python scripts/generate_master_report.py

# Architecture comparison report (MobileNetV3 vs EfficientNet)
python scripts/generate_architecture_report.py

# Saudi Arabia pipeline flowchart + Word document
python scripts/generate_pipeline_doc.py
```

---

## 🏗️ Architecture

- **Backbone:** MobileNetV3-Large (ImageNet pre-trained, ~5M parameters)
- **Training:** Two-stage transfer learning — AdamW + Cosine Annealing LR
- **Preprocessing:** YOLOv8n-seg segmentation mask for background removal
- **Image Resolution:** 344 × 344 px
- **Loss:** CrossEntropyLoss with label smoothing (0.1) + class-frequency weights

---

## 📌 Known Limitations

- **Wrapped Shawarma** (pita-style) is the most persistent failure — exterior looks like flatbread
- **Mandi** struggles due to dominant rice visuals hiding the meat
- **90.9% restaurant accuracy ceiling** — bottleneck is training data, not model architecture

---

*DishDetect SA — Built with PyTorch, YOLOv8, CLIP & MobileNetV3*
