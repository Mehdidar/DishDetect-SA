"""
generate_code_docs.py
---------------------
Generates a Word document explaining every file and function
in the food_classifier project.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from docx import Document
from docx.shared import Pt, RGBColor, Cm, Inches
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.enum.table import WD_TABLE_ALIGNMENT, WD_ALIGN_VERTICAL
from docx.oxml.ns import qn
from docx.oxml import OxmlElement
from datetime import datetime

OUTPUT_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "code_documentation.docx")

# ── Helpers ───────────────────────────────────────────────────────────────────

def set_cell_bg(cell, hex_color):
    tc = cell._tc
    tcPr = tc.get_or_add_tcPr()
    shd = OxmlElement("w:shd")
    shd.set(qn("w:val"),   "clear")
    shd.set(qn("w:color"), "auto")
    shd.set(qn("w:fill"),  hex_color.upper())
    tcPr.append(shd)

def add_borders(table):
    for tr in table._tbl.findall(qn("w:tr")):
        for tc in tr.findall(qn("w:tc")):
            tcPr = tc.get_or_add_tcPr()
            tcB  = OxmlElement("w:tcBorders")
            for side in ("top","left","bottom","right","insideH","insideV"):
                b = OxmlElement(f"w:{side}")
                b.set(qn("w:val"),   "single")
                b.set(qn("w:sz"),    "4")
                b.set(qn("w:space"), "0")
                b.set(qn("w:color"), "CCCCCC")
                tcB.append(b)
            tcPr.append(tcB)

def heading(doc, text, level=1, color="1F3864"):
    p = doc.add_heading(text, level=level)
    p.alignment = WD_ALIGN_PARAGRAPH.LEFT
    for run in p.runs:
        run.font.color.rgb = RGBColor.from_string(color)
    return p

def make_table(doc, headers, rows, hdr_bg="1F3864", alt_bg="EBF3FB", col_widths=None):
    table = doc.add_table(rows=1 + len(rows), cols=len(headers))
    table.alignment = WD_TABLE_ALIGNMENT.CENTER
    table.style = "Table Grid"
    for i, h in enumerate(headers):
        cell = table.rows[0].cells[i]
        cell.text = h
        set_cell_bg(cell, hdr_bg)
        for run in cell.paragraphs[0].runs:
            run.bold = True
            run.font.color.rgb = RGBColor(0xFF,0xFF,0xFF)
            run.font.size = Pt(10)
        cell.paragraphs[0].alignment = WD_ALIGN_PARAGRAPH.CENTER
    for ri, row_data in enumerate(rows):
        bg  = alt_bg if ri % 2 else "FFFFFF"
        row = table.rows[ri + 1]
        for ci, val in enumerate(row_data):
            cell = row.cells[ci]
            cell.text = str(val)
            set_cell_bg(cell, bg)
            cell.paragraphs[0].alignment = WD_ALIGN_PARAGRAPH.LEFT
            for run in cell.paragraphs[0].runs:
                run.font.size = Pt(10)
    add_borders(table)
    if col_widths:
        for i, w in enumerate(col_widths):
            for row in table.rows:
                row.cells[i].width = Cm(w)
    return table

def code_block(doc, code_text):
    """Add a light-grey shaded paragraph that looks like a code block."""
    p = doc.add_paragraph()
    p.paragraph_format.left_indent  = Cm(0.5)
    p.paragraph_format.space_before = Pt(4)
    p.paragraph_format.space_after  = Pt(4)
    # shade the paragraph
    pPr = p._p.get_or_add_pPr()
    shd = OxmlElement("w:shd")
    shd.set(qn("w:val"),   "clear")
    shd.set(qn("w:color"), "auto")
    shd.set(qn("w:fill"),  "F2F2F2")
    pPr.append(shd)
    run = p.add_run(code_text)
    run.font.name = "Courier New"
    run.font.size = Pt(9)
    run.font.color.rgb = RGBColor(0x1F,0x49,0x7D)
    return p

def section_box(doc, label, color="2E75B6"):
    """Coloured file-name banner."""
    p = doc.add_paragraph()
    run = p.add_run(f"  {label}  ")
    run.bold = True
    run.font.size = Pt(12)
    run.font.color.rgb = RGBColor(0xFF,0xFF,0xFF)
    pPr = p._p.get_or_add_pPr()
    shd = OxmlElement("w:shd")
    shd.set(qn("w:val"),   "clear")
    shd.set(qn("w:color"), "auto")
    shd.set(qn("w:fill"),  color.lstrip("#").upper())
    pPr.append(shd)
    return p

def bullet(doc, text, bold_pre=None):
    p = doc.add_paragraph(style="List Bullet")
    if bold_pre:
        r = p.add_run(bold_pre); r.bold = True
    p.add_run(text)
    return p

def note(doc, text):
    p = doc.add_paragraph()
    run = p.add_run(f"ℹ️  {text}")
    run.italic = True
    run.font.size = Pt(10)
    run.font.color.rgb = RGBColor(0x44,0x72,0xC4)
    p.paragraph_format.left_indent = Cm(0.5)
    return p

# ── Document ───────────────────────────────────────────────────────────────────
doc = Document()
sec = doc.sections[0]
sec.left_margin = sec.right_margin = Cm(2.5)
sec.top_margin  = sec.bottom_margin = Cm(2.0)

# ── Cover ─────────────────────────────────────────────────────────────────────
t = doc.add_paragraph()
t.alignment = WD_ALIGN_PARAGRAPH.CENTER
r = t.add_run("🍽️  Food Classifier — Code Documentation")
r.bold = True; r.font.size = Pt(22)
r.font.color.rgb = RGBColor.from_string("1F3864")

sub = doc.add_paragraph()
sub.alignment = WD_ALIGN_PARAGRAPH.CENTER
sub.add_run(f"Generated: {datetime.now().strftime('%B %d, %Y  %H:%M')}").italic = True

doc.add_paragraph()
note(doc, "This document explains every file, class, and function in the project — written for clarity, not just code comments.")
doc.add_paragraph()

# ── 1. Project Structure ───────────────────────────────────────────────────────
heading(doc, "1. Project Structure", 1)
doc.add_paragraph(
    "The project is organised into three main areas: core library code, "
    "executable scripts, and result folders. Here is the full file tree:"
)
doc.add_paragraph()
code_block(doc,
"""food_classifier/
├── core/
│   ├── models/
│   │   ├── __init__.py         ← FoodClassifier factory function
│   │   ├── mobilenet.py        ← MobileNet V3 Large model definition
│   │   └── efficientnet.py     ← EfficientNet B2 model definition
│   └── utils.py                ← Dataset, transforms, data-split, helpers
│
├── scripts/
│   ├── train.py                ← Main training loop (2-stage transfer learning)
│   ├── evaluate.py             ← Evaluate on val or internal test split
│   ├── evaluate_external.py    ← Evaluate on a separate external test folder
│   ├── split_dataset.py        ← Create 3-way train/val/test split (CLI)
│   ├── yolo_crop.py            ← YOLOv8 detection-based crop (old)
│   └── yolo_seg_crop.py        ← YOLOv8 segmentation-based crop (new, tighter)
│
├── updates/
│   ├── generate_report.py      ← Generate basic performance Word report
│   ├── generate_full_report.py ← Full YOLO-Seg pipeline Word report
│   └── generate_code_docs.py   ← THIS FILE — code documentation
│
├── results/                    ← Raw datasets and YOLO-cropped datasets
├── results_mobilenet_v3_large/ ← Training artifacts for MobileNet (no YOLO)
├── results_efficientnet_b2/    ← Training artifacts for EfficientNet
├── results_mobilenet_v3_large_yolo_seg/ ← NEW best model results
├── checkpoints/                ← Saved model weights (.pth files)
└── test_dataset_dishes/        ← External held-out test images (manually added)
""")
doc.add_paragraph()

# ── 2. core/models/__init__.py ────────────────────────────────────────────────
heading(doc, "2. core/models/__init__.py — Model Factory", 1)
section_box(doc, "core/models/__init__.py")
doc.add_paragraph()
doc.add_paragraph(
    "This file acts as a factory (a function that creates objects). "
    "It lets the rest of the code ask for a model by name without needing to "
    "import each model file separately."
)
doc.add_paragraph()
heading(doc, "FoodClassifier()  —  the factory function", 2, "2E75B6")
make_table(doc,
    headers=["Parameter", "Type", "Default", "Description"],
    rows=[
        ["num_classes",      "int",  "2",              "Number of output classes (veg / non_veg = 2)"],
        ["freeze_backbone",  "bool", "True",            "If True, only the classifier head trains (Stage 1)"],
        ["architecture",     "str",  "'efficientnet_b2'","Which backbone to build: 'mobilenet_v3_large', 'mobilenet_v3_large_yolo_seg', 'efficientnet_b2'"],
    ],
    col_widths=[4.0, 2.0, 3.5, 7.0],
)
doc.add_paragraph()
code_block(doc, "model = FoodClassifier(num_classes=2, freeze_backbone=True, architecture='mobilenet_v3_large_yolo_seg')")
note(doc, "All 'mobilenet_v3_large*' variants use the same MobileNetClassifier — the name difference is just for saving results to separate folders.")
doc.add_paragraph()

# ── 3. core/models/mobilenet.py ───────────────────────────────────────────────
heading(doc, "3. core/models/mobilenet.py — MobileNet V3 Large", 1)
section_box(doc, "core/models/mobilenet.py")
doc.add_paragraph()
doc.add_paragraph(
    "Defines the MobileNet V3 Large neural network used for classification. "
    "MobileNet V3 Large is a lightweight convolutional neural network pre-trained on "
    "ImageNet (1.2 million images, 1000 classes). We replace its final layer to classify our 2 food categories."
)
doc.add_paragraph()
heading(doc, "MobileNetClassifier class", 2, "2E75B6")
make_table(doc,
    headers=["Method", "What it does"],
    rows=[
        ["__init__(num_classes, freeze_backbone)",
         "Downloads/loads the pretrained MobileNet V3 Large weights. Replaces the last linear layer with a new one (output = num_classes). Optionally freezes all layers except the classifier."],
        ["freeze_backbone()",
         "Sets requires_grad=False on ALL layers, then re-enables it for the classifier head only. Used in Stage 1 of training."],
        ["unfreeze_backbone()",
         "Sets requires_grad=True on ALL layers. Called at the start of Stage 2 (fine-tuning)."],
        ["forward(x)",
         "Passes an input batch x through the network and returns raw logits (un-normalised scores for each class)."],
    ],
    col_widths=[5.5, 11.0],
)
doc.add_paragraph()
note(doc, "The original MobileNet classifier layer has 1280 input features. We keep all layers except the final Linear(1280 → 1000) which becomes Linear(1280 → 2).")
doc.add_paragraph()

# ── 4. core/models/efficientnet.py ────────────────────────────────────────────
heading(doc, "4. core/models/efficientnet.py — EfficientNet B2", 1)
section_box(doc, "core/models/efficientnet.py")
doc.add_paragraph(
    "Defines the EfficientNet B2 model — a slightly larger and more accurate alternative to MobileNet. "
    "The structure mirrors MobileNetClassifier exactly: pretrained backbone + replaced classifier head + freeze/unfreeze methods."
)
doc.add_paragraph()
make_table(doc,
    headers=["Detail", "Value"],
    rows=[
        ["Pretrained weights",  "EfficientNet_B2_Weights.IMAGENET1K_V1"],
        ["Replaced layer",      "model.classifier[1] → Linear(in_features → num_classes)"],
        ["Parameters (total)",  "~8.8 million"],
        ["Best val accuracy",   "89.3% (test split), 96% (full dataset)"],
    ],
    col_widths=[5.5, 11.0],
)
doc.add_paragraph()

# ── 5. core/utils.py ──────────────────────────────────────────────────────────
heading(doc, "5. core/utils.py — Shared Utilities", 1)
section_box(doc, "core/utils.py")
doc.add_paragraph("This is the backbone of the project. It contains all the helper code shared across training, evaluation, and data preparation.")
doc.add_paragraph()

heading(doc, "5.1  get_device()", 2, "2E75B6")
doc.add_paragraph(
    "Automatically selects the best available hardware for PyTorch:"
)
bullet(doc, "Apple MPS (Metal Performance Shaders) — for Apple Silicon Macs", "1st choice: ")
bullet(doc, "NVIDIA CUDA GPU", "2nd choice: ")
bullet(doc, "Standard CPU", "Fallback: ")
note(doc, "This project runs on Apple MPS (M-series chip). MPS is ~5–10× faster than CPU for deep learning.")
doc.add_paragraph()

heading(doc, "5.2  FoodDataset (class)", 2, "2E75B6")
doc.add_paragraph(
    "A custom PyTorch Dataset class. PyTorch needs a Dataset object to load images one by one "
    "during training. This class wraps a list of file paths and labels."
)
make_table(doc,
    headers=["Method", "What it does"],
    rows=[
        ["__init__(filepaths, labels, transform)",
         "Stores the list of image file paths, their integer class labels, and the transform pipeline to apply."],
        ["__len__()",
         "Returns total number of images — required by PyTorch's DataLoader."],
        ["__getitem__(idx)",
         "Opens one image file as PIL RGB, applies the transform, and returns (image_tensor, label). If the file is corrupt, returns a black zero-tensor so training doesn't crash."],
    ],
    col_widths=[5.5, 11.0],
)
doc.add_paragraph()
code_block(doc, "dataset = FoodDataset(file_paths, labels, transform=train_transform)")
doc.add_paragraph()

heading(doc, "5.3  get_transforms(img_size)", 2, "2E75B6")
doc.add_paragraph(
    "Returns two torchvision transform pipelines — one for training (with augmentation) and one for validation/test (clean)."
)
make_table(doc,
    headers=["Transform", "Applied to", "Purpose"],
    rows=[
        ["Resize(img_size+32)",          "Both",    "Make image slightly larger than target so we can crop"],
        ["RandomCrop(img_size)",         "Train",   "Randomly crop to final size — adds positional variety"],
        ["CenterCrop(img_size)",         "Val/Test","Deterministic crop from centre — no randomness"],
        ["RandomHorizontalFlip()",       "Train",   "Mirror image left↔right 50% of the time"],
        ["RandomVerticalFlip(p=0.2)",    "Train",   "Mirror image top↔bottom 20% of the time"],
        ["RandAugment()",                "Train",   "Random colour jitter, sharpening, rotation — strong augmentation"],
        ["GaussianBlur(p=0.2)",          "Train",   "Slight blur applied 20% of the time — reduces over-reliance on fine details"],
        ["ToTensor()",                   "Both",    "Convert PIL image to PyTorch tensor (0–1 float)"],
        ["Normalize(mean, std)",         "Both",    "Standardise using ImageNet statistics so pretrained weights apply correctly"],
    ],
    col_widths=[4.5, 2.5, 9.5],
)
doc.add_paragraph()

heading(doc, "5.4  prepare_data() — 2-way split (legacy)", 2, "2E75B6")
doc.add_paragraph(
    "Splits a dataset into Train (80%) and Val (20%). Also computes perceptual hashes (pHash) "
    "to group near-duplicate images so they never appear in both train and val (preventing data leakage)."
)
bullet(doc, "Scans all images, computes pHash for each", "Step 1: ")
bullet(doc, "Groups images with Hamming distance ≤ hash_threshold into the same group", "Step 2: ")
bullet(doc, "Splits at the group level using GroupShuffleSplit — whole groups go to one split only", "Step 3: ")
bullet(doc, "Computes inverse-frequency class weights (to handle class imbalance)", "Step 4: ")
bullet(doc, "Saves split_info.json and class_mapping.json to results_dir", "Step 5: ")
doc.add_paragraph()

heading(doc, "5.5  prepare_data_three_way() — 3-way split (current)", 2, "2E75B6")
doc.add_paragraph(
    "The scientifically correct split: Train 70% / Val 15% / Test 15%. "
    "Performs the same pHash grouping as above, but splits into three sets instead of two. "
    "Stratified — each class keeps its proportional representation across all three splits."
)
note(doc, "The Test set from this function is held out completely — never used for gradient updates or checkpoint selection. It is only used for final reporting.")
doc.add_paragraph()

heading(doc, "5.6  EarlyStopping (class)", 2, "2E75B6")
doc.add_paragraph(
    "Monitors validation loss each epoch. If it doesn't improve for 'patience' consecutive epochs, "
    "training stops to prevent overfitting. Also saves the best checkpoint whenever a new lowest val loss is found."
)
make_table(doc,
    headers=["Parameter", "Default", "Description"],
    rows=[
        ["patience",   "5",  "How many epochs without improvement before stopping"],
        ["min_delta",  "0",  "Minimum improvement to count as 'better'"],
        ["path",       "checkpoints/best.pth", "Where to save the best model weights"],
    ],
    col_widths=[3.5, 3.5, 9.5],
)
doc.add_paragraph()

heading(doc, "5.7  plot_training_curves()", 2, "2E75B6")
doc.add_paragraph("Generates and saves a 2-panel matplotlib chart: Loss curves (left) and Accuracy curves (right) for both train and validation sets across all epochs. Saved as training_curves.png.")
doc.add_paragraph()

heading(doc, "5.8  plot_confusion_matrix()", 2, "2E75B6")
doc.add_paragraph("Generates and saves a heatmap of the confusion matrix showing true vs predicted labels. Numbers are printed inside each cell. Saved as confusion_matrix.png.")
doc.add_paragraph()

# ── 6. scripts/train.py ───────────────────────────────────────────────────────
heading(doc, "6. scripts/train.py — Training Script", 1)
section_box(doc, "scripts/train.py")
doc.add_paragraph(
    "The main training script. Implements a 2-stage transfer learning strategy — the industry standard "
    "approach when fine-tuning pretrained models on small datasets."
)
doc.add_paragraph()

heading(doc, "2-Stage Training Strategy", 2, "2E75B6")
make_table(doc,
    headers=["Stage", "What trains", "LR", "Epochs", "Scheduler", "Purpose"],
    rows=[
        ["Stage 1\n(Frozen)", "Classifier head only\n(~3,000 params)", "1e-3", "15",
         "ReduceLROnPlateau", "Quickly learn food features without corrupting pretrained backbone weights"],
        ["Stage 2\n(Unfrozen)", "All 4.2M parameters", "1e-5\n(100× smaller)", "25",
         "CosineAnnealingLR", "Fine-tune entire network gently — backbone adapts to food domain"],
    ],
    col_widths=[2.2, 3.5, 1.5, 1.8, 4.5, 4.5],
)
doc.add_paragraph()

heading(doc, "train_one_epoch(model, dataloader, criterion, optimizer, device)", 2, "2E75B6")
doc.add_paragraph("Runs one complete pass through the training data:")
bullet(doc, "Sets model to training mode (enables dropout, batch norm updates)")
bullet(doc, "For each batch: forward pass → compute loss → backpropagate → update weights")
bullet(doc, "Returns average loss and accuracy for the epoch")
doc.add_paragraph()

heading(doc, "validate_one_epoch(model, dataloader, criterion, device)", 2, "2E75B6")
doc.add_paragraph("Same as above but with torch.no_grad() — no gradient computation, no weight updates. Used to measure how well the model generalises to unseen data.")
doc.add_paragraph()

heading(doc, "Loss Function", 2, "2E75B6")
code_block(doc, "criterion = nn.CrossEntropyLoss(weight=class_weights, label_smoothing=0.1)")
make_table(doc,
    headers=["Option", "Purpose"],
    rows=[
        ["weight=class_weights",  "Penalises mistakes on minority classes more — handles class imbalance"],
        ["label_smoothing=0.1",   "Instead of training towards hard 0/1 labels, target 0.05/0.95 — reduces overconfidence and overfitting"],
    ],
    col_widths=[5.5, 11.0],
)
doc.add_paragraph()

heading(doc, "Command Line Arguments", 2, "2E75B6")
make_table(doc,
    headers=["Argument", "Default", "Description"],
    rows=[
        ["--dataset",            "(required)", "Path to training dataset folder"],
        ["--model",              "mobilenet_v3_large", "Architecture name — used for checkpoint and results folder name"],
        ["--batch_size",         "32",   "Images processed per weight update"],
        ["--lr",                 "1e-3", "Learning rate for Stage 1"],
        ["--stage1_epochs",      "15",   "Max epochs for Stage 1 (frozen backbone)"],
        ["--stage2_epochs",      "25",   "Max epochs for Stage 2 (full fine-tune)"],
        ["--patience",           "7",    "Early stopping patience (epochs)"],
        ["--img_size",           "288",  "Resize images to img_size × img_size"],
        ["--seed",               "42",   "Random seed for reproducibility"],
        ["--hash_threshold",     "5",    "Perceptual hash Hamming distance for near-duplicate detection"],
        ["--use_three_way_split","False", "If set, uses 70/15/15 split with a held-out test set"],
    ],
    col_widths=[4.0, 2.5, 10.0],
)
doc.add_paragraph()

# ── 7. scripts/yolo_seg_crop.py ───────────────────────────────────────────────
heading(doc, "7. scripts/yolo_seg_crop.py — YOLO Segmentation Crop", 1)
section_box(doc, "scripts/yolo_seg_crop.py", color="1D6038")
doc.add_paragraph(
    "Pre-processes images by using YOLOv8's segmentation model to detect the food item "
    "and crop tightly around it. This removes background clutter so the classifier focuses only on the dish."
)
doc.add_paragraph()

heading(doc, "How segmentation differs from detection", 2, "2E75B6")
make_table(doc,
    headers=["Method", "What it produces", "Crop quality"],
    rows=[
        ["Detection (old yolo_crop.py)", "Bounding box (rectangle)", "Loose — includes background around dish"],
        ["Segmentation (yolo_seg_crop.py)", "Polygon mask around food shape", "Tight — follows the actual food contour"],
    ],
    col_widths=[5.0, 5.5, 6.0],
)
doc.add_paragraph()

heading(doc, "Priority order per image", 2, "2E75B6")
bullet(doc, "Use bounding box derived from segmentation mask polygon  ✅ (93.9% of images)", "1st: ")
bullet(doc, "Use YOLO detection bounding box if no mask found  ⚠️", "2nd: ")
bullet(doc, "Center crop (central 80% of image) if YOLO finds nothing  ℹ️", "3rd: ")
doc.add_paragraph()

heading(doc, "get_mask_bbox(mask_xy, img_w, img_h)", 2, "2E75B6")
doc.add_paragraph("Converts a list of (x, y) polygon points from the YOLO segmentation mask into a tight axis-aligned bounding box. Clamps the result to image boundaries.")
doc.add_paragraph()

heading(doc, "crop_with_yolo_seg(model, img_path, padding=0.08)", 2, "2E75B6")
doc.add_paragraph("Runs YOLOv8-seg on one image, finds the best mask/box, adds 8% padding around it, and returns a PIL image crop plus metadata (method used, whether fallback was needed).")
doc.add_paragraph()

heading(doc, "Command Line Usage", 2, "2E75B6")
code_block(doc, "# Crop training data:\npython scripts/yolo_seg_crop.py --input results/categorized_dataset --output results/yolo_seg_dataset --model yolov8n-seg.pt\n\n# Crop test data:\npython scripts/yolo_seg_crop.py --input test_dataset_dishes --output results/yolo_seg_test --model yolov8n-seg.pt")
doc.add_paragraph()

# ── 8. scripts/split_dataset.py ───────────────────────────────────────────────
heading(doc, "8. scripts/split_dataset.py — Dataset Splitter", 1)
section_box(doc, "scripts/split_dataset.py")
doc.add_paragraph(
    "A standalone CLI tool to create and verify a 3-way Train/Val/Test split. "
    "Calls prepare_data_three_way() from utils.py and then runs two checks:"
)
bullet(doc, "check_leakage() — verifies zero images appear in more than one split", "Check 1: ")
bullet(doc, "print_split_summary() — prints image counts per class per split", "Check 2: ")
doc.add_paragraph()
code_block(doc, "python scripts/split_dataset.py --dataset results/categorized_dataset --model mobilenet_v3_large")
note(doc, "This produces split_info.json (used by train.py and evaluate.py) and 3 CSV files: splits/train.csv, splits/val.csv, splits/test.csv")
doc.add_paragraph()

# ── 9. scripts/evaluate.py ────────────────────────────────────────────────────
heading(doc, "9. scripts/evaluate.py — Internal Evaluator", 1)
section_box(doc, "scripts/evaluate.py")
doc.add_paragraph(
    "Evaluates the model on either the validation or internal test split stored in split_info.json. "
    "Loads images from the split, runs inference, and computes all metrics."
)
make_table(doc,
    headers=["Output file", "Contents"],
    rows=[
        ["classification_report.txt", "Accuracy, precision, recall, F1 per class"],
        ["metrics.json",              "All metrics as structured JSON for programmatic use"],
        ["confusion_matrix.png",      "Visual heatmap of true vs predicted classes"],
        ["predictions.csv",           "Row per image: path, true label, predicted label, confidence"],
    ],
    col_widths=[5.0, 11.5],
)
doc.add_paragraph()
code_block(doc, "python scripts/evaluate.py --model mobilenet_v3_large --split test --img_size 288")
doc.add_paragraph()

# ── 10. scripts/evaluate_external.py ─────────────────────────────────────────
heading(doc, "10. scripts/evaluate_external.py — External Test Evaluator", 1)
section_box(doc, "scripts/evaluate_external.py", color="7030A0")
doc.add_paragraph(
    "The key difference from evaluate.py: this script evaluates on a "
)
p = doc.paragraphs[-1]
r = p.add_run("completely separate folder")
r.bold = True
p.add_run(
    " — test_dataset_dishes — that was never used during training. "
    "It does NOT need split_info.json. It reads class labels directly from folder structure."
)
doc.add_paragraph()

heading(doc, "load_external_test_set(test_dir, class_mapping)", 2, "2E75B6")
doc.add_paragraph("Walks the test directory, infers class (veg/non_veg) from the top-level folder name, and returns file paths + labels. Supports both flat (class/image.jpg) and nested (class/dish/image.jpg) structures.")
doc.add_paragraph()

heading(doc, "Why this matters", 2, "2E75B6")
make_table(doc,
    headers=["Evaluator", "Test images source", "Reliability"],
    rows=[
        ["evaluate.py",          "15% of training data (split_info.json)", "⚠️ Could be biased if val set influenced training"],
        ["evaluate_external.py", "Completely separate folder (534 imgs)",   "✅ Fully unbiased — images never seen by model"],
    ],
    col_widths=[4.0, 7.5, 5.0],
)
doc.add_paragraph()
code_block(doc, "python scripts/evaluate_external.py --model mobilenet_v3_large_yolo_seg --test_dir results/yolo_seg_test --img_size 288")
doc.add_paragraph()

# ── 11. Data Flow ─────────────────────────────────────────────────────────────
heading(doc, "11. End-to-End Data Flow", 1)
doc.add_paragraph("This diagram shows how data moves through the full pipeline:")
doc.add_paragraph()
code_block(doc,
"""Raw images (test_dataset_dishes / categorized_dataset)
        │
        ▼
yolo_seg_crop.py  ──► YOLOv8n-seg detects food → tight mask crop
        │
        ▼
yolo_seg_dataset/       yolo_seg_test/
(1,203 cropped imgs)    (534 cropped imgs)
        │                       │
        ▼                       │
  train.py                      │
  ┌─────────────────────────┐   │
  │ prepare_data() →        │   │
  │   pHash grouping        │   │
  │   80/20 train/val split │   │
  │                         │   │
  │ Stage 1: freeze backbone│   │
  │   train classifier head │   │
  │                         │   │
  │ Stage 2: unfreeze all   │   │
  │   full fine-tune        │   │
  └─────────────────────────┘   │
        │                       │
        ▼                       ▼
  best checkpoint.pth ──► evaluate_external.py
                                │
                                ▼
                    92.70% accuracy on 534 external images
""")
doc.add_paragraph()

# ── 12. Key Concepts ──────────────────────────────────────────────────────────
heading(doc, "12. Key Concepts Explained", 1)

heading(doc, "Transfer Learning", 2, "2E75B6")
doc.add_paragraph(
    "Instead of training a neural network from scratch (which needs millions of images), "
    "we start with a model already trained on ImageNet. The early layers already understand "
    "edges, textures, and shapes. We only need to teach the final layer to distinguish veg vs non_veg. "
    "This is why we can get 92%+ accuracy with only ~1,200 images."
)
doc.add_paragraph()

heading(doc, "Perceptual Hashing (pHash)", 2, "2E75B6")
doc.add_paragraph(
    "A technique to find near-duplicate images. Two images that are slightly cropped/resized versions "
    "of each other will have similar hashes. We compare the Hamming distance between hashes — "
    "if distance ≤ 5, they are grouped together and kept in the same split. "
    "This prevents the model from 'memorising' training images that also appear in the test set."
)
doc.add_paragraph()

heading(doc, "Class Weights", 2, "2E75B6")
doc.add_paragraph(
    "If one class has more images than another, the model will naturally be biased toward it. "
    "Class weights counter this: minority classes get a higher loss multiplier, "
    "so the model is penalised more for getting them wrong."
)
code_block(doc, "weight = total_images / (num_classes × class_count)")
doc.add_paragraph()

heading(doc, "Label Smoothing", 2, "2E75B6")
doc.add_paragraph(
    "Instead of training towards 100% confidence (label = 1.0), we target 95% (label = 0.95). "
    "This prevents the model from becoming overconfident and generalises better to new images."
)
doc.add_paragraph()

heading(doc, "CosineAnnealingLR", 2, "2E75B6")
doc.add_paragraph(
    "A learning rate schedule that smoothly decreases the LR from its starting value to near-zero "
    "following a cosine curve. This allows large learning steps early in Stage 2 and tiny steps near the end, "
    "which helps the model settle into a good minimum without oscillating."
)
doc.add_paragraph()

# ── Footer ────────────────────────────────────────────────────────────────────
fp = doc.add_paragraph()
fp.alignment = WD_ALIGN_PARAGRAPH.CENTER
r = fp.add_run(f"Food Classifier Project  •  Code Documentation  •  {datetime.now().strftime('%Y-%m-%d')}")
r.italic = True; r.font.size = Pt(9)
r.font.color.rgb = RGBColor(0x88,0x88,0x88)

doc.save(OUTPUT_PATH)
print(f"✅  Documentation saved to: {OUTPUT_PATH}")
