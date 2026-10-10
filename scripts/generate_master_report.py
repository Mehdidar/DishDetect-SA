"""
generate_master_report.py
Generates the complete master project report: Food Classifier (Veg vs Non-Veg)
Covers every phase, dataset, metric, visualization, flaw, and lesson learned.
"""

import os
import json
from datetime import datetime
from docx import Document
from docx.shared import Pt, RGBColor, Cm, Inches, Emu
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.enum.table import WD_TABLE_ALIGNMENT
from docx.oxml.ns import qn
from docx.oxml import OxmlElement

# ─── Paths ───────────────────────────────────────────────────────────────────
ROOT  = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
DOCS  = os.path.join(ROOT, "docs")
os.makedirs(DOCS, exist_ok=True)
OUTPUT = os.path.join(DOCS, "Master_Project_Report.docx")

def p(path): return os.path.join(ROOT, path)   # shorthand

# ─── Helpers ─────────────────────────────────────────────────────────────────
def set_cell_bg(cell, hex_color):
    tc   = cell._tc
    tcPr = tc.get_or_add_tcPr()
    shd  = OxmlElement("w:shd")
    shd.set(qn("w:val"),   "clear")
    shd.set(qn("w:color"), "auto")
    shd.set(qn("w:fill"),  hex_color.upper())
    tcPr.append(shd)

def heading(doc, text, level=1, color="1F3864"):
    h = doc.add_heading(text, level=level)
    h.alignment = WD_ALIGN_PARAGRAPH.LEFT
    for run in h.runs:
        run.font.color.rgb = RGBColor.from_string(color)
    return h

def body(doc, text, italic=False, bold=False, size=11):
    par = doc.add_paragraph()
    r = par.add_run(text)
    r.italic = italic
    r.bold   = bold
    r.font.size = Pt(size)
    return par

def bullet(doc, label, value=None, color_label="1F3864"):
    par = doc.add_paragraph(style="List Bullet")
    r = par.add_run(f"{label}")
    r.bold = True
    r.font.color.rgb = RGBColor.from_string(color_label)
    if value:
        par.add_run(f" {value}")
    return par

def insert_image(doc, path, width_in=5.5, caption=None):
    if os.path.exists(path):
        doc.add_picture(path, width=Inches(width_in))
        if caption:
            cp = doc.add_paragraph(caption)
            cp.alignment = WD_ALIGN_PARAGRAPH.CENTER
            cp.runs[0].italic = True
            cp.runs[0].font.size = Pt(9)
    else:
        body(doc, f"[Image not found: {os.path.basename(path)}]", italic=True)

def metrics_table(doc, headers, rows):
    tbl = doc.add_table(rows=1, cols=len(headers))
    tbl.style = "Table Grid"
    tbl.alignment = WD_TABLE_ALIGNMENT.CENTER
    hdr = tbl.rows[0].cells
    for i, h in enumerate(headers):
        hdr[i].text = h
        set_cell_bg(hdr[i], "1F3864")
        for run in hdr[i].paragraphs[0].runs:
            run.font.color.rgb = RGBColor.from_string("FFFFFF")
            run.bold = True
    for row in rows:
        rc = tbl.add_row().cells
        for i, val in enumerate(row):
            rc[i].text = str(val)
    return tbl

# ─── Main ─────────────────────────────────────────────────────────────────────
def main():
    doc = Document()
    for sec in doc.sections:
        sec.top_margin    = Cm(2.0)
        sec.bottom_margin = Cm(2.0)
        sec.left_margin   = Cm(2.5)
        sec.right_margin  = Cm(2.5)

    # ── TITLE PAGE
    t = doc.add_paragraph()
    t.alignment = WD_ALIGN_PARAGRAPH.CENTER
    r = t.add_run("FOOD CLASSIFIER — VEG vs NON-VEG\nMASTER PROJECT REPORT")
    r.bold = True
    r.font.size = Pt(26)
    r.font.color.rgb = RGBColor.from_string("1F3864")

    doc.add_paragraph()
    sub = doc.add_paragraph()
    sub.alignment = WD_ALIGN_PARAGRAPH.CENTER
    sr = sub.add_run(
        "Complete record of all phases, datasets, results, visualizations, "
        "flaws & lessons learned\nGenerated: " + datetime.now().strftime("%d %B %Y")
    )
    sr.font.size = Pt(12)
    sr.italic = True

    doc.add_page_break()

    # ── 1. PROJECT OVERVIEW
    heading(doc, "1. Project Overview")
    body(doc,
        "This report documents the full lifecycle of the Food Classifier project — a deep-learning "
        "system trained to classify food images as Vegetarian (Veg) or Non-Vegetarian (Non-Veg). "
        "The project evolved through 10+ iterative phases spanning multiple architectures, datasets, "
        "preprocessing pipelines, and real-world restaurant evaluations."
    )
    doc.add_paragraph()
    bullet(doc, "Task:", "Binary Image Classification  —  Veg vs Non-Veg")
    bullet(doc, "Primary Architecture:", "MobileNetV3-Large (PyTorch / TorchVision)")
    bullet(doc, "Secondary Architecture:", "EfficientNet-B2 (architecture comparison, Phase 10)")
    bullet(doc, "Preprocessing:", "YOLOv8n-seg — background/plate removal via segmentation masks")
    bullet(doc, "Framework:", "PyTorch with AdamW optimizer & Cosine Annealing LR scheduler")
    bullet(doc, "Best Standard Test Accuracy:", "92.70%  (MobileNetV3 + YOLO Seg, 534 images)")
    bullet(doc, "Best Restaurant Accuracy:", "92.73%  (Phase 8 — 51/55 correct)")

    doc.add_page_break()

    # ── 2. DATASET EVOLUTION
    heading(doc, "2. Dataset Evolution & Pre-Processing")
    body(doc,
        "The dataset was never static — it grew and was cleaned across every phase. "
        "Rigorous data validation was performed before each training run."
    )
    heading(doc, "2.1  Dataset Versions", level=2)
    metrics_table(doc,
        ["Version", "Total Images", "Non-Veg", "Veg", "Key Change"],
        [
            ["V1 (Baseline)",   "~1 439", "758",   "681",   "Raw Arabic dishes, 25 dish classes"],
            ["V2 (Binary)",     "~1 439", "758",   "681",   "Re-labelled to Veg / Non-Veg only"],
            ["V3 (YOLO Seg)",   "~1 439", "758",   "681",   "YOLOv8 background removal applied"],
            ["V4 (German)",     "2 191",  "1 172", "1 019", "Cross-domain: German restaurant dishes"],
            ["V5 (Shawarma+)",  "1 319",  "615",   "704",   "Shawarma images boosted from 5 to 86"],
            ["V6 (Rebalanced)", "1 432",  "728",   "704",   "Non-veg rebalanced; Shish Tawook boosted"],
            ["35-Dish Variant", "1 306",  "—",     "—",     "35-class hierarchical dish classifier"],
        ]
    )
    doc.add_paragraph()
    heading(doc, "2.2  Data Cleaning Steps", level=2)
    bullet(doc, "Corrupted & unsupported files removed:", "1 corrupt + 8 .avif / .DS_Store files")
    bullet(doc, "Size filtering:", "Images below minimum CNN-usable resolution discarded")
    bullet(doc, "Duplicate leakage prevention:", "Perceptual hashing (threshold=5) groups near-duplicates into the same split")
    bullet(doc, "Train / Val / Test split:", "70% / 15% / 15%  (three-way stratified)")
    bullet(doc, "Image resolution:", "344x344 px (primary) — also tested at 288 and 224")

    doc.add_page_break()

    # ── 3. PIPELINE ARCHITECTURE
    heading(doc, "3. Pipeline Architecture")
    heading(doc, "3.1  Pre-processing: YOLO Segmentation", level=2)
    body(doc,
        "Before passing images to the classifier, YOLOv8n-seg (nano segmentation model) is used "
        "to automatically crop out the food object by removing the plate, table, and background. "
        "This dramatically reduces background noise and focuses the CNN on food pixels only."
    )
    insert_image(doc, p("results/phase9_flowchart.png"), 6.0,
                 "Figure 1 — Phase 9 pipeline flowchart (YOLO crop → MobileNetV3 classifier)")

    heading(doc, "3.2  Two-Stage Transfer Learning", level=2)
    body(doc, "All models are trained using a two-stage transfer learning strategy:")
    bullet(doc, "Stage 1 — Frozen Backbone (LR 1e-3):",
           "Only the classification head is trained. Max 15 epochs. ImageNet-pretrained features preserved.")
    bullet(doc, "Stage 2 — Full Fine-Tuning (LR 1e-5):",
           "Entire network is unfrozen. Cosine Annealing scheduler. Early stopping with patience=7.")
    bullet(doc, "Loss Function:", "CrossEntropyLoss with label smoothing (0.1) & class-frequency weights")
    bullet(doc, "Optimizer:", "AdamW with weight decay")
    bullet(doc, "Random Seed:", "42 (for reproducibility)")

    heading(doc, "3.3  Codebase Structure", level=2)
    metrics_table(doc,
        ["Directory", "Purpose", "Key Files"],
        [
            ["core/",    "Model factory & shared utilities",    "models/__init__.py, mobilenet.py, efficientnet.py, utils.py"],
            ["scripts/", "Active training & evaluation pipeline","train.py, evaluate.py, predict.py, yolo_seg_crop.py"],
            ["tools/",   "Dataset diagnostics & reporting",     "dataset_check.py, full_dataset_diagnostic.py, gather_metrics.py"],
            ["app/",     "Interactive UI front-end",            "ui.py, patch_ui.py"],
            ["docs/",    "Generated project reports",           "Master_Project_Report.docx + all prior reports"],
            ["updates/", "Historical one-off analysis scripts", "generate_*.py, german reports"],
        ]
    )

    doc.add_page_break()

    # ── 4. PHASE HISTORY
    heading(doc, "4. Phase-by-Phase Development History")

    # Phase 1
    heading(doc, "Phase 1 — Baseline 25-Class Arabic Dish Classifier", level=2, color="2E75B6")
    metrics_table(doc,
        ["Parameter", "Value"],
        [
            ["Architecture",     "MobileNetV3-Large"],
            ["Dataset",          "~1 439 images, 25 Arabic dish classes"],
            ["Image Resolution", "288x288 px"],
            ["LR Stage 1",       "1e-3 (AdamW)"],
            ["LR Stage 2",       "1e-5 (Cosine Annealing)"],
            ["Batch Size",       "32"],
            ["Accuracy",         "Good (qualitative baseline — exact % not retained)"],
        ]
    )
    body(doc,
        "The initial model attempted to predict the exact dish name across 25 Arabic categories. "
        "Visual ambiguity between similar-looking rice and meat dishes caused inter-class confusion. "
        "This phase confirmed the need to simplify the task to binary classification."
    )

    # Phase 2
    doc.add_paragraph()
    heading(doc, "Phase 2 — Binary Veg/Non-Veg (EfficientNet-B2, No YOLO)", level=2, color="2E75B6")
    metrics_table(doc,
        ["Parameter", "Value"],
        [
            ["Architecture",    "EfficientNet-B2"],
            ["Dataset",         "~1 439 images  |  veg + non_veg labels"],
            ["Test Accuracy",   "89.30%"],
            ["Precision",       "0.90"],
            ["Recall",          "0.89"],
            ["F1-Score",        "0.89"],
            ["Loss Function",   "CrossEntropyLoss + Label Smoothing 0.1"],
            ["Early Stop Pat.", "7 epochs"],
        ]
    )
    body(doc,
        "Restructuring from 25 classes to 2 classes caused a dramatic accuracy jump, proving that "
        "learning high-level dietary features is far easier than distinguishing exact dish names."
    )
    insert_image(doc, p("results_efficientnet_b2/training_curves.png"), 6.0,
                 "Figure 2 — EfficientNet-B2 training curves (Phase 2)")
    insert_image(doc, p("results_efficientnet_b2/confusion_matrix.png"), 5.5,
                 "Figure 3 — EfficientNet-B2 confusion matrix (Phase 2)")

    # Phase 2B
    doc.add_paragraph()
    heading(doc, "Phase 2B — MobileNetV3 + YOLO Segmentation (First YOLO Run)", level=2, color="2E75B6")
    metrics_table(doc,
        ["Parameter", "Value"],
        [
            ["Architecture",  "MobileNetV3-Large + YOLOv8n-seg"],
            ["Standard Test", "92.70%  (534 images)"],
            ["Precision",     "0.93"],
            ["Recall",        "0.93"],
            ["F1-Score",      "0.93"],
        ]
    )
    body(doc,
        "Introducing YOLO segmentation was the single biggest accuracy improvement — "
        "from 89.30% to 92.70% on the standard test set (+3.4pp). "
        "Removing background and plate noise forced the model to focus purely on food textures."
    )
    insert_image(doc, p("results_mobilenet_v3_large_yolo_seg/training_curves.png"), 6.0,
                 "Figure 4 — MobileNetV3 + YOLO Seg training curves (Phase 2B)")
    insert_image(doc, p("results_mobilenet_v3_large_yolo_seg/confusion_matrix.png"), 5.5,
                 "Figure 5 — MobileNetV3 + YOLO Seg confusion matrix (Phase 2B)")

    # Phase 3
    doc.add_page_break()
    heading(doc, "Phase 3 — Cross-Domain Transfer: German Restaurant Dataset", level=2, color="2E75B6")
    metrics_table(doc,
        ["Parameter", "Value"],
        [
            ["Architecture",    "MobileNetV3-Large + YOLOv8n-seg"],
            ["Dataset",         "2 191 German restaurant images (30 dish categories)"],
            ["Overall Accuracy","95.53%"],
            ["Precision",       "0.955"],
            ["Recall",          "0.955"],
            ["F1-Score",        "0.955"],
            ["Avg Confidence",  "86.63%"],
            ["Uncertain (<65%)","193 images"],
        ]
    )
    body(doc,
        "Fine-tuning on an entirely new German restaurant dataset tested cross-domain generalization. "
        "The model reached 95.53%, demonstrating strong transfer learning capability."
    )
    heading(doc, "German Test — Per-Dish Accuracy (highlights)", level=3)
    metrics_table(doc,
        ["Dish", "Category", "Accuracy", "Correct/Total"],
        [
            ["Miesmuscheln",     "Non-Veg", "100.0%", "72/72"],
            ["Rouladen",         "Non-Veg", "100.0%", "75/75"],
            ["Sauerbraten",      "Non-Veg", "100.0%", "75/75"],
            ["Brokkolisuppe",    "Veg",     "100.0%", "75/75"],
            ["Currywurst",       "Non-Veg", "98.6%",  "71/72"],
            ["Schnitzel",        "Non-Veg", "90.7%",  "68/75"],
            ["Garnelen gebraten","Non-Veg", "88.9%",  "64/72"],
            ["Bratkartoffeln",   "Veg",     "84.0%",  "63/75"],
            ["Kartoffelpuffer",  "Veg",     "83.3%",  "60/72"],
        ]
    )
    insert_image(doc, p("results_mobilenet_v3_large_yolo_seg/prediction_images/summary_overview.png"), 6.0,
                 "Figure 6 — German dataset summary overview")
    insert_image(doc, p("results_mobilenet_v3_large_yolo_seg/prediction_images/uncertain_predictions.png"), 6.0,
                 "Figure 7 — German dataset uncertain predictions")

    # Phase 4
    doc.add_page_break()
    heading(doc, "Phase 4 — Real-World Arabic Restaurant Testing (First Run)", level=2, color="2E75B6")
    metrics_table(doc,
        ["Parameter", "Value"],
        [
            ["Model",      "German-tuned MobileNetV3 + YOLO"],
            ["Test Set",   "53 unseen Arabic restaurant images"],
            ["Accuracy",   "~77.36%  (41/53 correct)"],
        ]
    )
    body(doc,
        "Deploying the German-tuned model on unseen Arabic restaurant photos exposed a domain gap. "
        "Cheese pastries resembled meat pies; large rice portions hid the meat in Mandi."
    )
    insert_image(doc, p("results/arabic_confusion_matrix.png"), 5.0,
                 "Figure 8 — Arabic restaurant confusion matrix (Phase 4)")

    # Phase 5
    doc.add_paragraph()
    heading(doc, "Phase 5 — Hierarchical 26-Class Dish Classifier", level=2, color="2E75B6")
    metrics_table(doc,
        ["Parameter", "Value"],
        [
            ["Architecture",   "MobileNetV3-Large"],
            ["Dataset",        "YOLO-segmented Arabic dishes, 26 classes, ~1 439 images"],
            ["Validation Acc.","67.61%  (247 images)"],
            ["Notable Wins",   "Hummus 90%, Shawarma 75%, Tamees 85%, Shish Tawook 100%"],
            ["Notable Fails",  "Meat-Madfoon 14%, Pistachio Tamriya 40%, Grilled Kofta 50%"],
        ]
    )
    body(doc,
        "Dish-level classification (26 classes) achieved ~67.6% val accuracy — strong for a 26-way task. "
        "High per-class variance indicated the model was limited by training data volume for rare dishes."
    )
    insert_image(doc, p("results_mobilenet_v3_large/training_curves.png"), 6.0,
                 "Figure 9 — MobileNetV3 26-class training curves (Phase 5)")
    insert_image(doc, p("results_mobilenet_v3_large/confusion_matrix.png"), 6.0,
                 "Figure 10 — MobileNetV3 26-class confusion matrix (Phase 5, val split)")

    # Phase 6
    doc.add_page_break()
    heading(doc, "Phase 6 — 35-Class Expanded Hierarchical Classifier", level=2, color="2E75B6")
    metrics_table(doc,
        ["Parameter", "Value"],
        [
            ["Architecture",  "MobileNetV3-Large + YOLO Seg"],
            ["Dataset",       "1 306 images, 35 dish classes"],
            ["Test Accuracy", "67.01%  (194 images)"],
            ["Precision",     "0.7035"],
            ["Recall",        "0.6701"],
            ["F1-Score",      "0.6681"],
        ]
    )
    body(doc,
        "Expanding from 26 to 35 classes naturally caused an accuracy drop. "
        "More classes with similar appearance and fewer images per class make discrimination harder. "
        "However, binary Veg/Non-Veg accuracy on the same dataset remained at 98%+ via category aggregation."
    )
    insert_image(doc, p("results_mobilenet_v3_large_35dishes/training_curves.png"), 6.0,
                 "Figure 11 — 35-class training curves (Phase 6)")
    insert_image(doc, p("results_mobilenet_v3_large_35dishes/confusion_matrix.png"), 6.0,
                 "Figure 12 — 35-class confusion matrix (Phase 6)")

    # Phase 7
    doc.add_paragraph()
    heading(doc, "Phase 7 — Refined Binary (V5 Dataset, 1 231 Images)", level=2, color="2E75B6")
    metrics_table(doc,
        ["Parameter", "Value"],
        [
            ["Architecture",  "MobileNetV3-Large + YOLO Seg"],
            ["Dataset",       "1 231 images, back to Veg/Non-Veg binary"],
            ["Accuracy",      "89.43%"],
            ["Early Stop",    "Triggered at Epoch 8 (Stage 2)"],
        ]
    )
    body(doc,
        "Cleaned back to 26 core dishes and retrained as binary classification. "
        "89.43% confirmed that the model easily learns dietary categories when quality data is available."
    )

    # Phase 8
    doc.add_page_break()
    heading(doc, "Phase 8 — Shawarma-Boosted Binary Classifier (Peak Restaurant Score)", level=2, color="2E75B6")
    metrics_table(doc,
        ["Parameter", "Value"],
        [
            ["Architecture",       "MobileNetV3-Large + YOLO Seg (V5 Dataset)"],
            ["Dataset",            "1 319 images  |  non_veg: 615  |  veg: 704"],
            ["Shawarma boost",     "5 to 86 images (17x)"],
            ["Standard Test Acc.", "91.20%  (534 images)"],
            ["Restaurant Acc.",    "92.73%  (55 images — 51/55 correct)  BEST EVER"],
        ]
    )
    body(doc,
        "Massively boosting Shawarma training data caused 8 out of 10 restaurant dishes to "
        "score 100%. This was the project's best ever real-world restaurant accuracy."
    )
    heading(doc, "Phase 8 — Per-Dish Restaurant Results", level=3)
    metrics_table(doc,
        ["Dish", "Category", "Restaurant Accuracy"],
        [
            ["Cheese Fatayer",    "Veg",     "100%"],
            ["Hummus",            "Veg",     "100%"],
            ["Mandi",             "Non-Veg", "100%"],
            ["Meat Madfoon",      "Non-Veg", "100%"],
            ["Mixed Grill",       "Non-Veg", "100%"],
            ["Okra Tagine",       "Veg",     "100%"],
            ["Tamees Bread",      "Veg",     "100%"],
            ["Za'atara Manaqish", "Veg",     "100%"],
            ["Shish Tawook",      "Non-Veg", "85.7%"],
            ["Shawarma",          "Non-Veg", "57.1%  (WARNING)"],
        ]
    )

    # Phase 9
    doc.add_paragraph()
    heading(doc, "Phase 9 — V6 Rebalance (1 432 Images)", level=2, color="2E75B6")
    metrics_table(doc,
        ["Parameter", "Value"],
        [
            ["Architecture",        "MobileNetV3-Large + YOLO Seg (V6 Dataset)"],
            ["Dataset",             "1 432 images  |  non_veg: 728  |  veg: 704"],
            ["Shish Tawook boost",  "to 80 images"],
            ["Shawarma boost",      "to 124 images"],
            ["Standard Test Acc.",  "92.0%  (534 images)"],
            ["Restaurant Acc.",     "90.9%  (55 images — 50/55 correct)"],
        ]
    )
    heading(doc, "Phase 9 — Per-Dish Restaurant Results", level=3)
    metrics_table(doc,
        ["Dish", "Category", "Restaurant Accuracy"],
        [
            ["Cheese Fatayer",    "Veg",     "100%"],
            ["Hummus",            "Veg",     "100%"],
            ["Meat Madfoon",      "Non-Veg", "100%"],
            ["Mixed Grill",       "Non-Veg", "100%"],
            ["Tamees Bread",      "Veg",     "100%"],
            ["Za'atara Manaqish", "Veg",     "100%"],
            ["Shish Tawook",      "Non-Veg", "100% (recovered!)"],
            ["Mandi",             "Non-Veg", "85.7%"],
            ["Okra Tagine",       "Veg",     "75.0%"],
            ["Shawarma",          "Non-Veg", "57.1%  (WARNING)"],
        ]
    )

    # Phase 10
    doc.add_page_break()
    heading(doc, "Phase 10 — Architecture Experiment: MobileNetV3 vs EfficientNet-B2", level=2, color="2E75B6")
    body(doc,
        "Both architectures were trained on V6 dataset at 344x344 with identical hyperparameters. "
        "EfficientNet Stage 2 fine-tuning was cancelled (>5 hours/epoch on CPU). "
        "Only Stage 1 (frozen backbone) EfficientNet results shown."
    )
    metrics_table(doc,
        ["Metric", "MobileNetV3-Large (344x344)", "EfficientNet-B2 (Stage 1 only)"],
        [
            ["Standard Test Accuracy", "92.0%",  "84.0%"],
            ["Restaurant Accuracy",    "90.9%",  "90.9%"],
            ["False Positives",        "1",      "2"],
            ["False Negatives",        "4",      "3"],
        ]
    )
    heading(doc, "Phase 10 — Per-Dish Comparison", level=3)
    metrics_table(doc,
        ["Dish", "MobileNetV3", "EfficientNet-B2", "Analysis"],
        [
            ["Cheese Fatayer",   "100%",        "100%",        "Perfect on both"],
            ["Hummus",           "100%",        "100%",        "Perfect on both"],
            ["Mandi",            "100%",        "100%",        "Perfect on both"],
            ["Meat Madfoon",     "100%",        "100%",        "Perfect on both"],
            ["Mixed Grill",      "100%",        "100%",        "Perfect on both"],
            ["Tamees Bread",     "100%",        "100%",        "Perfect on both"],
            ["Shawarma",         "71.4% (5/7)", "85.7% (6/7)", "EfficientNet identified hidden meat texture"],
            ["Shish Tawook",     "85.7% (6/7)", "85.7% (6/7)", "Identical — fails on heavy garlic sauce"],
            ["Za'atar Manaqish", "100% (5/5)",  "80.0% (4/5)", "EfficientNet confused by herb textures"],
            ["Okra Tagine",      "75.0% (3/4)", "50.0% (2/4)", "EfficientNet confused by generic stew texture"],
        ]
    )

    doc.add_page_break()

    # ── 5. KEY VISUALIZATIONS
    heading(doc, "5. Key Visualizations")

    heading(doc, "5.1  Restaurant Test — Overall Accuracy Pie", level=2)
    insert_image(doc, p("results/restaurant_analysis/overall_accuracy_pie.png"), 4.5,
                 "Figure 13 — Overall accuracy pie chart for the 55-image restaurant test set")

    heading(doc, "5.2  Restaurant Test — Per-Dish Accuracy Bar Chart", level=2)
    insert_image(doc, p("results/restaurant_analysis/per_dish_accuracy.png"), 6.0,
                 "Figure 14 — Per-dish accuracy bar chart across all 10 restaurant dishes")

    heading(doc, "5.3  Restaurant Test — Confidence Score Distribution", level=2)
    insert_image(doc, p("results/restaurant_analysis/confidence_distribution.png"), 6.0,
                 "Figure 15 — Confidence score distribution for restaurant predictions")

    heading(doc, "5.4  MobileNetV3 + YOLO — Training Curves", level=2)
    insert_image(doc, p("results_mobilenet_v3_large_yolo_seg/training_curves.png"), 6.0,
                 "Figure 16 — Training & validation loss/accuracy curves for the primary model")

    heading(doc, "5.5  External Standard Test — Confusion Matrix", level=2)
    insert_image(doc, p("results/test_external_mobilenet_v3_large_yolo_seg/confusion_matrix.png"), 5.5,
                 "Figure 17 — External standard test confusion matrix (MobileNetV3+YOLO, 534 images)")

    heading(doc, "5.6  German Model — Wrong Predictions Sample", level=2)
    insert_image(doc, p("results_mobilenet_v3_large_yolo_seg/prediction_images/wrong_predictions.png"), 6.0,
                 "Figure 18 — Sample of wrong predictions from the German dataset evaluation")

    doc.add_page_break()

    # ── 6. FULL PERFORMANCE SUMMARY
    heading(doc, "6. Full Performance Summary Across All Phases")
    metrics_table(doc,
        ["Phase", "Model", "Dataset", "Std Test Acc.", "Restaurant Acc."],
        [
            ["Phase 1",  "MobileNetV3",           "25-class Arabic",        "N/A",   "N/A"],
            ["Phase 2",  "EfficientNet-B2",        "V1 Binary (No YOLO)",    "89.3%", "N/A"],
            ["Phase 2B", "MobileNetV3 + YOLO",     "V1 Binary + YOLO",       "92.7%", "N/A"],
            ["Phase 3",  "MobileNetV3 + YOLO",     "German (2 191 imgs)",    "95.5%", "N/A"],
            ["Phase 4",  "German-Tuned MobileNet", "Arabic Restaurant",       "N/A",   "~77.4%"],
            ["Phase 5",  "MobileNetV3 + YOLO",     "26-Class Dish",           "~68%",  "N/A"],
            ["Phase 6",  "MobileNetV3 + YOLO",     "35-Class Dish",           "67.0%", "N/A"],
            ["Phase 7",  "MobileNetV3 + YOLO",     "V4 Binary (1 231 imgs)", "89.4%", "N/A"],
            ["Phase 8",  "MobileNetV3 + YOLO",     "V5 (Shawarma x17)",      "91.2%", "92.7% BEST"],
            ["Phase 9",  "MobileNetV3 + YOLO",     "V6 (1 432 imgs)",        "92.0%", "90.9%"],
            ["Phase 10", "EfficientNet-B2 (S1)",   "V6 (1 432 imgs)",        "84.0%", "90.9%"],
            ["Phase 10", "MobileNetV3 + YOLO",     "V6 (1 432 imgs)",        "92.0%", "90.9%"],
        ]
    )

    doc.add_page_break()

    # ── 7. KNOWN FLAWS
    heading(doc, "7. Known Flaws & Failure Cases")

    heading(doc, "7.1  Shawarma — The Persistent Failure", level=2, color="C00000")
    body(doc,
        "Shawarma is the most persistent failure. The pita-wrapped presentation looks like "
        "flatbread-based veg dishes. Even boosting training data from 5 to 124 images only pushed "
        "restaurant accuracy from ~43% to 57-71%."
    )
    bullet(doc, "Root cause:",
           "Wrapped Shawarma shows mostly bread on the exterior — the hidden meat is invisible "
           "unless the image shows the inside or cross-section.")
    bullet(doc, "Attempted fix:", "17x data boost (Phase 8) — helped but did not solve")
    bullet(doc, "Remaining gap:", "New training images were not representative of the restaurant pita-wrap style")

    heading(doc, "7.2  Mandi — The 'Sea of Rice' Problem", level=2, color="C00000")
    body(doc,
        "Mandi consistently fools the model because the dish is visually dominated by a large mound "
        "of yellow/brown rice (70-90% of pixels). The meat is often small, buried, or camouflaged."
    )
    bullet(doc, "YOLO crop side-effect:",
           "YOLO sometimes identifies the rice mound as the primary food object and crops out the "
           "meat at the edges — the classifier then only sees rice and predicts Veg.")
    bullet(doc, "Meat camouflage:",
           "Slow-roasted meat in Mandi has a golden-brown colour very similar to roasted vegetables, "
           "fried potatoes, and nuts.")
    bullet(doc, "Phase 9 result:", "Mandi accuracy dropped from 100% (Phase 8) to 85.7% after rebalancing")

    heading(doc, "7.3  Okra Tagine — Ambiguous Stew Texture", level=2, color="C00000")
    body(doc,
        "Okra Tagine appears as a generic stew. Depending on preparation it can be fully veg or "
        "contain meat, creating inherent label noise. EfficientNet scored only 50% here due to "
        "over-reliance on stew texture features."
    )

    heading(doc, "7.4  The 90.9% Restaurant Accuracy Ceiling", level=2, color="C00000")
    body(doc,
        "Across Phases 9 and 10, every variation — different resolution, architecture, preprocessing — "
        "hits the exact same 90.9% ceiling on the 55-image restaurant test. "
        "This proves the bottleneck is the training data, not the model."
    )
    bullet(doc, "Modified variables tested:", "Image resolution (288 vs 344), YOLO vs raw, MobileNetV3 vs EfficientNet-B2")
    bullet(doc, "Conclusion:", "The 4/5 wrong predictions always fall on Shawarma, Okra Tagine, Za'atar Manaqish")
    bullet(doc, "Fix required:", "Targeted data collection — restaurant-style, pita-wrapped Shawarma photos")

    heading(doc, "7.5  EfficientNet Stage 2 Cancellation", level=2, color="C00000")
    body(doc,
        "EfficientNet-B2 has ~9M parameters vs MobileNetV3 ~5M. Stage 2 fine-tuning on CPU "
        "was estimated at >5 hours per epoch making it impractical. Phase 10 EfficientNet results "
        "are therefore only Stage 1 (frozen backbone) — not a fair comparison."
    )

    heading(doc, "7.6  Data Leakage Risk", level=2, color="C00000")
    body(doc,
        "Near-duplicate images (augmented Shawarma and Shrimp Sayadiyah pairs) were identified via "
        "perceptual hashing and forcefully bucketed into the same split. However, phases before "
        "this check was added may have had minor leakage."
    )

    doc.add_page_break()

    # ── 8. LESSONS LEARNED
    heading(doc, "8. Lessons Learned & Key Insights")
    bullet(doc, "Data > Architecture:",
           "The 90.9% restaurant ceiling held across MobileNetV3 and EfficientNet-B2. "
           "Better targeted training data would outperform any architecture switch.")
    bullet(doc, "YOLO preprocessing was transformative:",
           "+3.4 percentage points gained by removing background noise (89.3% to 92.7%).")
    bullet(doc, "Binary is much easier than multi-class:",
           "Veg/Non-Veg consistently outperforms 26/35-class dish-level classification by 20-25pp.")
    bullet(doc, "Cross-domain transfer works:",
           "German dataset fine-tuning reached 95.5% — proving the YOLO+MobileNetV3 "
           "backbone generalizes across completely different cuisines.")
    bullet(doc, "Targeted augmentation matters:",
           "Shawarma 17x boost improved standard test accuracy but restaurant accuracy did not "
           "fully recover — augmentation must match the deployment domain.")
    bullet(doc, "Class imbalance must be addressed:",
           "Using inverse-frequency class weights in the loss was critical for preventing "
           "the model from defaulting to the majority class.")
    bullet(doc, "YOLO can hurt as well as help:",
           "For Mandi, YOLO cropping can remove the meat that sits on the plate edges, "
           "causing mis-classification as Veg.")

    doc.add_page_break()

    # ── 9. NEXT STEPS
    heading(doc, "9. Recommended Next Steps")
    heading(doc, "Immediate Data Wins", level=2)
    bullet(doc, "Collect 50-100 restaurant-style pita-wrapped Shawarma images",
           "(cross-section view revealing the meat)")
    bullet(doc, "Collect diverse Mandi images", "where chicken/lamb is prominently visible above the rice")
    bullet(doc, "Review Okra Tagine labelling", "— ensure consistent veg/non-veg ground truth")

    heading(doc, "Pipeline Improvements", level=2)
    bullet(doc, "Experiment with YOLO crop disabled for Mandi", "to avoid cropping out the meat")
    bullet(doc, "Train EfficientNet-B2 Stage 2", "on a GPU for a fair architecture comparison")
    bullet(doc, "Implement test-time augmentation (TTA)", "to improve robustness on ambiguous images")
    bullet(doc, "Add confidence thresholding in the UI", "— flag predictions below 65% as Uncertain")

    heading(doc, "Scaling", level=2)
    bullet(doc, "Expand to additional Arabic cuisines", "— Jareesh, Kabsa, Saleeg")
    bullet(doc, "Export model to ONNX / TFLite", "for mobile deployment")
    bullet(doc, "Build a REST API wrapper", "around the prediction pipeline for restaurant POS integration")

    doc.add_page_break()

    # ── 10. CODEBASE REFERENCE
    heading(doc, "10. Codebase Reference — Key Scripts")
    metrics_table(doc,
        ["Script", "Location", "Purpose"],
        [
            ["train.py",               "scripts/", "Two-stage training pipeline with early stopping"],
            ["evaluate.py",            "scripts/", "Evaluates model on internal test split"],
            ["evaluate_external.py",   "scripts/", "Evaluates on a fully external test directory"],
            ["predict.py",             "scripts/", "Single-image prediction with YOLO crop"],
            ["predict_menu.py",        "scripts/", "Batch prediction for restaurant menu images"],
            ["yolo_seg_crop.py",       "scripts/", "YOLO segmentation preprocessing"],
            ["restaurant_analysis.py", "scripts/", "Restaurant test set analysis + chart generation"],
            ["split_dataset.py",       "scripts/", "Stratified train/val/test split with leakage prevention"],
            ["dataset_check.py",       "tools/",   "Validates dataset integrity"],
            ["generate_master_report.py","scripts/","THIS FILE — generates the master project report"],
            ["ui.py",                  "app/",     "Tkinter GUI for interactive food classification"],
        ]
    )

    doc.add_paragraph()
    body(doc, "Generated automatically — " + datetime.now().strftime("%d %B %Y, %H:%M"), italic=True)

    # ── SAVE
    doc.save(OUTPUT)
    print("\n  Master report saved to: " + OUTPUT)


if __name__ == "__main__":
    main()
