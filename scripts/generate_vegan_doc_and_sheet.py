"""
generate_vegan_doc_and_sheet.py
-------------------------------
Generates high-fidelity documentation artifacts for the Meat-Free Vegan vs. Vegetarian module:
  1. docs/Vegan_vs_Vegetarian_Intelligence_Report.docx (Full illustrated Word report with YOLOv8-Seg ablation)
  2. docs/Vegan_vs_Vegetarian_Metrics_Sheet.xlsx (Comprehensive 5-tab Excel metrics tracker)
  3. docs/charts/ (High-res dark-mode comparison charts)

Also mirrors both files to Vegan_vs_Vegetarian_Classifier/docs/.
"""

import os
import sys
import json
import csv
from datetime import datetime
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import docx
from docx.shared import Inches, Pt, RGBColor
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.enum.table import WD_TABLE_ALIGNMENT, WD_ALIGN_VERTICAL
from docx.oxml import OxmlElement
from docx.oxml.ns import qn

import openpyxl
from openpyxl.styles import Font, PatternFill, Alignment, Border, Side
from openpyxl.utils import get_column_letter

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
DOCS_DIR = os.path.join(ROOT, "docs")
CHARTS = os.path.join(DOCS_DIR, "charts")
PACKAGE_DIR = os.path.join(ROOT, "Vegan_vs_Vegetarian_Classifier")

DOCX_OUT = os.path.join(DOCS_DIR, "Vegan_vs_Vegetarian_Intelligence_Report.docx")
XLSX_OUT = os.path.join(DOCS_DIR, "Vegan_vs_Vegetarian_Metrics_Sheet.xlsx")

# ─────────────────────────────────────────────────────────────────────────────
# 1. CHART GENERATION
# ─────────────────────────────────────────────────────────────────────────────
def generate_charts():
    os.makedirs(CHARTS, exist_ok=True)
    print("Generating High-Resolution Dark-Mode Performance Visualizations...")

    # Color Palette
    DARK   = "#111827"
    PANEL  = "#1F2937"
    GREEN  = "#10B981"
    BLUE   = "#3B82F6"
    PURPLE = "#8B5CF6"
    AMBER  = "#F59E0B"
    RED    = "#EF4444"
    WHITE  = "#F9FAFB"
    SILVER = "#9CA3AF"

    # ── Chart A: Baseline vs YOLO-Seg Test Confusion Matrices
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.8))
    fig.patch.set_facecolor(DARK)

    # Matrix 1: Baseline Test Split (107 images)
    # TP=49, TN=42, FP=3, FN=13 (Vegan=0, Vegetarian=1)
    cm_base = np.array([[49, 13], [3, 42]])
    # Matrix 2: YOLO-Seg Test Split (104 images)
    # TP=51, TN=38, FP=5, FN=10
    cm_seg  = np.array([[51, 10], [5, 38]])

    matrices = [
        (ax1, cm_base, "Uncropped Baseline Test (85.05%)\n107 images", BLUE),
        (ax2, cm_seg,  "YOLOv8-Seg Cropped Test (85.58%)\n104 images", GREEN)
    ]

    for ax, cm, title, tint in matrices:
        ax.set_facecolor(PANEL)
        cax = ax.matshow(cm, cmap="Blues" if tint == BLUE else "Greens", alpha=0.85)
        ax.set_xticks([0, 1])
        ax.set_yticks([0, 1])
        ax.set_xticklabels(["Pred Vegan", "Pred Veg"], color=SILVER, fontsize=10)
        ax.set_yticklabels(["True Vegan", "True Veg"], color=SILVER, fontsize=10)
        ax.tick_params(colors=SILVER, bottom=False, top=True)
        ax.set_title(title, color=WHITE, fontsize=11.5, fontweight="bold", pad=15)

        for i in range(2):
            for j in range(2):
                val = cm[i, j]
                ax.text(j, i, str(val), ha="center", va="center",
                        color=WHITE if val > 20 else SILVER, fontweight="bold", fontsize=13)
        ax.spines["bottom"].set_color(SILVER)
        ax.spines["left"].set_color(SILVER)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    fig.suptitle("MobileNetV3-Large Test Confusion Matrices: Baseline vs. YOLOv8-Seg",
                 color=WHITE, fontsize=13.5, fontweight="bold", y=1.03)
    fig.tight_layout()
    cm_path = os.path.join(CHARTS, "vegan_veg_confusion_matrix.png")
    fig.savefig(cm_path, dpi=160, bbox_inches="tight", facecolor=DARK)
    plt.close(fig)

    # ── Chart B: Grouped Bar Chart: Restaurant Accuracy Before vs After YOLO-Seg
    fig, ax = plt.subplots(figsize=(11, 5.2))
    fig.patch.set_facecolor(DARK)
    ax.set_facecolor(DARK)
    ax.spines["bottom"].set_color(SILVER)
    ax.spines["left"].set_color(SILVER)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(colors=SILVER, labelsize=9.5)
    ax.xaxis.label.set_color(SILVER)
    ax.yaxis.label.set_color(SILVER)

    categories = [
        "Hummus\n(Vegan)",
        "Okra Tagine\n(Vegan)",
        "Tamees Bread\n(Vegan)",
        "Cheese Fatayer\n(Vegetarian)",
        "Za’atara Manaqish\n(Vegan)",
        "OVERALL RESTAURANT\n(24 Images)"
    ]
    base_scores = [100.0, 100.0, 100.0, 80.0, 40.0, 83.33]
    seg_scores  = [100.0, 100.0, 100.0, 80.0, 100.0, 95.83]

    x = np.arange(len(categories))
    width = 0.35

    bars1 = ax.bar(x - width/2, base_scores, width, label="Uncropped Baseline (83.33%)", color=BLUE, edgecolor=WHITE, lw=0.8)
    bars2 = ax.bar(x + width/2, seg_scores, width, label="YOLOv8-Seg Cropped (95.83%)", color=GREEN, edgecolor=WHITE, lw=0.8)

    for b, s in zip(bars1, base_scores):
        ax.text(b.get_x() + b.get_width()/2, s + 1.5, f"{s:.1f}%",
                ha="center", va="bottom", color=SILVER, fontweight="bold", fontsize=8.5)

    for b, s in zip(bars2, seg_scores):
        ax.text(b.get_x() + b.get_width()/2, s + 1.5, f"{s:.1f}%",
                ha="center", va="bottom", color=WHITE, fontweight="bold", fontsize=9)

    ax.set_xticks(x)
    ax.set_xticklabels(categories, color=WHITE, fontsize=9.5)
    ax.set_ylim(0, 118)
    ax.set_ylabel("Accuracy (%)", color=SILVER, fontsize=10)
    ax.set_title("Real-World Restaurant Test: Uncropped Baseline vs. YOLOv8-Seg Food Cropping (+12.50% Jump)",
                 color=WHITE, fontsize=12.5, fontweight="bold", pad=14)
    ax.legend(facecolor=PANEL, edgecolor=SILVER, labelcolor=WHITE, loc="upper right")
    fig.tight_layout()
    rest_path = os.path.join(CHARTS, "vegan_veg_restaurant_accuracy.png")
    fig.savefig(rest_path, dpi=160, bbox_inches="tight", facecolor=DARK)
    plt.close(fig)

    # ── Chart C: Training Curves
    src_curves = os.path.join(ROOT, "results_mobilenet_v3_large_vegan_veg_yolo_seg", "training_curves.png")
    dest_curves = os.path.join(CHARTS, "vegan_veg_training_curves.png")
    if os.path.exists(src_curves):
        import shutil
        shutil.copy2(src_curves, dest_curves)
    elif os.path.exists(os.path.join(PACKAGE_DIR, "results", "training_curves.png")):
        import shutil
        shutil.copy2(os.path.join(PACKAGE_DIR, "results", "training_curves.png"), dest_curves)

    print("  Visualizations generated successfully.")
    return cm_path, rest_path, dest_curves

# ─────────────────────────────────────────────────────────────────────────────
# 2. WORD DOCUMENT REPORT GENERATION
# ─────────────────────────────────────────────────────────────────────────────
def set_cell_bg(cell, hex_color):
    tc = cell._tc; tcPr = tc.get_or_add_tcPr()
    shd = OxmlElement("w:shd")
    shd.set(qn("w:val"), "clear"); shd.set(qn("w:color"), "auto")
    shd.set(qn("w:fill"), hex_color.upper()); tcPr.append(shd)

def whead(doc, text, level=1, color="1F3864"):
    h = doc.add_heading(text, level=level)
    h.alignment = WD_ALIGN_PARAGRAPH.LEFT
    for r in h.runs:
        r.font.color.rgb = RGBColor.from_string(color)
        r.font.name = "Segoe UI"
    return h

def wbody(doc, text):
    p = doc.add_paragraph()
    p.paragraph_format.line_spacing = 1.15
    p.paragraph_format.space_after = Pt(6)
    r = p.add_run(text)
    r.font.name = "Segoe UI"
    r.font.size = Pt(10)
    r.font.color.rgb = RGBColor(0x33, 0x33, 0x33)
    return p

def wbullet(doc, bold_prefix, text):
    p = doc.add_paragraph(style="List Bullet")
    p.paragraph_format.space_after = Pt(4)
    r1 = p.add_run(bold_prefix + " ")
    r1.bold = True
    r1.font.name = "Segoe UI"
    r1.font.size = Pt(10)
    r1.font.color.rgb = RGBColor(0x1F, 0x38, 0x64)
    r2 = p.add_run(text)
    r2.font.name = "Segoe UI"
    r2.font.size = Pt(10)
    r2.font.color.rgb = RGBColor(0x33, 0x33, 0x33)
    return p

def wtable(doc, headers, rows):
    t = doc.add_table(rows=len(rows) + 1, cols=len(headers))
    t.alignment = WD_TABLE_ALIGNMENT.CENTER
    t.autofit = False

    for j, h in enumerate(headers):
        cell = t.cell(0, j)
        set_cell_bg(cell, "1F3864")
        p = cell.paragraphs[0]
        p.alignment = WD_ALIGN_PARAGRAPH.CENTER
        r = p.add_run(h)
        r.bold = True
        r.font.name = "Segoe UI"
        r.font.size = Pt(9.5)
        r.font.color.rgb = RGBColor(0xFF, 0xFF, 0xFF)

    for i, row in enumerate(rows):
        bg = "F2F6FC" if i % 2 == 1 else "FFFFFF"
        for j, val in enumerate(row):
            cell = t.cell(i + 1, j)
            set_cell_bg(cell, bg)
            p = cell.paragraphs[0]
            if j == 0:
                p.alignment = WD_ALIGN_PARAGRAPH.LEFT
            elif j in (1, 2, 3):
                p.alignment = WD_ALIGN_PARAGRAPH.CENTER
            else:
                p.alignment = WD_ALIGN_PARAGRAPH.LEFT
            r = p.add_run(str(val))
            r.font.name = "Segoe UI"
            r.font.size = Pt(9)
            if j == 0:
                r.bold = True
                r.font.color.rgb = RGBColor(0x1F, 0x38, 0x64)
            else:
                r.font.color.rgb = RGBColor(0x33, 0x33, 0x33)

    for row in t.rows:
        for cell in row.cells:
            tcPr = cell._tc.get_or_add_tcPr()
            tcBorders = OxmlElement('w:tcBorders')
            for b_name in ['top', 'left', 'bottom', 'right']:
                b = OxmlElement(f'w:{b_name}')
                b.set(qn('w:val'), 'single')
                b.set(qn('w:sz'), '4')
                b.set(qn('w:space'), '0')
                b.set(qn('w:color'), 'D9D9D9')
                tcBorders.append(b)
            tcPr.append(tcBorders)
    return t

def wimg(doc, img_path, width=6.0, caption=""):
    if not os.path.exists(img_path):
        return
    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    doc.add_picture(img_path, width=Inches(width))
    if caption:
        cp = doc.add_paragraph()
        cp.alignment = WD_ALIGN_PARAGRAPH.CENTER
        cr = cp.add_run(caption)
        cr.italic = True
        cr.font.size = Pt(9)
        cr.font.color.rgb = RGBColor(0x59, 0x59, 0x59)

def build_word_report(cm_path, rest_path, curves_path):
    print("Building Standalone Word Intelligence Report (.docx)...")
    doc = docx.Document()

    # Document Title Block
    title = doc.add_heading("DishDetect SA: Vegan vs. Vegetarian Intelligence Report", level=0)
    title.alignment = WD_ALIGN_PARAGRAPH.CENTER
    for r in title.runs:
        r.font.color.rgb = RGBColor(0x1F, 0x38, 0x64)
        r.font.name = "Segoe UI"
        r.bold = True

    sub = doc.add_paragraph()
    sub.alignment = WD_ALIGN_PARAGRAPH.CENTER
    sr = sub.add_run("Meat-Free Cohort Analysis (15 Dishes) & YOLOv8-Seg Polygon Cropping Benchmark\nDate: October 2026 · Confidential & Proprietary Research Document")
    sr.italic = True
    sr.font.size = Pt(10)
    sr.font.color.rgb = RGBColor(0x59, 0x59, 0x59)
    doc.add_paragraph()

    # ── 1. EXECUTIVE SUMMARY
    whead(doc, "1. Executive Summary & Breakthrough Results", color="1F3864")
    wbody(doc,
        "In culinary AI, identifying whether a meatless meal is vegan requires analyzing invisible or subtle dairy byproducts "
        "(cheese, butter, eggs, milk, ashta cream, and samneh / clarified butter). In Middle Eastern cuisine, flatbreads and pastries "
        "often present confusing visual textures: for example, Za’atar Manaqish (wild thyme in olive oil) can visually mimic melted cheese "
        "under reflective restaurant spotlights, while table clutter introduces extraneous noise. "
        "To resolve this, we implemented Strategy 1 (YOLOv8-Seg Polygon Mask Food Cropping) to isolate the edible food mass. "
        "The model achieved an exceptional 95.83% accuracy on real-world unseen restaurant photography (+12.50% boost over uncropped baseline), "
        "eliminating false classifications on Za’atar Manaqish completely."
    )
    
    wtable(doc,
        ["Benchmark Metric", "Uncropped Baseline", "YOLOv8-Seg Cropped", "Observed Delta / Impact"],
        [
            ["Real-World Restaurant Accuracy", "83.33% (20 / 24)", "95.83% (23 / 24)", "+12.50% (Significant real-world gain)"],
            ["Za’atar Manaqish Restaurant Acc.",  "40.0% (2 / 5)",   "100.0% (5 / 5)",  "+60.0% (Zero false cheese predictions)"],
            ["Held-Out Test Accuracy",          "85.05%",           "85.58%",          "+0.53% (Consistent generalization)"],
            ["Vegan Held-Out Recall",            "79.03%",           "83.61%",          "+4.58% (Fewer missed vegan dishes)"],
            ["Vegan Held-Out F1-Score",          "85.96%",           "87.18%",          "+1.22% (Harmonic precision-recall gain)"],
            ["Vegetarian Precision",             "76.36%",           "79.17%",          "+2.81% (Higher purity on dairy detections)"],
            ["Excluded Meat Categories",         "10 Dishes (0 imgs)", "10 Dishes (0 imgs)", "Strictly 0 meat/fish images in dataset"],
        ]
    )

    doc.add_paragraph()

    # ── 2. DATASET COHORT & INGREDIENT AUDIT
    whead(doc, "2. Meat-Free Dataset Cohort (15 Dishes, 703 Images)", color="1F3864")
    wbody(doc,
        "All 10 meat, poultry, fish, and seafood dishes (Mixed Grill, Shawarma, Shish Tawook, Grilled Kofta, Meat Maqluba, "
        "Meat Pie, Kofta Stew, Shrimp Sayadiyah, Stuffed Pigeon, and Meat Madfoon) were strictly excluded. "
        "The cohort consists exclusively of the 15 meatless dishes in the dataset:"
    )

    wtable(doc,
        ["Dish Name", "Category", "Images", "Key Ingredients & Animal Byproduct Audit"],
        [
            ["Hummus",            "Vegan",      "61", "Chickpeas, tahini (sesame paste), olive oil, garlic. Naturally 100% plant-based."],
            ["Tamees Bread",      "Vegan",      "53", "Plain Saudi tandoor wheat bread: Flour, water, yeast, salt, vegetable oil."],
            ["Za’atar Manakish",  "Vegan",      "52", "Flatbread dough topped with wild thyme, sumac, sesame seeds, and extra virgin olive oil."],
            ["Okra Tagine",       "Vegan",      "45", "Bamya bil-Zayt: Baby okra pods simmered in tomato gravy, cilantro, garlic, olive oil."],
            ["Sesame Brittle",    "Vegan",      "45", "Simsimiya: Toasted sesame seeds bound with sugar syrup, glucose, lemon."],
            ["Spinach Fatayer",   "Vegan",      "44", "Levantine spinach pie: Spinach, onions, sumac, pine nuts in olive oil dough."],
            ["Onion Soup",        "Vegan",      "42", "Shorbat Basal: Caramelized onions, olive oil, vegetable broth, cumin."],
            ["Pomegranate Fattoush","Vegan",    "37", "Romaine, tomatoes, cucumbers, toasted pita chips, sumac, pomegranate molasses."],
            ["Spinach Stew",      "Vegan",      "32", "Sabanegh bil-Zayt: Fresh spinach, chickpeas, garlic, coriander, olive oil."],
            ["Fatayer Cheese",    "Vegetarian", "86", "Pastry filled with dairy cheese (Akkawi / Halloumi / Nabulsi) and butter."],
            ["Saffron Cake",      "Vegetarian", "59", "Sponge cake made with eggs, soaked in saffron condensed milk & heavy cream."],
            ["Rice Pudding",      "Vegetarian", "57", "Roz Bel Laban: Short-grain rice cooked in whole cow's milk and ashta cream."],
            ["Pistachio Sweet",   "Vegetarian", "57", "Halawet el-Jibn / Mafroukeh: Contains sweet cheese, ashta, and clarified butter."],
            ["Pistachio Tamriya", "Vegetarian", "21", "Date confection pan-roasted in clarified butter (samneh / ghee)."],
            ["Tamriya",           "Vegetarian", "13", "Authentic Saudi date confection pan-roasted in samneh (animal butterfat)."],
        ]
    )

    doc.add_paragraph()

    # ── 3. VISUALIZATIONS & PERFORMANCE CHARTS
    whead(doc, "3. Benchmark Visualizations & Comparative Performance", color="1F3864")
    wimg(doc, rest_path, width=6.2, caption="Figure 1 — Real-World Arabic Restaurant Test: Uncropped Baseline vs. YOLOv8-Seg Cropping (+12.50% Jump)")

    doc.add_paragraph()
    wimg(doc, cm_path, width=6.2, caption="Figure 2 — Held-Out Test Confusion Matrices: Baseline vs. YOLOv8-Seg Cropped")

    doc.add_paragraph()
    if os.path.exists(curves_path):
        wimg(doc, curves_path, width=6.0, caption="Figure 3 — Two-Stage Transfer Learning Convergence Curves (AdamW + Cosine Annealing on Segmented Crops)")

    doc.add_paragraph()

    # ── 4. REAL-WORLD RESTAURANT TEST
    whead(doc, "4. Real-World Restaurant Test Breakdown (24 Unseen Photos)", color="1F3864")
    wbody(doc,
        "The model was benchmarked against the external Arabic restaurant test set. Below is the per-dish accuracy before and after YOLOv8-Seg polygon cropping:"
    )
    wtable(doc,
        ["Restaurant Dish", "Ground Truth", "Baseline Acc.", "YOLOv8-Seg Acc.", "Culinary Analysis & Error Recovery"],
        [
            ["Hummus",          "Vegan",      "5 / 5 (100%)",  "5 / 5 (100%)",  "Perfect plant-based legume dip recognition in both modes."],
            ["Okra Tagine",     "Vegan",      "4 / 4 (100%)",  "4 / 4 (100%)",  "Accurately recognizes olive-oil vegetable stew across all lighting."],
            ["Tamees Bread",    "Vegan",      "5 / 5 (100%)",  "5 / 5 (100%)",  "Perfect recognition of Saudi tandoor flatbread without dairy."],
            ["Cheese Fatayer",  "Vegetarian", "4 / 5 (80%)",   "4 / 5 (80%)",   "Detects dairy cheese stuffing; 1 pastry misclassified as vegan."],
            ["Za’atar Manaqish","Vegan",      "2 / 5 (40%)",   "5 / 5 (100%)",  "+60% GAIN. Mask isolates dark thyme/oil layer, removing cheese false positive."],
            ["OVERALL",         "—",          "20 / 24 (83.33%)", "23 / 24 (95.83%)", "+12.50% OVERALL BOOST across real-world restaurant photography."],
        ]
    )

    doc.add_paragraph()

    # ── 5. CODE ARTIFACTS & CLI USAGE
    whead(doc, "5. Standalone Package & CLI Usage", color="1F3864")
    wbullet(doc, "Package Directory:", "Vegan_vs_Vegetarian_Classifier/")
    wbullet(doc, "YOLOv8-Seg Checkpoint:", "Vegan_vs_Vegetarian_Classifier/checkpoints/mobilenet_v3_large_vegan_veg_yolo_seg_best.pth")
    wbullet(doc, "Baseline Checkpoint:", "Vegan_vs_Vegetarian_Classifier/checkpoints/mobilenet_v3_large_vegan_veg_best.pth")
    wbullet(doc, "Inference Tool (YOLO-Seg):", ".venv/bin/python Vegan_vs_Vegetarian_Classifier/scripts/predict.py --image <path> --yolo_seg")
    wbullet(doc, "Inference Tool (Baseline):", ".venv/bin/python Vegan_vs_Vegetarian_Classifier/scripts/predict.py --image <path> --baseline")
    wbullet(doc, "Restaurant Evaluator:", ".venv/bin/python Vegan_vs_Vegetarian_Classifier/scripts/evaluate_restaurant.py --all")

    doc.save(DOCX_OUT)
    print(f"  Word Report saved → {DOCX_OUT}")

# ─────────────────────────────────────────────────────────────────────────────
# 3. EXCEL WORKBOOK GENERATION
# ─────────────────────────────────────────────────────────────────────────────
def build_excel_sheet():
    print("Building Multi-Tab Excel Metrics Workbook...")
    wb = openpyxl.Workbook()
    wb.remove(wb.active)

    # Styles
    f_title = Font(name="Calibri", size=15, bold=True, color="1F3864")
    f_hdr   = Font(name="Calibri", size=10, bold=True, color="FFFFFF")
    f_body  = Font(name="Calibri", size=10, color="000000")
    f_bold  = Font(name="Calibri", size=10, bold=True, color="000000")

    fill_navy  = PatternFill(start_color="1F3864", end_color="1F3864", fill_type="solid")
    fill_blue  = PatternFill(start_color="2E75B6", end_color="2E75B6", fill_type="solid")
    fill_green = PatternFill(start_color="E2EFDA", end_color="E2EFDA", fill_type="solid")
    fill_alt   = PatternFill(start_color="F2F6FC", end_color="F2F6FC", fill_type="solid")

    thin_border = Border(
        left=Side(style='thin', color='D9D9D9'),
        right=Side(style='thin', color='D9D9D9'),
        top=Side(style='thin', color='D9D9D9'),
        bottom=Side(style='thin', color='D9D9D9')
    )

    # ── TAB 1: EXECUTIVE OVERVIEW
    ws1 = wb.create_sheet(title="Executive_Overview")
    ws1.views.sheetView[0].showGridLines = True
    
    ws1["A1"] = "DISHDETECT SA — VEGAN VS. VEGETARIAN (MEAT-FREE) INTELLIGENCE"
    ws1["A1"].font = f_title
    ws1["A2"] = "Performance Benchmarks & YOLOv8-Seg Ablation · Generated: " + datetime.now().strftime("%Y-%m-%d")
    ws1["A2"].font = Font(name="Calibri", size=10, italic=True, color="595959")

    ws1.append([])
    ws1.append(["Benchmark Metric", "Uncropped Baseline", "YOLOv8-Seg Cropped", "Observed Delta", "Culinary Significance"])
    for col in range(1, 6):
        cell = ws1.cell(row=4, column=col)
        cell.font = f_hdr; cell.fill = fill_navy
        cell.alignment = Alignment(horizontal="center", vertical="center")

    overview_rows = [
        ["Real-World Restaurant Accuracy", "83.33% (20/24)", "95.83% (23/24)", "+12.50%", "Huge boost in commercial restaurant robustness"],
        ["Za’atar Manaqish Restaurant Acc.", "40.0% (2/5)", "100.0% (5/5)", "+60.00%", "Polygon crop isolates herb paste from table reflections"],
        ["Held-Out Test Accuracy", "85.05% (91/107)", "85.58% (89/104)", "+0.53%", "Zero data leakage held-out test split"],
        ["Vegan Test Recall", "79.03%", "83.61%", "+4.58%", "Reduced false negative rate on plant-based foods"],
        ["Vegan Test F1-Score", "85.96%", "87.18%", "+1.22%", "Harmonic precision-recall improvement"],
        ["Vegetarian Test Precision", "76.36%", "79.17%", "+2.81%", "Higher accuracy in flagging dairy/egg dishes"],
        ["Total Training Images", "704 images", "703 images", "-1 img", "1 corrupted image skipped during polygon extraction"],
        ["Polygon Mask Extraction Rate", "0%", "92.9%", "+92.9%", "653 tight seg-masks, 50 center-crops"],
        ["Excluded Meat Dishes", "10 Dishes (0 imgs)", "10 Dishes (0 imgs)", "0", "All animal flesh strictly excluded"],
        ["Architecture", "MobileNetV3-Large", "MobileNetV3-Large", "Same", "Pretrained on ImageNet1K"],
    ]
    for r_idx, row in enumerate(overview_rows, start=5):
        ws1.append(row)
        for c_idx in range(1, 6):
            cell = ws1.cell(row=r_idx, column=c_idx)
            cell.font = f_bold if c_idx == 1 else f_body
            cell.border = thin_border
            if r_idx % 2 == 0: cell.fill = fill_alt

    # ── TAB 2: DISH TAXONOMY
    ws2 = wb.create_sheet(title="Dish_Taxonomy_15")
    ws2.views.sheetView[0].showGridLines = True
    ws2["A1"] = "15 MEAT-FREE ARABIC DISHES — CULINARY INGREDIENT TAXONOMY"
    ws2["A1"].font = f_title
    ws2.append([])
    ws2.append(["Dish Name", "Dietary Category", "Images", "Key Plant Ingredients", "Animal Byproduct Detected", "Culinary Rationale"])
    for col in range(1, 7):
        cell = ws2.cell(row=3, column=col)
        cell.font = f_hdr; cell.fill = fill_navy

    taxonomy_rows = [
        ["Hummus", "Vegan", 61, "Chickpeas, tahini, olive oil, lemon, garlic", "None", "Naturally 100% plant-based legume dip."],
        ["Tamees Bread", "Vegan", 53, "Wheat flour, water, yeast, salt, vegetable oil", "None", "Plain Saudi tandoor bread; zero dairy."],
        ["Za’atar Manakish", "Vegan", 52, "Wild thyme, sumac, sesame, extra virgin olive oil", "None", "Traditional Levantine oil flatbread."],
        ["Okra Tagine", "Vegan", 45, "Okra pods, tomato gravy, coriander, garlic, olive oil", "None", "Bamya bil-Zayt: Vegetable stew in olive oil."],
        ["Sesame Brittle", "Vegan", 45, "Roasted sesame seeds, boiled sugar syrup, lemon", "None", "Simsimiya: Confection made with sugar, not honey."],
        ["Spinach Fatayer", "Vegan", 44, "Spinach, onions, sumac, pine nuts, olive oil dough", "None", "Traditional meatless pie in olive oil dough."],
        ["Onion Soup", "Vegan", 42, "Caramelized onions, olive oil, vegetable broth, cumin", "None", "Shorbat Basal without butter or chicken stock."],
        ["Pomegranate Fattoush", "Vegan", 37, "Romaine, tomatoes, cucumbers, pita, pomegranate molasses", "None", "Fresh salad with pomegranate molasses dressing."],
        ["Spinach Stew", "Vegan", 32, "Spinach, chickpeas, garlic, coriander, olive oil", "None", "Sabanegh bil-Zayt: Meatless stew in olive oil."],
        ["Fatayer Cheese", "Vegetarian", 86, "Pastry dough, Akkawi/Halloumi cheese, butter", "Dairy Cheese & Butter", "Stuffed with cow/sheep dairy cheese."],
        ["Saffron Cake", "Vegetarian", 59, "Flour, eggs, whole milk, condensed milk, cream", "Eggs & Dairy Cream", "Eggs and soaked condensed dairy milk sauce."],
        ["Rice Pudding", "Vegetarian", 57, "Rice, cow's milk, cream (ashta), sugar", "Whole Cow's Milk & Cream", "Base is 100% whole dairy milk."],
        ["Pistachio Sweet", "Vegetarian", 57, "Semolina dough, sweet cheese, ashta, samneh", "Sweet Cheese & Ashta Cream", "Halawet el-Jibn / Mafroukeh with sweet cheese."],
        ["Pistachio Tamriya", "Vegetarian", 21, "Date paste, clarified butter (samneh / ghee), pistachios", "Dairy Ghee / Samneh", "Pan-roasted in animal butterfat."],
        ["Tamriya", "Vegetarian", 13, "Dates, roasted flour, clarified butter (samneh / ghee)", "Dairy Ghee / Samneh", "Authentic Saudi date confection in samneh."],
    ]
    for r_idx, row in enumerate(taxonomy_rows, start=4):
        ws2.append(row)
        for c_idx in range(1, 7):
            cell = ws2.cell(row=r_idx, column=c_idx)
            cell.font = f_body; cell.border = thin_border
            if row[1] == "Vegan": cell.fill = fill_green
            elif r_idx % 2 == 0: cell.fill = fill_alt

    # ── TAB 3: RESTAURANT COMPARISON LOG (24 IMAGES)
    ws3 = wb.create_sheet(title="Restaurant_Comparison_24")
    ws3.views.sheetView[0].showGridLines = True
    ws3["A1"] = "REAL-WORLD RESTAURANT PREDICTIONS: BASELINE (83.33%) VS YOLOV8-SEG (95.83%)"
    ws3["A1"].font = f_title
    ws3.append([])
    ws3.append(["Dish Name", "File Name", "Ground Truth", "Baseline Pred", "Baseline Correct?", "YOLO-Seg Pred", "YOLO-Seg Correct?", "YOLO Confidence"])
    for col in range(1, 9):
        cell = ws3.cell(row=3, column=col)
        cell.font = f_hdr; cell.fill = fill_navy

    base_csv = os.path.join(PACKAGE_DIR, "results", "restaurant_evaluation", "restaurant_vegan_veg_predictions.csv")
    seg_csv  = os.path.join(PACKAGE_DIR, "results", "restaurant_evaluation", "restaurant_vegan_veg_yolo_seg_predictions.csv")

    base_dict = {}
    if os.path.exists(base_csv):
        with open(base_csv) as f:
            for r in csv.DictReader(f):
                base_dict[(r["dish"], r["file"])] = r

    if os.path.exists(seg_csv):
        with open(seg_csv) as f:
            for idx, r in enumerate(csv.DictReader(f), start=1):
                key = (r["dish"], r["file"])
                b_row = base_dict.get(key, {})
                b_pred = b_row.get("pred_label", "—")
                b_corr = "CORRECT" if b_row.get("correct", "").lower() == "true" else "INCORRECT"
                s_pred = r["pred_label"]
                s_corr = "CORRECT" if r["correct"].lower() == "true" else "INCORRECT"
                s_conf = f"{float(r['confidence'])*100:.2f}%"

                ws3.append([r["dish"], r["file"], r["true_label"], b_pred, b_corr, s_pred, s_corr, s_conf])
                r_num = 3 + idx
                for col in range(1, 9):
                    cell = ws3.cell(row=r_num, column=col)
                    cell.font = f_body; cell.border = thin_border
                    if s_corr == "CORRECT": cell.fill = fill_green
                    else: cell.fill = PatternFill(start_color="FCE4D6", end_color="FCE4D6", fill_type="solid")

    # ── TAB 4: HELD-OUT TEST PREDICTIONS (YOLO-SEG)
    ws4 = wb.create_sheet(title="YOLO_Seg_Test_104")
    ws4.views.sheetView[0].showGridLines = True
    ws4["A1"] = "YOLOV8-SEG HELD-OUT TEST SET PREDICTIONS (104 IMAGES — 85.58% ACCURACY)"
    ws4["A1"].font = f_title
    ws4.append([])
    ws4.append(["Sample #", "Image Path", "True Category", "Predicted Category", "Correct?", "Confidence Score"])
    for col in range(1, 7):
        cell = ws4.cell(row=3, column=col)
        cell.font = f_hdr; cell.fill = fill_navy

    test_pred_csv = os.path.join(ROOT, "results_mobilenet_v3_large_vegan_veg_yolo_seg", "test", "predictions.csv")
    if os.path.exists(test_pred_csv):
        with open(test_pred_csv) as f:
            reader = csv.reader(f)
            next(reader, None)
            for idx, r in enumerate(reader, start=1):
                if len(r) >= 5:
                    is_corr = "CORRECT" if r[3] == "1" else "INCORRECT"
                    ws4.append([idx, os.path.basename(r[0]), r[1], r[2], is_corr, f"{float(r[4])*100:.2f}%"])
                    for col in range(1, 7):
                        cell = ws4.cell(row=3+idx, column=col)
                        cell.font = f_body; cell.border = thin_border
                        if is_corr == "INCORRECT": cell.fill = PatternFill(start_color="FCE4D6", end_color="FCE4D6", fill_type="solid")

    # ── TAB 5: BASELINE TEST PREDICTIONS (107 IMAGES)
    ws5 = wb.create_sheet(title="Baseline_Test_107")
    ws5.views.sheetView[0].showGridLines = True
    ws5["A1"] = "UNCROPPED BASELINE HELD-OUT TEST PREDICTIONS (107 IMAGES — 85.05% ACCURACY)"
    ws5["A1"].font = f_title
    ws5.append([])
    ws5.append(["Sample #", "Image Path", "True Category", "Predicted Category", "Correct?", "Confidence Score"])
    for col in range(1, 7):
        cell = ws5.cell(row=3, column=col)
        cell.font = f_hdr; cell.fill = fill_navy

    base_test_csv = os.path.join(PACKAGE_DIR, "results", "test", "predictions.csv")
    if os.path.exists(base_test_csv):
        with open(base_test_csv) as f:
            reader = csv.reader(f)
            next(reader, None)
            for idx, r in enumerate(reader, start=1):
                if len(r) >= 5:
                    is_corr = "CORRECT" if r[3] == "1" else "INCORRECT"
                    ws5.append([idx, os.path.basename(r[0]), r[1], r[2], is_corr, f"{float(r[4])*100:.2f}%"])
                    for col in range(1, 7):
                        cell = ws5.cell(row=3+idx, column=col)
                        cell.font = f_body; cell.border = thin_border
                        if is_corr == "INCORRECT": cell.fill = PatternFill(start_color="FCE4D6", end_color="FCE4D6", fill_type="solid")

    # Auto-adjust column widths
    for ws in wb.worksheets:
        for col in ws.columns:
            max_len = max(len(str(cell.value or "")) for cell in col)
            col_letter = get_column_letter(col[0].column)
            ws.column_dimensions[col_letter].width = max(max_len + 3, 12)

    wb.save(XLSX_OUT)
    print(f"  Excel Workbook saved → {XLSX_OUT}")

def main():
    cm_path, rest_path, curves_path = generate_charts()
    build_word_report(cm_path, rest_path, curves_path)
    build_excel_sheet()

    # Mirror to package directory
    import shutil
    os.makedirs(os.path.join(PACKAGE_DIR, "docs"), exist_ok=True)
    shutil.copy2(DOCX_OUT, os.path.join(PACKAGE_DIR, "docs", "Vegan_vs_Vegetarian_Intelligence_Report.docx"))
    shutil.copy2(XLSX_OUT, os.path.join(PACKAGE_DIR, "docs", "Vegan_vs_Vegetarian_Metrics_Sheet.xlsx"))
    print("\nAll doc sheets and workbooks successfully created and mirrored!")

if __name__ == "__main__":
    main()
