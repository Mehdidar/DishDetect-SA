"""
generate_vegan_doc_and_sheet.py
Generates:
1. Comprehensive Word Document Report:
   docs/Vegan_vs_Vegetarian_Intelligence_Report.docx
2. Complete Multi-Tab Excel Workbook / Sheet:
   docs/Vegan_vs_Vegetarian_Metrics_Sheet.xlsx
Includes all graphs, confusion matrices, metrics, per-dish breakdowns,
predictions logs, and ingredient analyses.
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
import matplotlib.patches as mpatches
from docx import Document
from docx.shared import Pt, RGBColor, Cm, Inches
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.enum.table import WD_TABLE_ALIGNMENT
from docx.oxml.ns import qn
from docx.oxml import OxmlElement
import openpyxl
from openpyxl.styles import Font, PatternFill, Alignment, Border, Side
from openpyxl.utils import get_column_letter

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
DOCS = os.path.join(ROOT, "docs")
CHARTS = os.path.join(DOCS, "_charts")
os.makedirs(CHARTS, exist_ok=True)

PACKAGE_DIR = os.path.join(ROOT, "Vegan_vs_Vegetarian_Classifier")
DOCX_OUT = os.path.join(DOCS, "Vegan_vs_Vegetarian_Intelligence_Report.docx")
XLSX_OUT = os.path.join(DOCS, "Vegan_vs_Vegetarian_Metrics_Sheet.xlsx")

# Palettes
NAVY = "#1F3864"
BLUE = "#1A6EBD"
TEAL = "#00B4D8"
GREEN = "#2ECC71"
ORANGE = "#F5A623"
RED = "#E94560"
WHITE = "#FFFFFF"
SILVER = "#B0BEC5"
DARK = "#0A1628"

# ─────────────────────────────────────────────────────────────────────────────
# 1. CHART GENERATION
# ─────────────────────────────────────────────────────────────────────────────
def generate_charts():
    print("Generating specialized charts for Vegan vs. Vegetarian report...")
    
    # ── Chart A: Confusion Matrix Heatmaps (Side-by-side Val & Test)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    fig.patch.set_facecolor(DARK)
    ax1.set_facecolor(DARK)
    ax2.set_facecolor(DARK)

    # Val CM: TP=56, TN=39, FP=6, FN=6
    val_cm = np.array([[56, 6], [6, 39]]) # [Pred Vegan, Pred Veg]
    test_cm = np.array([[52, 10], [3, 42]]) # [Pred Vegan, Pred Veg]

    cmap_veg = matplotlib.colormaps["Blues"]
    cmap_test = matplotlib.colormaps["Greens"]

    im1 = ax1.imshow(val_cm, cmap=cmap_veg, aspect="auto")
    im2 = ax2.imshow(test_cm, cmap=cmap_test, aspect="auto")

    classes = ["Vegan", "Vegetarian"]
    for ax, cm, title in [(ax1, val_cm, "Validation Set (107 imgs) — 88.79% Acc"),
                          (ax2, test_cm, "Held-Out Test Set (107 imgs) — 87.85% Acc")]:
        ax.set_xticks([0, 1])
        ax.set_yticks([0, 1])
        ax.set_xticklabels(classes, color=WHITE, fontsize=10, fontweight="bold")
        ax.set_yticklabels(classes, color=WHITE, fontsize=10, fontweight="bold")
        ax.set_xlabel("Predicted Label", color=SILVER, fontsize=10, labelpad=8)
        ax.set_ylabel("True Label", color=SILVER, fontsize=10, labelpad=8)
        ax.set_title(title, color=WHITE, fontsize=11.5, fontweight="bold", pad=10)
        
        for i in range(2):
            for j in range(2):
                val = cm[i, j]
                pct = val / cm.sum() * 100
                ax.text(j, i, f"{val}\n({pct:.1f}%)", ha="center", va="center",
                        color=WHITE if val > 20 else SILVER, fontweight="bold", fontsize=11)
        ax.spines["bottom"].set_color(SILVER)
        ax.spines["left"].set_color(SILVER)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    fig.suptitle("MobileNetV3-Large Confusion Matrices (Vegan vs. Vegetarian)",
                 color=WHITE, fontsize=13.5, fontweight="bold", y=1.03)
    fig.tight_layout()
    cm_path = os.path.join(CHARTS, "vegan_veg_confusion_matrix.png")
    fig.savefig(cm_path, dpi=160, bbox_inches="tight", facecolor=DARK)
    plt.close(fig)

    # ── Chart B: Real-World Restaurant Accuracy Bar Chart
    fig, ax = plt.subplots(figsize=(10, 5))
    fig.patch.set_facecolor(DARK)
    ax.set_facecolor(DARK)
    ax.spines["bottom"].set_color(SILVER)
    ax.spines["left"].set_color(SILVER)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(colors=SILVER, labelsize=9.5)
    ax.xaxis.label.set_color(SILVER)
    ax.yaxis.label.set_color(SILVER)

    dishes = ["Hummus\n(Vegan)", "Okra Tagine\n(Vegan)", "Tamees Bread\n(Vegan)", "Cheese Fatayer\n(Vegetarian)", "Za’atara Manaqish\n(Vegan)"]
    scores = [100.0, 100.0, 100.0, 80.0, 40.0]
    bar_cols = [GREEN, GREEN, GREEN, ORANGE, RED]

    bars = ax.bar(dishes, scores, color=bar_cols, width=0.55, edgecolor=WHITE, lw=1)
    for b, s in zip(bars, scores):
        ax.text(b.get_x() + b.get_width()/2, s + 2, f"{s:.1f}%",
                ha="center", va="bottom", color=WHITE, fontweight="bold", fontsize=10)

    ax.axhline(83.33, color=BLUE, linestyle="--", lw=1.5, label="Overall Restaurant Accuracy (83.33%)")
    ax.set_ylim(0, 115)
    ax.set_ylabel("Accuracy (%)", color=SILVER, fontsize=10)
    ax.set_title("Real-World Restaurant Test: Per-Dish Accuracy (24 Unseen Photos)",
                 color=WHITE, fontsize=12.5, fontweight="bold", pad=12)
    ax.legend(facecolor=DARK, edgecolor=SILVER, labelcolor=WHITE, loc="upper right")
    fig.tight_layout()
    rest_path = os.path.join(CHARTS, "vegan_veg_restaurant_accuracy.png")
    fig.savefig(rest_path, dpi=160, bbox_inches="tight", facecolor=DARK)
    plt.close(fig)

    # ── Chart C: Training Curves
    src_curves = os.path.join(PACKAGE_DIR, "results", "training_curves.png")
    dest_curves = os.path.join(CHARTS, "vegan_veg_training_curves.png")
    if os.path.exists(src_curves):
        import shutil
        shutil.copy2(src_curves, dest_curves)

    print("  Charts generated successfully.")
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

def wbody(doc, text, italic=False, bold=False, color="262626", size=10):
    par = doc.add_paragraph()
    r = par.add_run(text)
    r.italic = italic; r.bold = bold
    r.font.color.rgb = RGBColor.from_string(color)
    r.font.size = Pt(size)
    r.font.name = "Segoe UI"
    return par

def wbullet(doc, label, value=None, color_label="1F3864"):
    par = doc.add_paragraph(style="List Bullet")
    r = par.add_run(label)
    r.bold = True; r.font.color.rgb = RGBColor.from_string(color_label)
    r.font.name = "Segoe UI"; r.font.size = Pt(10)
    if value:
        r2 = par.add_run("  " + value)
        r2.font.name = "Segoe UI"; r2.font.size = Pt(10)
    return par

def wimg(doc, path, width=6.0, caption=""):
    if os.path.exists(path):
        doc.add_picture(path, width=Inches(width))
        if caption:
            cp = doc.add_paragraph(caption)
            cp.alignment = WD_ALIGN_PARAGRAPH.CENTER
            for r in cp.runs:
                r.italic = True; r.font.size = Pt(9)
                r.font.color.rgb = RGBColor.from_string("595959")
    else:
        wbody(doc, f"[Missing image: {os.path.basename(path)}]", italic=True)

def wtable(doc, headers, rows, hcol="1F3864", alt1="F2F6FC", alt2="FFFFFF"):
    tbl = doc.add_table(rows=1, cols=len(headers))
    tbl.style = "Table Grid"; tbl.alignment = WD_TABLE_ALIGNMENT.CENTER
    hdr = tbl.rows[0].cells
    for i, h in enumerate(headers):
        hdr[i].text = h; set_cell_bg(hdr[i], hcol)
        for r in hdr[i].paragraphs[0].runs:
            r.font.color.rgb = RGBColor.from_string("FFFFFF"); r.bold = True
            r.font.name = "Segoe UI"; r.font.size = Pt(9.5)
    for idx, row in enumerate(rows):
        rc = tbl.add_row().cells
        for i, v in enumerate(row):
            rc[i].text = str(v)
            set_cell_bg(rc[i], alt1 if idx % 2 == 0 else alt2)
            for r in rc[i].paragraphs[0].runs:
                r.font.name = "Segoe UI"; r.font.size = Pt(9)
    return tbl

def build_word_report(cm_path, rest_path, curves_path):
    print("Building Word Document Report...")
    doc = Document()
    for sec in doc.sections:
        sec.top_margin = Cm(1.8); sec.bottom_margin = Cm(1.8)
        sec.left_margin = Cm(2.2); sec.right_margin = Cm(2.2)

    # ── COVER / HEADER
    cover = doc.add_paragraph(); cover.alignment = WD_ALIGN_PARAGRAPH.CENTER
    r = cover.add_run("\n🌱 DishDetect SA\nVEGAN VS. VEGETARIAN (MEAT-FREE) INTELLIGENCE REPORT")
    r.bold = True; r.font.size = Pt(22); r.font.color.rgb = RGBColor.from_string("1F3864")
    
    sub = doc.add_paragraph(); sub.alignment = WD_ALIGN_PARAGRAPH.CENTER
    sr = sub.add_run("Deep Learning Discrimination of Plant-Based vs. Dairy/Egg/Ghee Arabic Dishes\n" + datetime.now().strftime("%d %B %Y"))
    sr.italic = True; sr.font.size = Pt(11); sr.font.color.rgb = RGBColor.from_string("595959")
    
    doc.add_paragraph()

    # ── 1. EXECUTIVE SUMMARY
    whead(doc, "1. Executive Summary", color="1F3864")
    wbody(doc,
        "In modern dietary compliance and nutrition technology, identifying whether a dish contains animal products "
        "requires going beyond detecting meat. While vegetarians abstain from animal flesh, they frequently consume "
        "dairy (cheese, milk, yogurt), eggs, and clarified butter (samneh / ghee). In contrast, strict vegans require "
        "100% plant-based foods. This report documents the dedicated Vegan vs. Vegetarian classification model trained "
        "on the meat-free cohort of 15 Middle Eastern / Arabic dishes, completely excluding all 10 meat/fish categories."
    )
    
    wtable(doc,
        ["Benchmark Metric", "Validation Set (107 imgs)", "Held-Out Test Set (107 imgs)", "Real Restaurant Test (24 imgs)"],
        [
            ["Overall Accuracy",    "88.79%",  "87.85%",  "83.33%"],
            ["Weighted Precision",  "0.8879",  "0.8875",  "—"],
            ["Weighted Recall",     "0.8879",  "0.8785",  "—"],
            ["Weighted F1-Score",   "0.8879",  "0.8793",  "—"],
            ["Vegan Precision",     "0.90",    "0.95 (95%)", "—"],
            ["Vegetarian Recall",   "0.87",    "0.93 (93%)", "—"],
        ]
    )

    doc.add_paragraph()

    # ── 2. DATASET COHORT & INGREDIENT AUDIT
    whead(doc, "2. Meat-Free Dataset Cohort (15 Dishes, 704 Images)", color="1F3864")
    wbody(doc,
        "All meat, poultry, fish, and seafood dishes (Mixed Grill, Shawarma, Shish Tawook, Grilled Kofta, Meat Maqluba, "
        "Meat Pie, Kofta Stew, Shrimp Sayadiyah, Stuffed Pigeon, and Meat Madfoon) were strictly excluded. "
        "The model evaluates exclusively on the 15 meatless dishes in the dataset:"
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
    whead(doc, "3. Benchmark Visualizations & Confusion Matrices", color="1F3864")
    wimg(doc, cm_path, width=6.2, caption="Figure 1 — MobileNetV3-Large Validation and Test Split Confusion Matrices")

    doc.add_paragraph()
    wimg(doc, rest_path, width=6.0, caption="Figure 2 — Real-World Arabic Restaurant Test: Per-Dish Accuracy Breakdown")

    doc.add_paragraph()
    if os.path.exists(curves_path):
        wimg(doc, curves_path, width=6.0, caption="Figure 3 — Two-Stage Transfer Learning Convergence Curves (AdamW + Cosine Annealing)")

    doc.add_paragraph()

    # ── 4. REAL-WORLD RESTAURANT TEST
    whead(doc, "4. Real-World Restaurant Test Results (24 Images)", color="1F3864")
    wbody(doc,
        "The model was tested against unseen, real-world Arabic restaurant photos from the Restaurant testing data set:"
    )
    wtable(doc,
        ["Restaurant Dish", "Ground Truth", "Correct / Total", "Accuracy", "Culinary Analysis"],
        [
            ["Hummus",          "Vegan",      "5 / 5", "100.0%", "Perfect plant-based legume dip recognition."],
            ["Okra Tagine",     "Vegan",      "4 / 4", "100.0%", "Correctly identified as olive-oil vegetable stew."],
            ["Tamees Bread",    "Vegan",      "5 / 5", "100.0%", "Perfect recognition of tandoor plain flatbread."],
            ["Cheese Fatayer",  "Vegetarian", "4 / 5", "80.0%",  "Successfully detects dairy cheese filling; recovered from previous veg failure."],
            ["Za’atara Manaqish","Vegan",     "2 / 5", "40.0%",  "Heavy olive oil sheen sometimes mimics melted cheese glaze."],
            ["OVERALL",         "—",          "20 / 24", "83.33%", "High real-world accuracy on unseen restaurant photography."],
        ]
    )

    doc.add_paragraph()

    # ── 5. CODE ARTIFACTS & CLI USAGE
    whead(doc, "5. Standalone Package & CLI Usage", color="1F3864")
    wbullet(doc, "Package Directory:", "Vegan_vs_Vegetarian_Classifier/")
    wbullet(doc, "Checkpoint Weights:", "Vegan_vs_Vegetarian_Classifier/checkpoints/mobilenet_v3_large_vegan_veg_best.pth")
    wbullet(doc, "Inference Tool:", ".venv/bin/python Vegan_vs_Vegetarian_Classifier/scripts/predict.py --image <path>")
    wbullet(doc, "Split Evaluator:", ".venv/bin/python Vegan_vs_Vegetarian_Classifier/scripts/evaluate.py --split test")
    wbullet(doc, "Restaurant Evaluator:", ".venv/bin/python Vegan_vs_Vegetarian_Classifier/scripts/evaluate_restaurant.py")

    doc.save(DOCX_OUT)
    print(f"  Word Report saved → {DOCX_OUT}")

# ─────────────────────────────────────────────────────────────────────────────
# 3. EXCEL WORKBOOK GENERATION
# ─────────────────────────────────────────────────────────────────────────────
def build_excel_sheet():
    print("Building Multi-Tab Excel Metrics Workbook...")
    wb = openpyxl.Workbook()
    # Remove default sheet
    wb.remove(wb.active)

    # Styles
    f_title = Font(name="Calibri", size=15, bold=True, color="1F3864")
    f_sec   = Font(name="Calibri", size=12, bold=True, color="FFFFFF")
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
    ws1["A2"] = "Complete Performance Benchmarks & Split Summaries · Generated: " + datetime.now().strftime("%Y-%m-%d")
    ws1["A2"].font = Font(name="Calibri", size=10, italic=True, color="595959")

    ws1.append([])
    ws1.append(["Metric Category", "Validation Set", "Held-Out Test Set", "Restaurant Test Set", "Notes"])
    for col in range(1, 6):
        cell = ws1.cell(row=4, column=col)
        cell.font = f_hdr; cell.fill = fill_navy
        cell.alignment = Alignment(horizontal="center", vertical="center")

    overview_rows = [
        ["Total Images Evaluated", "107", "107", "24", "Three-way hash split (70/15/15) + external restaurant set"],
        ["Overall Accuracy", "88.79%", "87.85%", "83.33%", "High generalization on meat-free cohort"],
        ["Weighted Precision", "0.8879", "0.8875", "—", "Balanced across both classes"],
        ["Weighted Recall", "0.8879", "0.8785", "—", "Balanced across both classes"],
        ["Weighted F1-Score", "0.8879", "0.8793", "—", "Stable convergence across both stages"],
        ["Vegan Precision", "0.90", "0.95 (95%)", "—", "Only 3 vegetarian images misclassified as vegan"],
        ["Vegetarian Recall", "0.87", "0.93 (93%)", "—", "93% of dairy/ghee dishes accurately detected"],
        ["Positive Class", "Vegan", "Vegan", "Vegan", "0: Vegan, 1: Vegetarian"],
        ["Architecture", "MobileNetV3-Large", "MobileNetV3-Large", "MobileNetV3-Large", "Pretrained on ImageNet1K"],
        ["Input Resolution", "224x224", "224x224", "224x224", "Standard TorchVision resolution"],
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

    # ── TAB 3: HELD-OUT TEST PREDICTIONS LOG
    ws3 = wb.create_sheet(title="Test_Predictions_107")
    ws3.views.sheetView[0].showGridLines = True
    ws3["A1"] = "HELD-OUT TEST SET PREDICTIONS (107 IMAGES — 87.85% ACCURACY)"
    ws3["A1"].font = f_title
    ws3.append([])
    ws3.append(["Sample #", "Image Path", "True Category", "Predicted Category", "Correct?", "Confidence Score"])
    for col in range(1, 7):
        cell = ws3.cell(row=3, column=col)
        cell.font = f_hdr; cell.fill = fill_navy

    test_pred_csv = os.path.join(PACKAGE_DIR, "results", "test", "predictions.csv")
    if os.path.exists(test_pred_csv):
        with open(test_pred_csv) as f:
            reader = csv.reader(f)
            next(reader, None) # skip header
            for idx, r in enumerate(reader, start=1):
                if len(r) >= 5:
                    is_corr = "CORRECT" if r[3] == "1" else "INCORRECT"
                    row = [idx, os.path.basename(r[0]), r[1], r[2], is_corr, f"{float(r[4])*100:.2f}%"]
                    ws3.append(row)
                    for col in range(1, 7):
                        cell = ws3.cell(row=3+idx, column=col)
                        cell.font = f_body; cell.border = thin_border
                        if is_corr == "INCORRECT": cell.fill = PatternFill(start_color="FCE4D6", end_color="FCE4D6", fill_type="solid")

    # ── TAB 4: RESTAURANT PREDICTIONS LOG
    ws4 = wb.create_sheet(title="Restaurant_Predictions_24")
    ws4.views.sheetView[0].showGridLines = True
    ws4["A1"] = "REAL-WORLD RESTAURANT PREDICTIONS LOG (24 IMAGES — 83.33% ACCURACY)"
    ws4["A1"].font = f_title
    ws4.append([])
    ws4.append(["Dish Folder", "File Name", "True Category", "Predicted Category", "Correct?", "Confidence"])
    for col in range(1, 7):
        cell = ws4.cell(row=3, column=col)
        cell.font = f_hdr; cell.fill = fill_navy

    rest_pred_csv = os.path.join(PACKAGE_DIR, "results", "restaurant_evaluation", "restaurant_vegan_veg_predictions.csv")
    if os.path.exists(rest_pred_csv):
        with open(rest_pred_csv) as f:
            reader = csv.DictReader(f)
            for idx, r in enumerate(reader, start=1):
                is_corr = "CORRECT" if r["correct"].lower() == "true" else "INCORRECT"
                conf_pct = f"{float(r['confidence'])*100:.2f}%"
                row = [r["dish"], r["file"], r["true_label"], r["pred_label"], is_corr, conf_pct]
                ws4.append(row)
                for col in range(1, 7):
                    cell = ws4.cell(row=3+idx, column=col)
                    cell.font = f_body; cell.border = thin_border
                    if is_corr == "CORRECT": cell.fill = fill_green
                    else: cell.fill = PatternFill(start_color="FCE4D6", end_color="FCE4D6", fill_type="solid")

    # Auto-adjust column widths on all sheets
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
    shutil.copy2(DOCX_OUT, os.path.join(PACKAGE_DIR, "docs", "Vegan_vs_Vegetarian_Intelligence_Report.docx"))
    shutil.copy2(XLSX_OUT, os.path.join(PACKAGE_DIR, "docs", "Vegan_vs_Vegetarian_Metrics_Sheet.xlsx"))
    print("\nAll doc sheets and workbooks successfully created and mirrored!")

if __name__ == "__main__":
    main()
