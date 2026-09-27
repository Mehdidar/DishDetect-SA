"""
Script to generate a Word document summarizing model performance.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from docx import Document
from docx.shared import Pt, RGBColor, Inches, Cm
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.enum.table import WD_TABLE_ALIGNMENT, WD_ALIGN_VERTICAL
from docx.oxml.ns import qn
from docx.oxml import OxmlElement
from datetime import datetime
import copy

OUTPUT_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "model_performance_report.docx")

# ── helpers ──────────────────────────────────────────────────────────────────

def set_cell_bg(cell, hex_color: str):
    """Fill a table cell with a solid background colour."""
    tc = cell._tc
    tcPr = tc.get_or_add_tcPr()
    shd = OxmlElement("w:shd")
    shd.set(qn("w:val"),   "clear")
    shd.set(qn("w:color"), "auto")
    shd.set(qn("w:fill"),  hex_color.upper())
    tcPr.append(shd)


def add_cell_border(table):
    """Add thin borders to every cell in a table."""
    tbl = table._tbl
    for tr in tbl.findall(qn("w:tr")):
        for tc in tr.findall(qn("w:tc")):
            tcPr = tc.get_or_add_tcPr()
            tcBorders = OxmlElement("w:tcBorders")
            for side in ("top", "left", "bottom", "right", "insideH", "insideV"):
                border = OxmlElement(f"w:{side}")
                border.set(qn("w:val"),   "single")
                border.set(qn("w:sz"),    "4")
                border.set(qn("w:space"), "0")
                border.set(qn("w:color"), "CCCCCC")
                tcBorders.append(border)
            tcPr.append(tcBorders)


def heading(doc, text, level=1, color="1F3864"):
    p = doc.add_heading(text, level=level)
    p.alignment = WD_ALIGN_PARAGRAPH.LEFT
    for run in p.runs:
        run.font.color.rgb = RGBColor.from_string(color)
    return p


def bullet(doc, text, bold_prefix=None):
    p = doc.add_paragraph(style="List Bullet")
    if bold_prefix:
        run = p.add_run(bold_prefix)
        run.bold = True
    p.add_run(text)
    return p


def add_table(doc, headers, rows, header_bg="1F3864", alt_bg="EBF3FB"):
    table = doc.add_table(rows=1 + len(rows), cols=len(headers))
    table.alignment = WD_TABLE_ALIGNMENT.CENTER
    table.style = "Table Grid"

    # header row
    hdr_row = table.rows[0]
    for i, h in enumerate(headers):
        cell = hdr_row.cells[i]
        cell.text = h
        cell.vertical_alignment = WD_ALIGN_VERTICAL.CENTER
        set_cell_bg(cell, header_bg)
        for run in cell.paragraphs[0].runs:
            run.bold = True
            run.font.color.rgb = RGBColor(0xFF, 0xFF, 0xFF)
            run.font.size = Pt(10)
        cell.paragraphs[0].alignment = WD_ALIGN_PARAGRAPH.CENTER

    # data rows
    for r_idx, row_data in enumerate(rows):
        row = table.rows[r_idx + 1]
        bg = alt_bg if r_idx % 2 == 1 else "FFFFFF"
        for c_idx, val in enumerate(row_data):
            cell = row.cells[c_idx]
            cell.text = str(val)
            set_cell_bg(cell, bg)
            cell.paragraphs[0].alignment = WD_ALIGN_PARAGRAPH.CENTER
            for run in cell.paragraphs[0].runs:
                run.font.size = Pt(10)

    add_cell_border(table)
    return table


# ── document ─────────────────────────────────────────────────────────────────

doc = Document()

# Page margins
section = doc.sections[0]
section.left_margin   = Cm(2.5)
section.right_margin  = Cm(2.5)
section.top_margin    = Cm(2.0)
section.bottom_margin = Cm(2.0)

# ── Title block ──────────────────────────────────────────────────────────────
title = doc.add_paragraph()
title.alignment = WD_ALIGN_PARAGRAPH.CENTER
run = title.add_run("🍽️  Food Classifier — Model Performance Report")
run.bold = True
run.font.size = Pt(20)
run.font.color.rgb = RGBColor.from_string("1F3864")

sub = doc.add_paragraph()
sub.alignment = WD_ALIGN_PARAGRAPH.CENTER
sub.add_run(f"Generated: {datetime.now().strftime('%B %d, %Y  %H:%M')}").italic = True

doc.add_paragraph()

# ── 1. Overview ──────────────────────────────────────────────────────────────
heading(doc, "1. Project Overview", level=1)
p = doc.add_paragraph(
    "This report summarises the training and evaluation results of the food classification "
    "pipeline. The task is binary classification: "
)
p.add_run("veg").bold = True
p.add_run(" vs ")
p.add_run("non_veg").bold = True
p.add_run(
    ". Three model variants were evaluated: MobileNet V3 Large, EfficientNet B2, "
    "and MobileNet V3 Large with YOLO-cropped inputs."
)

doc.add_paragraph()

# ── 2. Model Comparison ───────────────────────────────────────────────────────
heading(doc, "2. Model Comparison — Test Split", level=1)

add_table(
    doc,
    headers=["Model", "Accuracy", "Precision", "Recall", "F1-Score"],
    rows=[
        ["MobileNet V3 Large ★",  "92.2%", "92.4%", "92.2%", "92.2%"],
        ["EfficientNet B2",        "89.3%", "89.7%", "89.3%", "89.3%"],
        ["MobileNet V3 + YOLO",    "~81.7% (val)", "—",     "—",     "—"   ],
    ],
)

doc.add_paragraph()
doc.add_paragraph("★  Best performing model on the held-out test split.").italic = True
doc.add_paragraph()

# ── 3. Full Dataset Results ───────────────────────────────────────────────────
heading(doc, "3. Full Dataset Evaluation (1 203 images)", level=1)

add_table(
    doc,
    headers=["Model", "Accuracy", "non_veg F1", "veg F1"],
    rows=[
        ["MobileNet V3 Large",  "97.8%", "97%",  "98%"],
        ["EfficientNet B2",     "96.0%", "95%",  "96%"],
    ],
)

doc.add_paragraph()

# ── 4. Per-Class Results ──────────────────────────────────────────────────────
heading(doc, "4. Per-Class Results — Test Split", level=1)

heading(doc, "4.1  MobileNet V3 Large", level=2, color="2E75B6")
add_table(
    doc,
    headers=["Class", "Precision", "Recall", "F1-Score", "Support"],
    rows=[
        ["non_veg", "89%", "94%", "91%", "108"],
        ["veg",     "95%", "90%", "93%", "135"],
        ["Overall", "92.4%", "92.2%", "92.2%", "243"],
    ],
)

doc.add_paragraph()
heading(doc, "4.2  EfficientNet B2", level=2, color="2E75B6")
add_table(
    doc,
    headers=["Class", "Precision", "Recall", "F1-Score", "Support"],
    rows=[
        ["non_veg", "85%", "93%", "88%", "108"],
        ["veg",     "94%", "87%", "90%", "135"],
        ["Overall", "89.7%", "89.3%", "89.3%", "243"],
    ],
)

doc.add_paragraph()

# ── 5. Training Dynamics ──────────────────────────────────────────────────────
heading(doc, "5. Training Dynamics", level=1)

add_table(
    doc,
    headers=["Model", "Final Train Acc", "Final Val Acc", "Epochs", "Overfitting?"],
    rows=[
        ["MobileNet V3 Large",  "~99%",   "~82%",   "40", "⚠️ Moderate"],
        ["EfficientNet B2",     "~92.7%", "~89.7%", "40", "✅ Low"],
        ["MobileNet V3 + YOLO", "~99%",   "~81.7%", "40", "❌ High (~18% gap)"],
    ],
)

doc.add_paragraph()
p = doc.add_paragraph()
p.add_run("Key observation: ").bold = True
p.add_run(
    "EfficientNet B2 shows the best generalisation (smallest train/val gap) despite "
    "lower raw test accuracy. MobileNet V3 Large achieves the highest test accuracy but "
    "shows a moderate overfitting gap. The YOLO-cropped variant suffers from severe "
    "overfitting due to a small effective dataset size."
)

doc.add_paragraph()

# ── 6. Improvement Strategies ────────────────────────────────────────────────
heading(doc, "6. Recommended Improvement Strategies", level=1)

add_table(
    doc,
    headers=["Strategy", "Expected Gain", "Priority"],
    rows=[
        ["Stronger data augmentation (MixUp, CutMix)",        "+2–5%", "🔴 High"],
        ["Label smoothing in CrossEntropyLoss",                "+1–2%", "🔴 High"],
        ["Cosine annealing / OneCycleLR scheduler",           "+1–3%", "🔴 High"],
        ["Test-Time Augmentation (TTA)",                       "+1–2%", "🟠 Medium"],
        ["Ensemble: MobileNet + EfficientNet",                 "+2–4%", "🟠 Medium"],
        ["Larger backbone (EfficientNet B4/B5 or ConvNeXt)",  "+3–6%", "🟠 Medium"],
        ["Fix YOLO overfitting (dropout, weight decay, more data)", "Closes 18% gap", "🔴 High"],
    ],
)

doc.add_paragraph()

# ── 7. Dataset Summary ────────────────────────────────────────────────────────
heading(doc, "7. Dataset Summary", level=1)
add_table(
    doc,
    headers=["Split", "Images", "non_veg", "veg"],
    rows=[
        ["Train (70%)", "~842",  "~369",  "~473"],
        ["Val   (15%)", "~180",  "~79",   "~101"],
        ["Test  (15%)", "243",   "108",   "135"],
        ["Total",       "1 203", "527",   "676"],
    ],
)

doc.add_paragraph()

# ── 8. Notes ──────────────────────────────────────────────────────────────────
heading(doc, "8. Notes & Next Steps", level=1)
bullet(doc, "Retrain MobileNet V3 Large with cosine annealing + label smoothing.", "Action 1: ")
bullet(doc, "Add MixUp/CutMix augmentation to the training pipeline.",             "Action 2: ")
bullet(doc, "Collect more images for underrepresented classes.",                   "Action 3: ")
bullet(doc, "Evaluate ensemble of MobileNet + EfficientNet on the test set.",      "Action 4: ")
bullet(doc, "Fix YOLO overfitting — add weight decay, dropout, more crops.",       "Action 5: ")

doc.add_paragraph()

# ── Footer note ───────────────────────────────────────────────────────────────
footer_p = doc.add_paragraph()
footer_p.alignment = WD_ALIGN_PARAGRAPH.CENTER
run = footer_p.add_run("Food Classifier Project  •  Auto-generated report  •  " +
                        datetime.now().strftime("%Y-%m-%d"))
run.italic = True
run.font.size = Pt(9)
run.font.color.rgb = RGBColor(0x88, 0x88, 0x88)

# ── Save ──────────────────────────────────────────────────────────────────────
doc.save(OUTPUT_PATH)
print(f"✅  Report saved to: {OUTPUT_PATH}")
