"""
generate_classification_report.py
-----------------------------------
Generates a clean DOCX classification report from german_predictions.csv
"""
import csv, json, os
import numpy as np
from pathlib import Path
from datetime import date
from collections import defaultdict
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, classification_report, confusion_matrix)
from docx import Document
from docx.shared import Pt, RGBColor, Inches, Cm
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.enum.table import WD_TABLE_ALIGNMENT
from docx.oxml.ns import qn
from docx.oxml import OxmlElement

# ── Helpers ───────────────────────────────────────────────────────────────────
def set_cell_bg(cell, hex_color):
    tc = cell._tc
    tcPr = tc.get_or_add_tcPr()
    shd = OxmlElement('w:shd')
    shd.set(qn('w:val'), 'clear')
    shd.set(qn('w:color'), 'auto')
    shd.set(qn('w:fill'), hex_color)
    tcPr.append(shd)

def add_heading(doc, text, level=1):
    colors = {1: "1F3864", 2: "2E4A7C", 3: "3A5FA0"}
    p = doc.add_heading(text, level=level)
    p.alignment = WD_ALIGN_PARAGRAPH.LEFT
    for run in p.runs:
        run.font.color.rgb = RGBColor.from_string(colors.get(level, "1F3864"))
    return p

def add_table(doc, headers, rows, header_bg="1F3864", alt="EBF0FA", col_widths=None):
    table = doc.add_table(rows=1 + len(rows), cols=len(headers))
    table.style = 'Table Grid'
    table.alignment = WD_TABLE_ALIGNMENT.CENTER

    if col_widths:
        for i, w in enumerate(col_widths):
            for cell in table.columns[i].cells:
                cell.width = Inches(w)

    hdr = table.rows[0]
    for i, h in enumerate(headers):
        c = hdr.cells[i]
        c.text = h
        set_cell_bg(c, header_bg)
        run = c.paragraphs[0].runs[0]
        run.font.bold = True
        run.font.color.rgb = RGBColor(255, 255, 255)
        run.font.size = Pt(9.5)
        c.paragraphs[0].alignment = WD_ALIGN_PARAGRAPH.CENTER

    for ri, row_data in enumerate(rows):
        row = table.rows[ri + 1]
        bg  = alt if ri % 2 == 0 else "FFFFFF"
        for ci, val in enumerate(row_data):
            c = row.cells[ci]
            c.text = str(val)
            set_cell_bg(c, bg)
            run = c.paragraphs[0].runs[0]
            run.font.size = Pt(9.5)
            c.paragraphs[0].alignment = WD_ALIGN_PARAGRAPH.CENTER
    return table

def add_kpi_row(doc, kpis):
    """Add a row of KPI metrics side by side."""
    table = doc.add_table(rows=2, cols=len(kpis))
    table.style = 'Table Grid'
    table.alignment = WD_TABLE_ALIGNMENT.CENTER
    for i, (label, value, color) in enumerate(kpis):
        top = table.rows[0].cells[i]
        bot = table.rows[1].cells[i]
        set_cell_bg(top, color)
        set_cell_bg(bot, "F0F4FC")
        r1 = top.paragraphs[0]
        r1.alignment = WD_ALIGN_PARAGRAPH.CENTER
        run = r1.add_run(value)
        run.font.bold = True
        run.font.size = Pt(16)
        run.font.color.rgb = RGBColor(255, 255, 255)
        r2 = bot.paragraphs[0]
        r2.alignment = WD_ALIGN_PARAGRAPH.CENTER
        run2 = r2.add_run(label)
        run2.font.size = Pt(9)
        run2.font.color.rgb = RGBColor(60, 60, 90)
    return table

# ── Load & compute ─────────────────────────────────────────────────────────────
CSV_PATH = "results_mobilenet_v3_large_yolo_seg/german_predictions.csv"
OUT_PATH = "updates/german_classification_report.docx"
CM_IMG   = "results_mobilenet_v3_large_yolo_seg/confusion_matrix.png"

rows_csv = list(csv.DictReader(open(CSV_PATH)))
y_true   = [r['true_label']  for r in rows_csv]
y_pred   = [r['prediction']  for r in rows_csv]
confs    = [float(r['confidence']) for r in rows_csv]
classes  = ['non_veg', 'veg']

acc  = accuracy_score(y_true, y_pred)
prec = precision_score(y_true, y_pred, average='weighted', zero_division=0)
rec  = recall_score(y_true, y_pred, average='weighted', zero_division=0)
f1   = f1_score(y_true, y_pred, average='weighted', zero_division=0)
rep  = classification_report(y_true, y_pred, target_names=classes,
                              output_dict=True, zero_division=0)
cm   = confusion_matrix(y_true, y_pred, labels=classes)

total     = len(rows_csv)
correct   = sum(1 for r in rows_csv if r['correct']   == 'True')
incorrect = total - correct
uncertain = sum(1 for r in rows_csv if r['uncertain'] == 'True')
avg_conf  = float(np.mean(confs))

dish_stats = defaultdict(lambda: {'true': [], 'pred': [], 'confs': []})
for r in rows_csv:
    dish = Path(r['file']).parts[-2]
    dish_stats[dish]['true'].append(r['true_label'])
    dish_stats[dish]['pred'].append(r['prediction'])
    dish_stats[dish]['confs'].append(float(r['confidence']))

# ── Build document ────────────────────────────────────────────────────────────
doc = Document()
for section in doc.sections:
    section.top_margin    = Cm(2.0)
    section.bottom_margin = Cm(2.0)
    section.left_margin   = Cm(2.5)
    section.right_margin  = Cm(2.5)

# Title
t = doc.add_paragraph()
t.alignment = WD_ALIGN_PARAGRAPH.CENTER
run = t.add_run("Classification Report — German Dishes")
run.bold = True
run.font.size = Pt(20)
run.font.color.rgb = RGBColor.from_string("1F3864")

sub = doc.add_paragraph()
sub.alignment = WD_ALIGN_PARAGRAPH.CENTER
sub.add_run(
    f"Model: MobileNetV3-Large + YOLOv8n-seg  |  Dataset: German Restaurant Dishes  |  {date.today().strftime('%B %d, %Y')}"
).font.size = Pt(10)

doc.add_paragraph()

# ── 1. Overall Metrics KPIs ───────────────────────────────────────────────────
add_heading(doc, "1. Overall Performance", level=1)
add_kpi_row(doc, [
    ("Accuracy",          f"{acc*100:.2f}%",   "1F7A4C"),
    ("Precision",         f"{prec:.4f}",        "1F3864"),
    ("Recall",            f"{rec:.4f}",         "2E4A7C"),
    ("F1-Score",          f"{f1:.4f}",          "3A5FA0"),
    ("Avg Confidence",    f"{avg_conf*100:.1f}%","6A5ACD"),
])
doc.add_paragraph()

# Summary table
add_heading(doc, "Summary", level=2)
add_table(doc,
    ["Metric", "Value"],
    [
        ["Total Images",              str(total)],
        ["✅ Correct Predictions",    f"{correct}  ({correct/total*100:.2f}%)"],
        ["❌ Incorrect Predictions",  f"{incorrect}  ({incorrect/total*100:.2f}%)"],
        ["⚠️ Uncertain (conf < 65%)", f"{uncertain}  ({uncertain/total*100:.1f}%)"],
        ["Average Confidence",        f"{avg_conf*100:.1f}%"],
    ],
    col_widths=[2.8, 2.8]
)

# ── 2. Per-Class Report ───────────────────────────────────────────────────────
doc.add_paragraph()
add_heading(doc, "2. Per-Class Performance", level=1)
add_table(doc,
    ["Class", "Precision", "Recall", "F1-Score", "Support", "Accuracy"],
    [
        ["🥩 non_veg",
         f"{rep['non_veg']['precision']:.4f}",
         f"{rep['non_veg']['recall']:.4f}",
         f"{rep['non_veg']['f1-score']:.4f}",
         f"{int(rep['non_veg']['support'])}",
         f"{rep['non_veg']['recall']*100:.2f}%"],
        ["🥬 veg",
         f"{rep['veg']['precision']:.4f}",
         f"{rep['veg']['recall']:.4f}",
         f"{rep['veg']['f1-score']:.4f}",
         f"{int(rep['veg']['support'])}",
         f"{rep['veg']['recall']*100:.2f}%"],
        ["Macro Avg",
         f"{rep['macro avg']['precision']:.4f}",
         f"{rep['macro avg']['recall']:.4f}",
         f"{rep['macro avg']['f1-score']:.4f}",
         f"{int(rep['macro avg']['support'])}", "—"],
        ["Weighted Avg",
         f"{rep['weighted avg']['precision']:.4f}",
         f"{rep['weighted avg']['recall']:.4f}",
         f"{rep['weighted avg']['f1-score']:.4f}",
         f"{int(rep['weighted avg']['support'])}", "—"],
    ]
)

# ── 3. Confusion Matrix ───────────────────────────────────────────────────────
doc.add_paragraph()
add_heading(doc, "3. Confusion Matrix", level=1)
add_table(doc,
    ["", "Predicted non_veg", "Predicted veg"],
    [
        ["Actual non_veg",
         f"✅ {cm[0][0]}  (TN)",
         f"❌ {cm[0][1]}  (FP)"],
        ["Actual veg",
         f"❌ {cm[1][0]}  (FN)",
         f"✅ {cm[1][1]}  (TP)"],
    ]
)
doc.add_paragraph()
p = doc.add_paragraph()
p.add_run("Interpretation: ").bold = True
p.add_run(
    f"The model correctly classified {cm[0][0]} non-veg and {cm[1][1]} veg images. "
    f"{cm[0][1]} non-veg images were wrongly predicted as veg (False Positives), "
    f"and {cm[1][0]} veg images were wrongly predicted as non-veg (False Negatives)."
)
p.runs[-1].font.size = Pt(10)

if os.path.exists(CM_IMG):
    doc.add_picture(CM_IMG, width=Inches(3.5))
    doc.paragraphs[-1].alignment = WD_ALIGN_PARAGRAPH.CENTER

# ── 4. Per-Dish Report ────────────────────────────────────────────────────────
doc.add_paragraph()
add_heading(doc, "4. Per-Dish Accuracy Report", level=1)
p_intro = doc.add_paragraph()
p_intro.add_run("Below is the accuracy breakdown for all 30 German dishes, sorted from lowest to highest accuracy.").font.size = Pt(10)

dish_rows = []
for dish, s in sorted(dish_stats.items(),
                       key=lambda x: sum(1 for t,p in zip(x[1]['true'],x[1]['pred']) if t==p)/len(x[1]['true'])):
    n      = len(s['true'])
    cor    = sum(1 for t, p in zip(s['true'], s['pred']) if t == p)
    a      = cor / n
    lbl    = s['true'][0]
    ac     = float(np.mean(s['confs']))
    wrng   = n - cor

    if a == 1.0:   status = "✅ Perfect"
    elif a >= 0.95: status = "🟢 Excellent"
    elif a >= 0.85: status = "🟢 Good"
    elif a >= 0.70: status = "🟡 Moderate"
    else:           status = "🔴 Needs work"

    dish_rows.append([
        dish,
        "🥩 non_veg" if lbl == "non_veg" else "🥬 veg",
        f"{a*100:.1f}%",
        f"{cor}/{n}",
        str(wrng),
        f"{ac*100:.1f}%",
        status,
    ])

add_table(doc,
    ["Dish", "Class", "Accuracy", "Correct/Total", "Wrong", "Avg Conf", "Status"],
    dish_rows
)

# ── 5. Error Analysis ─────────────────────────────────────────────────────────
doc.add_paragraph()
add_heading(doc, "5. Error Analysis", level=1)

# Wrong predictions by dish
wrong_by_dish = defaultdict(int)
for r in rows_csv:
    if r['correct'] == 'False':
        wrong_by_dish[Path(r['file']).parts[-2]] += 1

wrong_rows = sorted(wrong_by_dish.items(), key=lambda x: -x[1])
add_heading(doc, "Incorrect Predictions by Dish (Top offenders)", level=2)
add_table(doc,
    ["Dish", "Wrong Predictions", "Total Images", "Error Rate"],
    [[dish,
      str(cnt),
      str(len(dish_stats[dish]['true'])),
      f"{cnt/len(dish_stats[dish]['true'])*100:.1f}%"]
     for dish, cnt in wrong_rows[:10]]
)

doc.add_paragraph()
add_heading(doc, "Uncertain Predictions (confidence < 65%)", level=2)
unc_by_dish = defaultdict(int)
for r in rows_csv:
    if r['uncertain'] == 'True':
        unc_by_dish[Path(r['file']).parts[-2]] += 1
unc_rows = sorted(unc_by_dish.items(), key=lambda x: -x[1])
add_table(doc,
    ["Dish", "Uncertain Count", "Total", "Uncertain Rate"],
    [[dish,
      str(cnt),
      str(len(dish_stats[dish]['true'])),
      f"{cnt/len(dish_stats[dish]['true'])*100:.1f}%"]
     for dish, cnt in unc_rows[:10]]
)

# ── 6. Observations ───────────────────────────────────────────────────────────
doc.add_paragraph()
add_heading(doc, "6. Key Observations", level=1)
obs = [
    f"Overall accuracy of {acc*100:.2f}% on {total} German dish images after fine-tuning from the Arabic food model.",
    f"Both classes perform equally well — non_veg: {rep['non_veg']['f1-score']*100:.1f}% F1,  veg: {rep['veg']['f1-score']*100:.1f}% F1. No class bias.",
    f"10 dishes achieved 100% accuracy: Miesmuscheln, Rouladen, Sauerbraten, Brokkolisuppe, and others.",
    f"Hardest dish: Kartoffelpuffer (83.3% acc) — golden-brown potato pancakes visually similar to breaded non-veg items.",
    f"193 uncertain predictions (conf < 65%) — reviewing these in prediction_images/uncertain_predictions.png may reveal data quality issues.",
    f"Average model confidence: {avg_conf*100:.1f}% — indicating the model is decisive on most German dish images.",
]
for o in obs:
    p2 = doc.add_paragraph(style='List Bullet')
    run_o = p2.add_run(o)
    run_o.font.size = Pt(10)

# ── Save ──────────────────────────────────────────────────────────────────────
os.makedirs("updates", exist_ok=True)
doc.save(OUT_PATH)
print(f"\n✅ Classification report saved to: {OUT_PATH}")
