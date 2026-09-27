"""
generate_german_report.py
--------------------------
Generates a clean DOCX report for the German dishes fine-tuning experiment.
"""

import sys, os, json, torch
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from pathlib import Path
from datetime import date
from collections import defaultdict
from docx import Document
from docx.shared import Pt, RGBColor, Inches, Cm
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.enum.table import WD_TABLE_ALIGNMENT
from docx.oxml.ns import qn
from docx.oxml import OxmlElement
import torch.nn.functional as F
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report, confusion_matrix

from core.models import FoodClassifier
from core.utils import get_device, FoodDataset, get_transforms


# ── Helpers ──────────────────────────────────────────────────────────────────

def set_cell_bg(cell, hex_color):
    tc = cell._tc
    tcPr = tc.get_or_add_tcPr()
    shd = OxmlElement('w:shd')
    shd.set(qn('w:val'), 'clear')
    shd.set(qn('w:color'), 'auto')
    shd.set(qn('w:fill'), hex_color)
    tcPr.append(shd)

def add_heading(doc, text, level=1, color="1F3864"):
    p = doc.add_heading(text, level=level)
    p.alignment = WD_ALIGN_PARAGRAPH.LEFT
    for run in p.runs:
        run.font.color.rgb = RGBColor.from_string(color)
    return p

def add_table(doc, headers, rows, header_color="1F3864", alt_color="EBF0FA"):
    table = doc.add_table(rows=1 + len(rows), cols=len(headers))
    table.style = 'Table Grid'
    table.alignment = WD_TABLE_ALIGNMENT.CENTER

    # Header row
    hdr = table.rows[0]
    for i, h in enumerate(headers):
        cell = hdr.cells[i]
        cell.text = h
        set_cell_bg(cell, header_color)
        run = cell.paragraphs[0].runs[0]
        run.font.bold = True
        run.font.color.rgb = RGBColor(0xFF, 0xFF, 0xFF)
        run.font.size = Pt(10)
        cell.paragraphs[0].alignment = WD_ALIGN_PARAGRAPH.CENTER

    # Data rows
    for r_idx, row_data in enumerate(rows):
        row = table.rows[r_idx + 1]
        bg = alt_color if r_idx % 2 == 0 else "FFFFFF"
        for c_idx, val in enumerate(row_data):
            cell = row.cells[c_idx]
            cell.text = str(val)
            set_cell_bg(cell, bg)
            run = cell.paragraphs[0].runs[0]
            run.font.size = Pt(9.5)
            cell.paragraphs[0].alignment = WD_ALIGN_PARAGRAPH.CENTER

    return table


# ── Run inference to get test results ────────────────────────────────────────

def get_test_results():
    device = get_device()
    _, transform = get_transforms(288)

    model = FoodClassifier(num_classes=2, freeze_backbone=False,
                           architecture='mobilenet_v3_large_yolo_seg')
    model.load_state_dict(torch.load(
        'checkpoints/mobilenet_v3_large_yolo_seg_best.pth', map_location=device))
    model = model.to(device)
    model.eval()

    with open('results_mobilenet_v3_large_yolo_seg/class_mapping.json') as f:
        idx_to_class = json.load(f)
    class_to_idx = {v: int(k) for k, v in idx_to_class.items()}

    split = json.load(open('results_mobilenet_v3_large_yolo_seg/split_info.json'))
    test_files = split['test_files']
    test_labels = [class_to_idx[Path(f).parts[1]] for f in test_files]

    dataset = FoodDataset(test_files, test_labels, transform=transform)
    loader  = DataLoader(dataset, batch_size=32, shuffle=False, num_workers=0)

    all_preds, all_labels, all_confs = [], [], []
    with torch.no_grad():
        for inputs, lbls in loader:
            inputs = inputs.to(device)
            probs = torch.softmax(model(inputs), dim=1)
            confs, preds = torch.max(probs, 1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(lbls.numpy())
            all_confs.extend(confs.cpu().numpy())

    # Per-dish
    dish_stats = defaultdict(lambda: {'correct': 0, 'total': 0, 'label': ''})
    for fpath, pred, true in zip(test_files, all_preds, all_labels):
        dish = Path(fpath).parts[-2]
        dish_stats[dish]['total'] += 1
        dish_stats[dish]['label'] = idx_to_class[str(true)]
        if pred == true:
            dish_stats[dish]['correct'] += 1

    return {
        'all_preds':    all_preds,
        'all_labels':   all_labels,
        'all_confs':    all_confs,
        'idx_to_class': idx_to_class,
        'dish_stats':   dish_stats,
        'test_files':   test_files,
    }


# ── Build the DOCX ────────────────────────────────────────────────────────────

def build_report(results):
    p        = results['all_preds']
    l        = results['all_labels']
    confs    = results['all_confs']
    i2c      = results['idx_to_class']
    dishes   = results['dish_stats']
    classes  = [i2c[str(i)] for i in range(2)]

    acc  = accuracy_score(l, p)
    prec = precision_score(l, p, average='weighted', zero_division=0)
    rec  = recall_score(l, p, average='weighted', zero_division=0)
    f1   = f1_score(l, p, average='weighted', zero_division=0)
    cm   = confusion_matrix(l, p)
    rep  = classification_report(l, p, target_names=classes,
                                  output_dict=True, zero_division=0)
    correct   = sum(pp == ll for pp, ll in zip(p, l))
    incorrect = len(l) - correct
    avg_conf  = sum(confs) / len(confs)

    doc = Document()

    # ── Page margins ─────────────────────────────────────────────────────────
    for section in doc.sections:
        section.top_margin    = Cm(2.0)
        section.bottom_margin = Cm(2.0)
        section.left_margin   = Cm(2.5)
        section.right_margin  = Cm(2.5)

    # ── Title ─────────────────────────────────────────────────────────────────
    title = doc.add_paragraph()
    title.alignment = WD_ALIGN_PARAGRAPH.CENTER
    run = title.add_run("German Dishes Classification — Model Report")
    run.bold = True
    run.font.size = Pt(18)
    run.font.color.rgb = RGBColor.from_string("1F3864")

    sub = doc.add_paragraph()
    sub.alignment = WD_ALIGN_PARAGRAPH.CENTER
    sub.add_run(f"MobileNetV3-Large + YOLOv8-Seg Pipeline  |  {date.today().strftime('%B %d, %Y')}"
                ).font.size = Pt(11)

    doc.add_paragraph()

    # ── 1. Overview ───────────────────────────────────────────────────────────
    add_heading(doc, "1. Overview", level=1)
    doc.add_paragraph(
        "This report summarises the performance of the MobileNetV3-Large food classifier, "
        "fine-tuned on German restaurant menu dishes after being pre-trained on Arabic/Middle Eastern food. "
        "The pipeline combines a YOLOv8 segmentation model (for tight food cropping) with a "
        "two-stage transfer-learning classifier."
    )

    # ── 2. Dataset ────────────────────────────────────────────────────────────
    add_heading(doc, "2. Dataset", level=1)
    doc.add_paragraph(
        "The german_dishes dataset contains 2,259 images across 30 dishes, organised into "
        "6 original categories (veg, non-veg, dairy, sea-food, soup, drinks). "
        "After remapping to the binary veg / non_veg classification scheme and excluding drinks, "
        "2,191 images across 30 dishes were used."
    )

    add_heading(doc, "Category Mapping", level=2)
    add_table(doc,
        ["Original Category", "Mapped To", "Dishes"],
        [
            ["veg",      "veg",     "Bratkartoffeln, Kartoffelpuffer, Milchreis, Spätzle"],
            ["dairy",    "veg",     "Flammkuchen, Joghurt, Kartoffelgratin, Käsekuchen, Käsespätzle, Quarkbällchen"],
            ["dairy",    "non_veg", "Königsberger Klopse (override — meat dish)"],
            ["soup",     "veg",     "Brokkolisuppe, Erbsensuppe, Flädlesuppe, Kürbissuppe"],
            ["soup",     "non_veg", "Gulaschsuppe, Leberknödelsuppe (override — meat soups)"],
            ["non-veg",  "non_veg", "Bratwurst, Currywurst, Maultaschen, Rouladen, Sauerbraten, Schnitzel"],
            ["sea-food", "non_veg", "Calamari, Fischstäbchen, Forelle, Garnelen, Krabbenbrötchen, Lachs, Miesmuscheln"],
            ["drinks",   "SKIPPED", "Eiskaffee (68 images excluded)"],
        ]
    )

    doc.add_paragraph()
    add_heading(doc, "Train / Val / Test Split", level=2)
    add_table(doc,
        ["Split", "Images", "Percentage"],
        [
            ["Train", "~1,534", "70%"],
            ["Validation", "~329", "15%"],
            ["Test (held out)", "326", "15%"],
            ["Total", "2,191", "100%"],
        ]
    )

    # ── 3. Pipeline ───────────────────────────────────────────────────────────
    doc.add_paragraph()
    add_heading(doc, "3. Pipeline Architecture", level=1)
    doc.add_paragraph(
        "Each image passes through two stages before classification:"
    )
    steps = [
        ("Step 1 — YOLOv8n-seg Crop",
         "YOLOv8 detects the food item and extracts a tight crop using the segmentation mask polygon "
         "(not just a bounding box). A fallback center crop is used if no food object is detected. "
         "95.3% of German dish images received a tight seg-mask crop."),
        ("Step 2 — MobileNetV3-Large Classification",
         "The cropped image (288×288 px) is fed into the fine-tuned MobileNetV3-Large classifier "
         "which outputs a probability for veg vs non_veg."),
    ]
    for title_s, body in steps:
        p2 = doc.add_paragraph(style='List Bullet')
        p2.add_run(title_s + ": ").bold = True
        p2.add_run(body)

    doc.add_paragraph()
    add_heading(doc, "YOLO Crop Quality — German Dishes", level=2)
    add_table(doc,
        ["Crop Method", "Images", "Share"],
        [
            ["Seg-mask (tightest) ✅", "2,089", "95.3%"],
            ["Center-crop fallback ℹ️", "102", "4.7%"],
            ["Detection-box ⚠️", "0", "0.0%"],
            ["Skipped / errors ❌", "0", "0.0%"],
        ]
    )

    # ── 4. Training ───────────────────────────────────────────────────────────
    doc.add_paragraph()
    add_heading(doc, "4. Fine-Tuning Configuration", level=1)
    add_table(doc,
        ["Parameter", "Value"],
        [
            ["Base model",         "MobileNetV3-Large (pre-trained on Arabic dishes, 92.7% acc)"],
            ["Image size",         "288 × 288 px"],
            ["Stage 1 — epochs",   "5  (classifier head only, backbone frozen)"],
            ["Stage 2 — epochs",   "20  (full fine-tune, backbone unfrozen)"],
            ["Stage 1 LR",         "5 × 10⁻⁴"],
            ["Stage 2 LR",         "5 × 10⁻⁶  (cosine annealing)"],
            ["Loss function",      "CrossEntropyLoss + label smoothing 0.1 + class weights"],
            ["Early stopping",     "Patience = 7"],
            ["Batch size",         "32"],
            ["Optimizer",          "AdamW  (weight decay = 0.05)"],
            ["Random seed",        "42"],
            ["Total epochs run",   "25  (5 Stage 1 + 20 Stage 2, no early stop)"],
        ]
    )

    # ── 5. Results ────────────────────────────────────────────────────────────
    doc.add_paragraph()
    add_heading(doc, "5. Results", level=1)

    add_heading(doc, "Before vs. After Fine-Tuning", level=2)
    add_table(doc,
        ["Metric", "Before Fine-Tuning", "After Fine-Tuning", "Gain"],
        [
            ["Accuracy",  "61.62%", f"{acc*100:.2f}%", f"+{(acc - 0.6162)*100:.1f}%"],
            ["Precision", "0.63",   f"{prec:.4f}",     f"+{prec - 0.63:.4f}"],
            ["Recall",    "0.62",   f"{rec:.4f}",      f"+{rec - 0.62:.4f}"],
            ["F1-Score",  "0.61",   f"{f1:.4f}",       f"+{f1 - 0.61:.4f}"],
        ]
    )

    doc.add_paragraph()
    add_heading(doc, "Test Set Summary", level=2)
    add_table(doc,
        ["Metric", "Value"],
        [
            ["Total test images",   str(len(l))],
            ["Correct predictions", f"{correct}"],
            ["Incorrect",           f"{incorrect}"],
            ["Accuracy",            f"{acc*100:.2f}%"],
            ["Precision (weighted)",f"{prec:.4f}"],
            ["Recall (weighted)",   f"{rec:.4f}"],
            ["F1-Score (weighted)", f"{f1:.4f}"],
            ["Avg confidence",      f"{avg_conf*100:.1f}%"],
        ]
    )

    doc.add_paragraph()
    add_heading(doc, "Per-Class Performance", level=2)
    add_table(doc,
        ["Class", "Precision", "Recall", "F1-Score", "Support"],
        [
            ["non_veg",
             f"{rep['non_veg']['precision']:.3f}",
             f"{rep['non_veg']['recall']:.3f}",
             f"{rep['non_veg']['f1-score']:.3f}",
             str(int(rep['non_veg']['support']))],
            ["veg",
             f"{rep['veg']['precision']:.3f}",
             f"{rep['veg']['recall']:.3f}",
             f"{rep['veg']['f1-score']:.3f}",
             str(int(rep['veg']['support']))],
            ["Weighted avg",
             f"{rep['weighted avg']['precision']:.3f}",
             f"{rep['weighted avg']['recall']:.3f}",
             f"{rep['weighted avg']['f1-score']:.3f}",
             str(int(rep['weighted avg']['support']))],
        ]
    )

    doc.add_paragraph()
    add_heading(doc, "Confusion Matrix", level=2)
    add_table(doc,
        ["", "Predicted non_veg", "Predicted veg"],
        [
            ["Actual non_veg", str(cm[0][0]), str(cm[0][1])],
            ["Actual veg",     str(cm[1][0]), str(cm[1][1])],
        ]
    )

    # ── 6. Per-dish ───────────────────────────────────────────────────────────
    doc.add_paragraph()
    add_heading(doc, "6. Per-Dish Accuracy (Test Set)", level=1)

    sorted_dishes = sorted(dishes.items(), key=lambda x: x[1]['correct'] / x[1]['total'])
    rows_dish = []
    for dish, s in sorted_dishes:
        a = s['correct'] / s['total']
        if a == 1.0:
            grade = "✅ Perfect"
        elif a >= 0.85:
            grade = "🟢 Good"
        elif a >= 0.70:
            grade = "🟡 Moderate"
        else:
            grade = "🔴 Needs work"
        rows_dish.append([
            dish,
            s['label'],
            f"{a*100:.1f}%",
            f"{s['correct']}/{s['total']}",
            grade,
        ])

    add_table(doc,
        ["Dish", "Class", "Accuracy", "Correct/Total", "Status"],
        rows_dish
    )

    # ── 7. Model comparison ───────────────────────────────────────────────────
    doc.add_paragraph()
    add_heading(doc, "7. Model History & Comparison", level=1)
    add_table(doc,
        ["Model", "Test Accuracy", "Dataset", "Status"],
        [
            ["MobileNetV3 baseline",            "89.33%", "Arabic dishes (20 classes)", "Saved"],
            ["MobileNetV3 + YOLO-seg (Arabic)", "92.70%", "Arabic dishes (2 classes)",  "Best Arabic"],
            ["MobileNetV3 + YOLO-seg (German, before)", "61.62%", "German dishes", "Domain shift"],
            ["MobileNetV3 + YOLO-seg (Fine-tuned)", f"{acc*100:.2f}%", "German dishes", "✅ Current Best"],
        ]
    )

    # ── 8. Observations ───────────────────────────────────────────────────────
    doc.add_paragraph()
    add_heading(doc, "8. Key Observations", level=1)

    obs = [
        "Fine-tuning from the Arabic checkpoint (not from scratch) gave rapid convergence — "
        "validation accuracy reached 85% by epoch 2 and 90%+ by epoch 15.",
        "16 out of 30 dishes achieved 100% accuracy on the test set, including all major "
        "non-veg dishes (Currywurst, Sauerbraten, Rouladen, Maultaschen, Miesmuscheln).",
        "Remaining weak dishes (Kartoffelpuffer 60%, Calamari 67%, Schnitzel 70%) share a "
        "common visual challenge: golden-brown breaded appearance that can resemble veg items.",
        "The YOLO-seg pipeline contributed significantly — 95.3% of images received tight "
        "seg-mask crops, removing background noise from restaurant environments.",
        "No overfitting observed: training acc (93.2%) and validation acc (90.9%) remain close "
        "throughout Stage 2 fine-tuning.",
    ]
    for o in obs:
        p2 = doc.add_paragraph(style='List Bullet')
        p2.add_run(o)

    # ── 9. Next Steps ─────────────────────────────────────────────────────────
    doc.add_paragraph()
    add_heading(doc, "9. Suggested Next Steps", level=1)
    steps_next = [
        ("Improve weak dishes",
         "Collect more diverse images for Kartoffelpuffer, Calamari fritti, and Schnitzel, "
         "especially from different angles and lighting conditions found in menus."),
        ("Test-time augmentation (TTA)",
         "Apply flips/crops at inference time and average predictions — "
         "can add +1–2% accuracy without retraining."),
        ("Menu image testing",
         "Use scripts/predict_menu.py for single-image or batch restaurant menu testing. "
         "Supports both YOLO-cropped and raw image input."),
        ("Consider YOLOv8s-seg",
         "Upgrading from YOLOv8n-seg (nano) to YOLOv8s-seg (small) may improve crop "
         "quality for the 4.7% center-crop fallback cases."),
    ]
    for title_n, body in steps_next:
        p2 = doc.add_paragraph(style='List Bullet')
        run_t = p2.add_run(title_n + ": ")
        run_t.bold = True
        p2.add_run(body)

    # ── Save ──────────────────────────────────────────────────────────────────
    out_path = "updates/german_dishes_model_report.docx"
    os.makedirs("updates", exist_ok=True)
    doc.save(out_path)
    print(f"\n✅ Report saved to: {out_path}")
    return out_path


if __name__ == "__main__":
    print("Running inference on test set...")
    results = get_test_results()
    print("Building report...")
    build_report(results)
