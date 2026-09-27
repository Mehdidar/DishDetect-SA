"""
generate_prediction_images.py
------------------------------
Reads german_predictions.csv and generates visual prediction grids:
  - One image grid per dish (all images with prediction overlay)
  - A summary grid of wrong predictions only
  - A summary grid of uncertain predictions only

Output → results_mobilenet_v3_large_yolo_seg/prediction_images/
"""

import csv
import os
from pathlib import Path
from collections import defaultdict
from PIL import Image, ImageDraw, ImageFont

# ── Config ────────────────────────────────────────────────────────────────────
CSV_PATH   = "results_mobilenet_v3_large_yolo_seg/german_predictions.csv"
OUT_DIR    = "results_mobilenet_v3_large_yolo_seg/prediction_images"
THUMB_SIZE = (180, 180)        # each image thumbnail
COLS       = 10                # images per row in per-dish grids
PADDING    = 6                 # px between images
LABEL_H    = 42                # px for label bar below each image

# Colors
GREEN  = (46, 204, 113)        # correct
RED    = (231, 76,  60)        # incorrect
ORANGE = (243, 156, 18)        # uncertain but correct
DARK   = (30,  30,  30)
WHITE  = (255, 255, 255)
GREY   = (245, 245, 245)
HEADER_BG = (31, 56, 100)


def try_font(size):
    for name in [
        "/System/Library/Fonts/Helvetica.ttc",
        "/System/Library/Fonts/SFNSDisplay.ttf",
        "/Library/Fonts/Arial.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    ]:
        try:
            return ImageFont.truetype(name, size)
        except Exception:
            pass
    return ImageFont.load_default()


FONT_SM  = try_font(11)
FONT_MED = try_font(13)
FONT_LG  = try_font(16)
FONT_HDR = try_font(18)


def load_thumb(path):
    try:
        img = Image.open(path).convert("RGB")
        img.thumbnail(THUMB_SIZE, Image.LANCZOS)
        # Pad to exact THUMB_SIZE
        canvas = Image.new("RGB", THUMB_SIZE, (200, 200, 200))
        offset = ((THUMB_SIZE[0] - img.width) // 2,
                  (THUMB_SIZE[1] - img.height) // 2)
        canvas.paste(img, offset)
        return canvas
    except Exception:
        return Image.new("RGB", THUMB_SIZE, (180, 180, 180))


def border_color(row):
    correct   = row["correct"] == "True"
    uncertain = row["uncertain"] == "True"
    if correct and not uncertain:
        return GREEN
    if correct and uncertain:
        return ORANGE
    return RED


def label_text(row):
    pred   = row["prediction"].upper()
    conf   = float(row["confidence"]) * 100
    correct = row["correct"] == "True"
    mark   = "✓" if correct else "✗"
    return f"{mark} {pred}  {conf:.0f}%"


def make_cell(row):
    """Return a (THUMB_W + 2*PAD) × (THUMB_H + LABEL_H + 2*PAD) image for one prediction."""
    W = THUMB_SIZE[0] + 2 * PADDING
    H = THUMB_SIZE[1] + LABEL_H + 3 * PADDING

    cell = Image.new("RGB", (W, H), DARK)
    thumb = load_thumb(row["file"])

    # Border
    bc = border_color(row)
    border_size = 3
    bordered = Image.new("RGB",
                         (THUMB_SIZE[0] + 2*border_size, THUMB_SIZE[1] + 2*border_size),
                         bc)
    bordered.paste(thumb, (border_size, border_size))
    cell.paste(bordered, (PADDING - border_size, PADDING - border_size))

    # Label bar
    bar_y = PADDING + THUMB_SIZE[1] + PADDING
    draw  = ImageDraw.Draw(cell)
    draw.rectangle([0, bar_y, W, H], fill=bc)

    txt = label_text(row)
    draw.text((4, bar_y + 4),  txt,                 font=FONT_SM,  fill=WHITE)
    conf_val = float(row["confidence"]) * 100
    draw.text((4, bar_y + 18), f"conf: {conf_val:.1f}%", font=FONT_SM, fill=WHITE)

    return cell


def make_grid(rows, title, subtitle=""):
    """Build a full prediction grid image from a list of CSV rows."""
    if not rows:
        return None

    cols = min(COLS, len(rows))
    grid_rows = (len(rows) + cols - 1) // cols

    cell_w = THUMB_SIZE[0] + 2 * PADDING
    cell_h = THUMB_SIZE[1] + LABEL_H + 3 * PADDING

    header_h = 60 if not subtitle else 80
    total_w  = cols * cell_w
    total_h  = header_h + grid_rows * cell_h

    canvas = Image.new("RGB", (total_w, total_h), GREY)
    draw   = ImageDraw.Draw(canvas)

    # Header
    draw.rectangle([0, 0, total_w, header_h], fill=HEADER_BG)
    draw.text((12, 10), title,    font=FONT_HDR, fill=WHITE)
    if subtitle:
        draw.text((12, 34), subtitle, font=FONT_MED, fill=(180, 200, 240))

    # Cells
    for idx, row in enumerate(rows):
        col = idx % cols
        r   = idx // cols
        x   = col * cell_w
        y   = header_h + r * cell_h
        cell = make_cell(row)
        canvas.paste(cell, (x, y))

    return canvas


def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    # Read CSV
    with open(CSV_PATH, newline="") as f:
        all_rows = list(csv.DictReader(f))

    print(f"Loaded {len(all_rows)} predictions from {CSV_PATH}\n")

    # Group by dish
    by_dish = defaultdict(list)
    for row in all_rows:
        parts = Path(row["file"]).parts
        dish  = parts[-2]   # e.g. "Bratwurst"
        by_dish[dish].append(row)

    # ── 1. Per-dish grids ────────────────────────────────────────────────────
    dish_dir = os.path.join(OUT_DIR, "per_dish")
    os.makedirs(dish_dir, exist_ok=True)

    for dish, rows in sorted(by_dish.items()):
        correct   = sum(1 for r in rows if r["correct"] == "True")
        acc       = correct / len(rows) * 100
        label     = rows[0]["true_label"]
        subtitle  = f"Class: {label}  |  Accuracy: {acc:.1f}%  ({correct}/{len(rows)})"
        grid = make_grid(rows, dish, subtitle)
        if grid:
            out = os.path.join(dish_dir, f"{dish.replace(' ', '_')}.png")
            grid.save(out)
            print(f"  ✅ {dish:<35} {acc:5.1f}%  → {os.path.basename(out)}")

    # ── 2. Wrong predictions grid ────────────────────────────────────────────
    wrong = [r for r in all_rows if r["correct"] == "False"]
    print(f"\nWrong predictions: {len(wrong)}")
    grid_wrong = make_grid(
        wrong,
        f"❌ Incorrect Predictions  ({len(wrong)} images)",
        f"Total: {len(all_rows)}  |  Wrong: {len(wrong)}  ({len(wrong)/len(all_rows)*100:.1f}%)"
    )
    if grid_wrong:
        p = os.path.join(OUT_DIR, "wrong_predictions.png")
        grid_wrong.save(p)
        print(f"  Saved → wrong_predictions.png")

    # ── 3. Uncertain predictions grid ────────────────────────────────────────
    uncertain = [r for r in all_rows if r["uncertain"] == "True"]
    print(f"Uncertain predictions: {len(uncertain)}")
    grid_unc = make_grid(
        uncertain,
        f"⚠️  Uncertain Predictions  ({len(uncertain)} images — confidence < 65%)",
        f"Correct among uncertain: {sum(1 for r in uncertain if r['correct']=='True')}/{len(uncertain)}"
    )
    if grid_unc:
        p = os.path.join(OUT_DIR, "uncertain_predictions.png")
        grid_unc.save(p)
        print(f"  Saved → uncertain_predictions.png")

    # ── 4. Summary overview (one representative per dish) ────────────────────
    # Pick the highest-confidence correct image per dish
    summary_rows = []
    for dish, rows in sorted(by_dish.items()):
        correct_rows = [r for r in rows if r["correct"] == "True"]
        if correct_rows:
            best = max(correct_rows, key=lambda r: float(r["confidence"]))
        else:
            best = rows[0]
        summary_rows.append(best)

    total      = len(all_rows)
    correct_n  = sum(1 for r in all_rows if r["correct"] == "True")
    grid_sum = make_grid(
        summary_rows,
        "🇩🇪 German Dishes — Model Overview (best image per dish)",
        f"Overall Accuracy: {correct_n/total*100:.1f}%  |  {correct_n}/{total} correct  |  MobileNetV3 + YOLOv8-seg"
    )
    if grid_sum:
        p = os.path.join(OUT_DIR, "summary_overview.png")
        grid_sum.save(p)
        print(f"\n  ✅ Summary overview → summary_overview.png")

    print(f"\n🎉 All prediction images saved to: {OUT_DIR}/")
    print(f"   ├── per_dish/            ({len(by_dish)} dish grids)")
    print(f"   ├── wrong_predictions.png")
    print(f"   ├── uncertain_predictions.png")
    print(f"   └── summary_overview.png")


if __name__ == "__main__":
    main()
