"""
generate_walkthrough_book.py
DishDetect SA — Complete Walkthrough Book
Generates flowcharts + professional Word document.
"""

import os, sys
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch
import numpy as np
from datetime import datetime
from docx import Document
from docx.shared import Pt, RGBColor, Cm, Inches
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.enum.table import WD_TABLE_ALIGNMENT
from docx.oxml.ns import qn
from docx.oxml import OxmlElement

ROOT   = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
DOCS   = os.path.join(ROOT, "docs")
CHARTS = os.path.join(DOCS, "_charts")
os.makedirs(CHARTS, exist_ok=True)
OUTPUT = os.path.join(DOCS, "DishDetect_SA_Walkthrough_Book.docx")

def rp(*parts): return os.path.join(ROOT, *parts)
def c(name):   return os.path.join(CHARTS, name)

# ══════════════════════════════════════════════════════════════════════════════
# COLOUR PALETTE
# ══════════════════════════════════════════════════════════════════════════════
BG      = "#0D1B2A"
NAVY    = "#1F3864"
BLUE    = "#1A6EBD"
TEAL    = "#00B4D8"
GREEN   = "#2ECC71"
ORANGE  = "#F5A623"
RED     = "#E94560"
PURPLE  = "#7B2D8B"
WHITE   = "#FFFFFF"
SILVER  = "#B0BEC5"
DARK    = "#0A1628"

# ══════════════════════════════════════════════════════════════════════════════
# CHART HELPERS
# ══════════════════════════════════════════════════════════════════════════════
def dark_fig(w=12, h=8):
    fig, ax = plt.subplots(figsize=(w, h))
    fig.patch.set_facecolor(BG)
    ax.set_facecolor(BG)
    ax.spines["bottom"].set_color(SILVER)
    ax.spines["left"].set_color(SILVER)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(colors=SILVER, labelsize=10)
    ax.xaxis.label.set_color(SILVER)
    ax.yaxis.label.set_color(SILVER)
    return fig, ax

def save(fig, name):
    path = c(name)
    fig.savefig(path, dpi=160, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"  chart → {name}")
    return path

def box(ax, x, y, w, h, color, label, sublabel="", radius=0.3):
    rect = FancyBboxPatch((x - w/2, y - h/2), w, h,
                          boxstyle=f"round,pad={radius}", linewidth=2,
                          edgecolor=color, facecolor=color + "28", zorder=3)
    ax.add_patch(rect)
    ax.text(x, y + 0.15, label, ha="center", va="center",
            fontsize=9.5, fontweight="bold", color=WHITE, zorder=4)
    if sublabel:
        ax.text(x, y - 0.22, sublabel, ha="center", va="center",
                fontsize=7.5, color=color, zorder=4, style="italic")

def arrow(ax, x1, y1, x2, y2, color=ORANGE, label=""):
    ax.annotate("", xy=(x2, y2), xytext=(x1, y1),
                arrowprops=dict(arrowstyle="-|>", color=color, lw=2, mutation_scale=16), zorder=5)
    if label:
        mx, my = (x1+x2)/2, (y1+y2)/2
        ax.text(mx + 0.05, my, label, fontsize=7, color=color, style="italic", zorder=6)

# ══════════════════════════════════════════════════════════════════════════════
# CHART 1 — Pipeline Flowchart
# ══════════════════════════════════════════════════════════════════════════════
def chart_pipeline():
    fig, ax = plt.subplots(figsize=(14, 18))
    fig.patch.set_facecolor(DARK)
    ax.set_facecolor(DARK)
    ax.set_xlim(0, 10); ax.set_ylim(0, 20); ax.axis("off")

    ax.text(5, 19.3, "DishDetect SA", ha="center", fontsize=22,
            fontweight="bold", color=WHITE)
    ax.text(5, 18.7, "Multi-Stage Food Classification Pipeline",
            ha="center", fontsize=12, color=SILVER)
    ax.plot([0.5, 9.5], [18.4, 18.4], color=RED, lw=1.5, alpha=0.6)

    stages = [
        (5, 16.8, TEAL,   "[01]  Stage 1",  "Image Data Collection",    "Requests / Selenium / BeautifulSoup"),
        (5, 13.8, BLUE,   "[02]  Stage 2",  "Zero-Shot Food Filter",    "OpenAI API"),
        (5, 10.8, PURPLE, "[03]  Stage 3",  "Image Pre-Processing",     "Pillow (PIL)"),
        (5,  7.8, RED,    "[04]  Stage 4",  "CLIP Classification",      "CLIP ViT-B/32"),
        (5,  4.8, ORANGE, "[05]  Stage 5",  "Object Detection & Crop",  "YOLOv8n-seg"),
        (5,  1.8, GREEN,  "[06]  Stage 6",  "Final Classification",     "MobileNetV3-Large"),
    ]
    labels = ["filter non-food", "normalise", "Veg / Non-Veg", "crop food region", "final label"]

    for i, (x, y, col, num, title, tool) in enumerate(stages):
        shadow = FancyBboxPatch((x-3.6+0.06, y-1.0-0.06), 7.2, 1.8,
                                boxstyle="round,pad=0.2", linewidth=0,
                                facecolor="#000", alpha=0.3, zorder=2)
        ax.add_patch(shadow)
        main_box = FancyBboxPatch((x-3.6, y-1.0), 7.2, 1.8,
                                  boxstyle="round,pad=0.2", linewidth=2.5,
                                  edgecolor=col, facecolor=col+"22", zorder=3)
        ax.add_patch(main_box)
        bar = FancyBboxPatch((x-3.6, y-1.0), 0.45, 1.8,
                             boxstyle="round,pad=0.0", linewidth=0,
                             facecolor=col, zorder=4)
        ax.add_patch(bar)
        ax.text(x-3.37, y, num, ha="center", va="center",
                fontsize=8.5, fontweight="bold", color=WHITE, zorder=5)
        ax.text(x-0.6, y+0.32, title, ha="left", va="center",
                fontsize=12, fontweight="bold", color=WHITE, zorder=5)
        ax.text(x-0.6, y-0.22, tool, ha="left", va="center",
                fontsize=9, color=col, fontweight="bold", zorder=5)

        if i < len(stages) - 1:
            ax.annotate("", xy=(5, stages[i+1][1]+1.02),
                        xytext=(5, y-1.02),
                        arrowprops=dict(arrowstyle="-|>", color=ORANGE, lw=2.2,
                                       mutation_scale=18), zorder=6)
            mid_y = (y - 1.02 + stages[i+1][1] + 1.02) / 2
            ax.text(5.15, mid_y, labels[i], fontsize=8, color=ORANGE,
                    style="italic", zorder=7)

    # Output badge
    out = FancyBboxPatch((2.2, 0.0), 5.6, 0.85,
                         boxstyle="round,pad=0.15", linewidth=2,
                         edgecolor=GREEN, facecolor=GREEN+"33", zorder=3)
    ax.add_patch(out)
    ax.text(5, 0.56, "OUTPUT", ha="center", fontsize=11,
            fontweight="bold", color=GREEN, zorder=4)
    ax.text(5, 0.22, "Vegetarian  /  Non-Vegetarian  +  Confidence Score",
            ha="center", fontsize=8.5, color=WHITE, zorder=4)
    ax.annotate("", xy=(5, 0.87), xytext=(5, 0.73),
                arrowprops=dict(arrowstyle="-|>", color=GREEN, lw=2, mutation_scale=16), zorder=6)

    ax.text(5, -0.25, f"DishDetect SA  •  {datetime.now().strftime('%d %B %Y')}",
            ha="center", fontsize=7.5, color=SILVER, alpha=0.6)
    return save(fig, "01_pipeline.png")

# ══════════════════════════════════════════════════════════════════════════════
# CHART 2 — Accuracy Evolution Bar Chart
# ══════════════════════════════════════════════════════════════════════════════
def chart_accuracy_evolution():
    fig, ax = dark_fig(13, 6)
    phases  = ["Ph.2\nEfficientNet\nNo YOLO", "Ph.2B\nMobileNet\n+YOLO",
               "Ph.3\nGerman\nDomain", "Ph.4\nRestaurant\n(Phase 4)",
               "Ph.7\nBinary\nRefined", "Ph.8\nShawarma\nBoost",
               "Ph.9\nV6\nRebalance", "Ph.10\nArch.\nCompare"]
    std_acc = [89.3, 92.7, 95.5, None, 89.4, 91.2, 92.0, 92.0]
    rst_acc = [None, None, None, 77.4, None, 92.7, 90.9, 90.9]

    x = np.arange(len(phases))
    w = 0.36

    b1 = ax.bar(x - w/2, [v if v else 0 for v in std_acc], w,
                label="Standard Test Accuracy", color=TEAL, alpha=0.85, zorder=3)
    b2 = ax.bar(x + w/2, [v if v else 0 for v in rst_acc], w,
                label="Restaurant Accuracy", color=ORANGE, alpha=0.85, zorder=3)

    for bar, val in zip(b1, std_acc):
        if val:
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.3,
                    f"{val}%", ha="center", va="bottom", fontsize=8,
                    color=TEAL, fontweight="bold")
    for bar, val in zip(b2, rst_acc):
        if val:
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.3,
                    f"{val}%", ha="center", va="bottom", fontsize=8,
                    color=ORANGE, fontweight="bold")

    ax.axhline(90, color=RED, lw=1.5, linestyle="--", alpha=0.6, label="90% threshold")
    ax.set_xticks(x); ax.set_xticklabels(phases, color=SILVER, fontsize=8)
    ax.set_ylim(70, 100); ax.set_ylabel("Accuracy (%)", color=SILVER, fontsize=11)
    ax.set_title("Accuracy Evolution Across All Phases", color=WHITE,
                 fontsize=14, fontweight="bold", pad=15)
    ax.legend(facecolor=BG, edgecolor=SILVER, labelcolor=WHITE, fontsize=9)
    ax.grid(axis="y", color=SILVER, alpha=0.15, linestyle="--")
    ax.set_facecolor(BG)
    return save(fig, "02_accuracy_evolution.png")

# ══════════════════════════════════════════════════════════════════════════════
# CHART 3 — Restaurant Per-Dish Accuracy
# ══════════════════════════════════════════════════════════════════════════════
def chart_restaurant_dishes():
    fig, ax = dark_fig(12, 6)
    dishes   = ["Okra\nTagine", "Za'atar\nManaqish", "Shawarma", "Shish\nTawook",
                "Cheese\nFatayer", "Hummus", "Mandi", "Meat\nMadfoon",
                "Mixed\nGrill", "Tamees\nBread"]
    accuracy = [50.0, 80.0, 85.7, 85.7, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0]
    conf     = [0.730, 0.775, 0.742, 0.850, 0.775, 0.864, 0.866, 0.965, 0.930, 0.915]
    colors   = [RED if a < 90 else ORANGE if a < 95 else GREEN for a in accuracy]

    x = np.arange(len(dishes))
    bars = ax.bar(x, accuracy, color=colors, alpha=0.85, zorder=3, width=0.65)
    for bar, acc, cf in zip(bars, accuracy, conf):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.8,
                f"{acc}%\n({cf:.2f})", ha="center", va="bottom",
                fontsize=8, color=WHITE, fontweight="bold")

    ax.axhline(90, color=WHITE, lw=1.2, linestyle="--", alpha=0.4, label="90% line")
    ax.set_xticks(x); ax.set_xticklabels(dishes, color=SILVER, fontsize=9)
    ax.set_ylim(0, 118); ax.set_ylabel("Accuracy (%)", color=SILVER, fontsize=11)
    ax.set_title("Restaurant Test — Per-Dish Accuracy & Avg Confidence",
                 color=WHITE, fontsize=13, fontweight="bold", pad=12)
    ax.legend(facecolor=BG, edgecolor=SILVER, labelcolor=WHITE, fontsize=9)
    ax.grid(axis="y", color=SILVER, alpha=0.15, linestyle="--")

    red_p   = mpatches.Patch(color=RED,    label="Below 90% (Problem)")
    org_p   = mpatches.Patch(color=ORANGE, label="85–90% (Good)")
    grn_p   = mpatches.Patch(color=GREEN,  label="100% (Perfect)")
    ax.legend(handles=[red_p, org_p, grn_p], facecolor=BG, edgecolor=SILVER,
              labelcolor=WHITE, fontsize=9, loc="lower right")
    return save(fig, "03_restaurant_dishes.png")

# ══════════════════════════════════════════════════════════════════════════════
# CHART 4 — Architecture Comparison Radar
# ══════════════════════════════════════════════════════════════════════════════
def chart_arch_comparison():
    cats   = ["Standard\nTest Acc.", "Restaurant\nAcc.", "Shawarma\nAcc.",
              "Okra Tagine\nAcc.", "Speed\n(Relative)", "Parameters\n(Inverse)"]
    mob    = [92.0, 90.9, 71.4, 75.0, 90.0, 85.0]   # higher=better for all
    eff    = [84.0, 90.9, 85.7, 50.0, 55.0, 50.0]

    N = len(cats)
    angles = [n / float(N) * 2 * np.pi for n in range(N)]
    angles += angles[:1]
    mob_v  = [v/100 for v in mob] + [mob[0]/100]
    eff_v  = [v/100 for v in eff] + [eff[0]/100]

    fig, ax = plt.subplots(figsize=(9, 9), subplot_kw=dict(polar=True))
    fig.patch.set_facecolor(BG)
    ax.set_facecolor(BG)
    ax.plot(angles, mob_v, "o-", linewidth=2.5, color=TEAL, label="MobileNetV3-Large")
    ax.fill(angles, mob_v, alpha=0.18, color=TEAL)
    ax.plot(angles, eff_v, "s--", linewidth=2.5, color=ORANGE, label="EfficientNet-B2 (Stage 1)")
    ax.fill(angles, eff_v, alpha=0.18, color=ORANGE)

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(cats, color=WHITE, fontsize=10)
    ax.set_yticklabels([]); ax.spines["polar"].set_color(SILVER)
    ax.grid(color=SILVER, alpha=0.25)
    ax.set_title("Architecture Comparison: MobileNetV3 vs EfficientNet-B2",
                 color=WHITE, fontsize=12, fontweight="bold", pad=20)
    ax.legend(loc="upper right", bbox_to_anchor=(1.3, 1.1),
              facecolor=BG, edgecolor=SILVER, labelcolor=WHITE, fontsize=10)
    return save(fig, "04_arch_radar.png")

# ══════════════════════════════════════════════════════════════════════════════
# CHART 5 — Dataset Growth Timeline
# ══════════════════════════════════════════════════════════════════════════════
def chart_dataset_timeline():
    fig, ax = dark_fig(13, 6)
    versions = ["V1\nBaseline", "V2\nBinary", "V3\nYOLO", "V4\nGerman",
                "V5\nShawarma+", "V6\nRebalanced"]
    totals   = [1439, 1439, 1439, 2191, 1319, 1432]
    non_veg  = [758,  758,  758,  1172, 615,  728]
    veg      = [681,  681,  681,  1019, 704,  704]

    x = np.arange(len(versions))
    ax.fill_between(x, 0, totals, alpha=0.15, color=TEAL)
    ax.plot(x, totals, "o-", color=TEAL, lw=2.5, ms=8, label="Total Images", zorder=4)
    ax.fill_between(x, 0, non_veg, alpha=0.12, color=RED)
    ax.plot(x, non_veg, "s--", color=RED, lw=2, ms=7, label="Non-Veg", zorder=4)
    ax.fill_between(x, 0, veg, alpha=0.12, color=GREEN)
    ax.plot(x, veg, "^--", color=GREEN, lw=2, ms=7, label="Veg", zorder=4)

    for i, (t, nv, v) in enumerate(zip(totals, non_veg, veg)):
        ax.text(i, t + 35, str(t), ha="center", fontsize=9, color=TEAL, fontweight="bold")

    ax.set_xticks(x); ax.set_xticklabels(versions, color=SILVER, fontsize=10)
    ax.set_ylabel("Number of Images", color=SILVER, fontsize=11)
    ax.set_title("Dataset Evolution Across Versions", color=WHITE,
                 fontsize=14, fontweight="bold", pad=15)
    ax.legend(facecolor=BG, edgecolor=SILVER, labelcolor=WHITE, fontsize=10)
    ax.grid(color=SILVER, alpha=0.15, linestyle="--")
    ax.set_ylim(0, 2700)
    return save(fig, "05_dataset_timeline.png")

# ══════════════════════════════════════════════════════════════════════════════
# CHART 6 — Two-Stage Training Flowchart
# ══════════════════════════════════════════════════════════════════════════════
def chart_training_flow():
    fig, ax = plt.subplots(figsize=(13, 8))
    fig.patch.set_facecolor(DARK); ax.set_facecolor(DARK)
    ax.set_xlim(0, 13); ax.set_ylim(0, 9); ax.axis("off")
    ax.set_title("Two-Stage Transfer Learning Strategy",
                 color=WHITE, fontsize=14, fontweight="bold", pad=10)

    def rbox(ax, x, y, w, h, col, top, bot=""):
        rect = FancyBboxPatch((x-w/2, y-h/2), w, h,
                              boxstyle="round,pad=0.2", linewidth=2,
                              edgecolor=col, facecolor=col+"22", zorder=3)
        ax.add_patch(rect)
        ax.text(x, y + (0.2 if bot else 0), top, ha="center", va="center",
                fontsize=9, fontweight="bold", color=WHITE, zorder=4)
        if bot:
            ax.text(x, y - 0.25, bot, ha="center", va="center",
                    fontsize=8, color=col, zorder=4)

    def arr(ax, x1, y1, x2, y2, col=ORANGE, lbl=""):
        ax.annotate("", xy=(x2, y2), xytext=(x1, y1),
                    arrowprops=dict(arrowstyle="-|>", color=col, lw=2, mutation_scale=16), zorder=5)
        if lbl:
            ax.text((x1+x2)/2 + 0.1, (y1+y2)/2, lbl, fontsize=7.5, color=col, style="italic")

    # ImageNet weights
    rbox(ax, 2.2, 7.5, 3.2, 0.9, TEAL, "ImageNet Pre-trained", "MobileNetV3-Large")
    # Stage 1
    rbox(ax, 6.5, 7.5, 3.2, 0.9, ORANGE, "Stage 1: Frozen Backbone", "LR=1e-3 · 15 epochs max · Head only")
    arr(ax, 3.82, 7.5, 4.9, 7.5, ORANGE, "freeze backbone")
    # Best ckpt
    rbox(ax, 10.5, 7.5, 2.2, 0.9, SILVER, "Best Checkpoint", "saved to checkpoints/")
    arr(ax, 8.15, 7.5, 9.4, 7.5, SILVER, "early stop")

    # Down
    arr(ax, 6.5, 7.0, 6.5, 6.0, BLUE, "load best")

    # Stage 2
    rbox(ax, 6.5, 5.5, 3.2, 0.9, BLUE, "Stage 2: Full Fine-Tuning", "LR=1e-5 · Cosine Anneal · All layers")
    rbox(ax, 2.2, 5.5, 3.2, 0.9, PURPLE, "Loss Function", "CrossEntropy + LabelSmoothing(0.1)")
    rbox(ax, 10.5, 5.5, 2.2, 0.9, PURPLE, "Optimizer", "AdamW + Weight Decay")

    arr(ax, 3.82, 5.5, 4.9, 5.5, PURPLE)
    arr(ax, 8.15, 5.5, 9.4, 5.5, PURPLE)
    arr(ax, 6.5, 5.0, 6.5, 4.0, GREEN, "early stop")

    # Final output
    rbox(ax, 6.5, 3.5, 3.2, 0.9, GREEN, "Final Model", "Best val-acc checkpoint saved")

    # Metrics boxes
    for i, (label, val, col) in enumerate([
        ("Test Accuracy", "92.70%",   TEAL),
        ("Precision",     "0.927",    BLUE),
        ("Recall",        "0.927",    BLUE),
        ("F1-Score",      "0.927",    GREEN),
    ]):
        bx = 1.5 + i * 2.8
        rbox(ax, bx, 1.8, 2.3, 0.75, col, label, val)

    arr(ax, 6.5, 3.05, 6.5, 2.2, WHITE)
    ax.text(6.5, 2.22, "evaluate →", ha="center", fontsize=7.5, color=SILVER)

    ax.text(6.5, 0.6, "Seed=42  ·  Batch=32  ·  Image=344×344  ·  Split=70/15/15",
            ha="center", fontsize=8.5, color=SILVER)

    return save(fig, "06_training_flow.png")

# ══════════════════════════════════════════════════════════════════════════════
# CHART 7 — YOLO Pipeline Diagram
# ══════════════════════════════════════════════════════════════════════════════
def chart_yolo_pipeline():
    fig, ax = plt.subplots(figsize=(14, 5))
    fig.patch.set_facecolor(DARK); ax.set_facecolor(DARK)
    ax.set_xlim(0, 14); ax.set_ylim(0, 6); ax.axis("off")
    ax.set_title("YOLO Segmentation Pre-processing Pipeline",
                 color=WHITE, fontsize=14, fontweight="bold", pad=10)

    steps = [
        (1.5, 3.0, RED,    "Raw Image",     "Restaurant photo\nwith background"),
        (4.5, 3.0, ORANGE, "YOLOv8n-seg",   "Detect food object\nGenerate seg mask"),
        (7.5, 3.0, TEAL,   "Mask Applied",  "Pixels outside mask\nset to black"),
        (10.5,3.0, BLUE,   "Cropped Food",  "344×344 resize\nClean food region only"),
        (13.0,3.0, GREEN,  "Classifier",    "MobileNetV3-Large\nVeg / Non-Veg"),
    ]
    for x, y, col, top, bot in steps:
        r = FancyBboxPatch((x-1.1, y-1.0), 2.2, 2.0,
                           boxstyle="round,pad=0.2", linewidth=2,
                           edgecolor=col, facecolor=col+"25", zorder=3)
        ax.add_patch(r)
        ax.text(x, y+0.32, top, ha="center", fontsize=9.5,
                fontweight="bold", color=WHITE, zorder=4)
        ax.text(x, y-0.32, bot, ha="center", fontsize=7.5,
                color=col, zorder=4, linespacing=1.4)

    for i in range(len(steps)-1):
        x1, x2 = steps[i][0]+1.12, steps[i+1][0]-1.12
        ax.annotate("", xy=(x2, 3.0), xytext=(x1, 3.0),
                    arrowprops=dict(arrowstyle="-|>", color=ORANGE,
                                   lw=2, mutation_scale=16), zorder=5)

    ax.text(7.0, 0.6,
            "+3.4 percentage points gained — 89.3% → 92.7% by removing background noise",
            ha="center", fontsize=9, color=GREEN, fontweight="bold")
    ax.text(7.0, 0.25, "Single biggest accuracy improvement in the entire project",
            ha="center", fontsize=8, color=SILVER)
    return save(fig, "07_yolo_pipeline.png")

# ══════════════════════════════════════════════════════════════════════════════
# CHART 8 — Known Failure Heatmap
# ══════════════════════════════════════════════════════════════════════════════
def chart_failure_heatmap():
    fig, ax = dark_fig(11, 5)
    dishes   = ["Shawarma", "Okra Tagine", "Za'atar\nManaqish", "Shish\nTawook",
                "Mandi", "Hummus", "Mixed\nGrill", "Tamees\nBread"]
    factors  = ["Visual\nAmbiguity", "YOLO\nCrop Issue", "Data\nVolume",
                "Domain\nGap", "Label\nNoise"]
    data = np.array([
        [5, 2, 5, 3, 1],   # Shawarma
        [4, 1, 3, 2, 5],   # Okra Tagine
        [3, 1, 3, 2, 2],   # Za'atar
        [2, 1, 3, 1, 1],   # Shish Tawook
        [5, 5, 3, 2, 1],   # Mandi
        [1, 1, 2, 1, 1],   # Hummus
        [1, 1, 2, 1, 1],   # Mixed Grill
        [1, 1, 2, 1, 1],   # Tamees Bread
    ])
    import matplotlib
    cmap = matplotlib.colormaps["RdYlGn_r"]
    im = ax.imshow(data, cmap=cmap, aspect="auto", vmin=1, vmax=5)
    for i in range(len(dishes)):
        for j in range(len(factors)):
            ax.text(j, i, ["", "Low", "Med-Low", "Medium", "Med-High", "High"][data[i,j]],
                    ha="center", va="center", fontsize=8.5,
                    color=WHITE if data[i,j] >= 4 else DARK, fontweight="bold")

    ax.set_xticks(range(len(factors))); ax.set_xticklabels(factors, color=SILVER, fontsize=9)
    ax.set_yticks(range(len(dishes))); ax.set_yticklabels(dishes, color=SILVER, fontsize=9)
    ax.set_title("Failure Factor Heatmap — Risk by Dish & Issue Type",
                 color=WHITE, fontsize=13, fontweight="bold", pad=12)
    cb = fig.colorbar(im, ax=ax, orientation="vertical", shrink=0.8)
    cb.set_label("Risk Level", color=SILVER)
    cb.ax.yaxis.set_tick_params(color=SILVER)
    plt.setp(cb.ax.yaxis.get_ticklabels(), color=SILVER)
    fig.tight_layout()
    return save(fig, "08_failure_heatmap.png")

# ══════════════════════════════════════════════════════════════════════════════
# CHART 9 — Vegan vs Vegetarian Intelligence Overview
# ══════════════════════════════════════════════════════════════════════════════
def chart_vegan_intelligence():
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.5))
    fig.patch.set_facecolor(BG)
    ax1.set_facecolor(BG)
    ax2.set_facecolor(BG)

    for ax in [ax1, ax2]:
        ax.spines["bottom"].set_color(SILVER)
        ax.spines["left"].set_color(SILVER)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.tick_params(colors=SILVER, labelsize=9)
        ax.xaxis.label.set_color(SILVER)
        ax.yaxis.label.set_color(SILVER)

    # Panel 1: Dataset Composition (Meat-Free Cohort)
    cats   = ["Vegan\n(100% Plant)", "Vegetarian\n(Dairy/Eggs/Ghee)", "Meat Dishes\n(Excluded)"]
    counts = [411, 293, 0]
    colors = [GREEN, ORANGE, RED]
    bars = ax1.bar(cats, counts, color=colors, width=0.55, edgecolor=WHITE, lw=1)
    for b, c, n in zip(bars, counts, ["9 dishes", "6 dishes", "10 dishes"]):
        h = b.get_height()
        if h > 0:
            ax1.text(b.get_x() + b.get_width()/2, h + 10, f"{h}\n({n})",
                     ha="center", va="bottom", color=WHITE, fontweight="bold", fontsize=9)
        else:
            ax1.text(b.get_x() + b.get_width()/2, 10, f"0\n({n})",
                     ha="center", va="bottom", color=RED, fontweight="bold", fontsize=9)
    ax1.set_ylim(0, 480)
    ax1.set_title("Meat-Free Cohort: 704 Images across 15 Dishes",
                  color=WHITE, fontsize=11.5, fontweight="bold", pad=10)
    ax1.set_ylabel("Image Count", color=SILVER)

    # Panel 2: Benchmark Performance Across Splits
    bench_names = ["Validation\nAccuracy", "Held-Out\nTest Acc.", "Restaurant\nTest Acc.", "Vegan\nPrecision", "Vegetarian\nRecall"]
    bench_vals  = [88.79, 87.85, 83.33, 94.55, 93.33]
    bcolors     = [TEAL, BLUE, PURPLE, GREEN, ORANGE]
    bars2 = ax2.bar(bench_names, bench_vals, color=bcolors, width=0.55, edgecolor=WHITE, lw=1)
    for b, v in zip(bars2, bench_vals):
        ax2.text(b.get_x() + b.get_width()/2, v + 1.2, f"{v:.1f}%",
                 ha="center", va="bottom", color=WHITE, fontweight="bold", fontsize=9)
    ax2.set_ylim(0, 105)
    ax2.axhline(80, color=SILVER, linestyle="--", alpha=0.4, lw=1)
    ax2.set_title("Performance Benchmarks (MobileNetV3-Large)",
                  color=WHITE, fontsize=11.5, fontweight="bold", pad=10)
    ax2.set_ylabel("Score (%)", color=SILVER)

    fig.suptitle("Vegan vs. Vegetarian Classification Benchmark Overview",
                 color=WHITE, fontsize=14, fontweight="bold", y=1.02)
    fig.tight_layout()
    return save(fig, "09_vegan_intelligence.png")

# ══════════════════════════════════════════════════════════════════════════════
# WORD DOC HELPERS
# ══════════════════════════════════════════════════════════════════════════════
def set_cell_bg(cell, hex_color):
    tc = cell._tc; tcPr = tc.get_or_add_tcPr()
    shd = OxmlElement("w:shd")
    shd.set(qn("w:val"), "clear"); shd.set(qn("w:color"), "auto")
    shd.set(qn("w:fill"), hex_color.upper()); tcPr.append(shd)

def whead(doc, text, level=1, color="1F3864"):
    h = doc.add_heading(text, level=level)
    h.alignment = WD_ALIGN_PARAGRAPH.LEFT
    for r in h.runs: r.font.color.rgb = RGBColor.from_string(color)
    return h

def wbody(doc, text, italic=False, bold=False, size=11, color=None):
    par = doc.add_paragraph()
    r = par.add_run(text)
    r.italic = italic; r.bold = bold; r.font.size = Pt(size)
    if color: r.font.color.rgb = RGBColor.from_string(color)
    return par

def wbullet(doc, label, value="", color="1F3864"):
    par = doc.add_paragraph(style="List Bullet")
    r = par.add_run(label); r.bold = True
    r.font.color.rgb = RGBColor.from_string(color)
    if value: par.add_run("  " + value)
    return par

def wcode(doc, text):
    par = doc.add_paragraph()
    r = par.add_run(text)
    r.font.name = "Courier New"; r.font.size = Pt(9)
    r.font.color.rgb = RGBColor.from_string("C7254E")
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
        wbody(doc, f"[Missing: {os.path.basename(path)}]", italic=True)

def wtable(doc, headers, rows, hcol="1F3864", alt1="F2F6FC", alt2="FFFFFF"):
    tbl = doc.add_table(rows=1, cols=len(headers))
    tbl.style = "Table Grid"; tbl.alignment = WD_TABLE_ALIGNMENT.CENTER
    hdr = tbl.rows[0].cells
    for i, h in enumerate(headers):
        hdr[i].text = h; set_cell_bg(hdr[i], hcol)
        for r in hdr[i].paragraphs[0].runs:
            r.font.color.rgb = RGBColor.from_string("FFFFFF"); r.bold = True
    for idx, row in enumerate(rows):
        rc = tbl.add_row().cells
        for i, v in enumerate(row):
            rc[i].text = str(v)
            set_cell_bg(rc[i], alt1 if idx % 2 == 0 else alt2)
    return tbl

# ══════════════════════════════════════════════════════════════════════════════
# BUILD THE BOOK
# ══════════════════════════════════════════════════════════════════════════════
def build_book(charts):
    doc = Document()
    for sec in doc.sections:
        sec.top_margin = Cm(1.8); sec.bottom_margin = Cm(1.8)
        sec.left_margin = Cm(2.2); sec.right_margin = Cm(2.2)

    # ── COVER PAGE ────────────────────────────────────────────────────────────
    cover = doc.add_paragraph(); cover.alignment = WD_ALIGN_PARAGRAPH.CENTER
    r = cover.add_run("\n\n\n🍽️  DishDetect SA")
    r.bold = True; r.font.size = Pt(32); r.font.color.rgb = RGBColor.from_string("1F3864")

    sub = doc.add_paragraph(); sub.alignment = WD_ALIGN_PARAGRAPH.CENTER
    r2 = sub.add_run("Veg / Non-Veg Food Intelligence")
    r2.font.size = Pt(18); r2.font.color.rgb = RGBColor.from_string("2E75B6")

    doc.add_paragraph()
    tag = doc.add_paragraph(); tag.alignment = WD_ALIGN_PARAGRAPH.CENTER
    r3 = tag.add_run("Complete Developer & Research Walkthrough Book")
    r3.bold = True; r3.font.size = Pt(14); r3.font.color.rgb = RGBColor.from_string("595959")

    doc.add_paragraph()
    meta = doc.add_paragraph(); meta.alignment = WD_ALIGN_PARAGRAPH.CENTER
    r4 = meta.add_run(
        f"Saudi Arabia  •  Arabic Food Classification  •  PyTorch  •  YOLOv8\n"
        f"github.com/Mehdidar/DishDetect-SA\n\n"
        f"Generated: {datetime.now().strftime('%d %B %Y')}"
    )
    r4.font.size = Pt(11); r4.italic = True
    r4.font.color.rgb = RGBColor.from_string("767676")
    doc.add_page_break()

    # ── TABLE OF CONTENTS ─────────────────────────────────────────────────────
    whead(doc, "Table of Contents")
    toc_items = [
        ("Chapter 1", "Project Overview & Goals"),
        ("Chapter 2", "The Six-Stage Pipeline"),
        ("Chapter 3", "Dataset — Collection, Cleaning & Evolution"),
        ("Chapter 4", "Model Architecture & Training Strategy"),
        ("Chapter 5", "Phase-by-Phase Development History"),
        ("Chapter 6", "Results & Performance Metrics"),
        ("Chapter 7", "Real-World Restaurant Testing"),
        ("Chapter 8", "Architecture Comparison: MobileNetV3 vs EfficientNet-B2"),
        ("Chapter 9", "Known Flaws & Failure Analysis"),
        ("Chapter 10","Lessons Learned"),
        ("Chapter 11","How to Run the Project"),
        ("Chapter 12","Next Steps & Roadmap"),
        ("Chapter 13","Vegan vs. Vegetarian (Meat-Free) Intelligence Module"),
    ]
    for ch, title in toc_items:
        p = doc.add_paragraph(style="List Number")
        r = p.add_run(f"{ch}:  "); r.bold = True
        r.font.color.rgb = RGBColor.from_string("1F3864")
        p.add_run(title)
    doc.add_page_break()

    # ══════════════════════════════════════════════════════════════════════════
    # CHAPTER 1 — PROJECT OVERVIEW
    # ══════════════════════════════════════════════════════════════════════════
    whead(doc, "Chapter 1 — Project Overview & Goals", color="1F3864")
    wbody(doc,
        "DishDetect SA is an end-to-end deep learning system designed to automatically "
        "classify food images from Saudi Arabian and Arabic restaurants as Vegetarian (Veg) "
        "or Non-Vegetarian (Non-Veg). The system was built to address a real-world need: "
        "restaurant-goers and food platforms often lack reliable dietary labelling for "
        "traditional Arabic cuisines, especially in the Saudi market."
    )
    doc.add_paragraph()
    wbody(doc, "Core Objectives:", bold=True)
    wbullet(doc, "Accurately classify", "food images as Veg or Non-Veg with >90% restaurant accuracy")
    wbullet(doc, "Handle ambiguous dishes", "such as Shawarma, Mandi, and Okra Tagine")
    wbullet(doc, "Generalize", "across different cuisines (Arabic, German cross-domain test)")
    wbullet(doc, "Build a complete production pipeline", "from data collection to a live GUI app")

    doc.add_paragraph()
    wbody(doc, "Headline Results:", bold=True)
    wtable(doc,
        ["Benchmark", "Accuracy", "Details"],
        [
            ["Standard Test Set",      "92.70%", "534 held-out images — MobileNetV3 + YOLO"],
            ["Restaurant Real-World",  "92.73%", "55 Saudi restaurant images (Phase 8 peak)"],
            ["German Cross-Domain",    "95.53%", "2,191 German restaurant images"],
            ["26-Class Dish Level",    "67.61%", "Exact dish recognition (26 categories)"],
            ["35-Class Dish Level",    "67.01%", "Expanded hierarchical classifier"],
        ]
    )
    doc.add_page_break()

    # ══════════════════════════════════════════════════════════════════════════
    # CHAPTER 2 — PIPELINE
    # ══════════════════════════════════════════════════════════════════════════
    whead(doc, "Chapter 2 — The Six-Stage Pipeline", color="1F3864")
    wbody(doc,
        "The DishDetect SA pipeline is a sequential six-stage system. Each stage applies a "
        "specialized tool to progressively refine raw internet images into a reliable "
        "Veg/Non-Veg classification. The flowchart below illustrates the complete data flow."
    )
    doc.add_paragraph()
    wimg(doc, charts[0], 6.2, "Figure 1 — DishDetect SA: Complete Six-Stage Pipeline Flowchart")
    doc.add_paragraph()

    stages_desc = [
        ("Stage 1 — Image Data Collection", "Requests, Selenium, BeautifulSoup",
         "Food images are automatically scraped from online sources relevant to the Saudi "
         "Arabian market. Requests handles direct HTTP downloads, Selenium automates "
         "JavaScript-heavy platforms, and BeautifulSoup parses HTML to extract image links."),
        ("Stage 2 — Zero-Shot Food Filter", "OpenAI API",
         "Not all scraped images are food. OpenAI's zero-shot classification model verifies "
         "each image is actually food. Non-food images (packaging, advertisements, people) "
         "are discarded, significantly improving dataset quality before any heavy processing."),
        ("Stage 3 — Image Pre-Processing", "Pillow (PIL)",
         "All verified images are resized to a consistent resolution, converted to RGB, "
         "normalised, and lightly augmented (flipping, brightness jitter). Corrupt or "
         "unreadable images are logged and discarded automatically."),
        ("Stage 4 — CLIP Classification", "CLIP ViT-B/32 (OpenAI)",
         "CLIP performs zero-shot Veg/Non-Veg classification using vision-language prompts "
         "('a vegetarian dish' vs 'a non-vegetarian dish with meat'). This provides a fast, "
         "training-free first-pass classification that filters the most obvious errors."),
        ("Stage 5 — Object Detection & Cropping", "YOLOv8n-seg (Ultralytics)",
         "YOLOv8 detects the primary food object and generates a pixel-level segmentation mask. "
         "The food is extracted and the background is removed entirely. This was the single "
         "largest accuracy improvement in the project: +3.4 percentage points."),
        ("Stage 6 — Final Classification", "MobileNetV3-Large (PyTorch)",
         "The clean, cropped food image is passed to a fine-tuned MobileNetV3-Large model "
         "for the final Veg/Non-Veg prediction with a confidence score. The model was trained "
         "using two-stage transfer learning on ~1,400+ Arabic food images."),
    ]
    for title, tool, desc in stages_desc:
        whead(doc, title, level=2, color="2E75B6")
        wbullet(doc, "Tool:", tool, color="2E75B6")
        wbody(doc, desc)
        doc.add_paragraph()

    whead(doc, "YOLO Segmentation Deep Dive", level=2, color="2E75B6")
    wbody(doc,
        "The YOLO segmentation step deserves special attention because it was the single "
        "most impactful change in the entire project. The diagram below shows exactly "
        "what happens when an image passes through YOLOv8n-seg:"
    )
    wimg(doc, charts[6], 6.2, "Figure 2 — YOLO Segmentation Pre-processing Pipeline (+3.4pp accuracy gain)")
    doc.add_page_break()

    # ══════════════════════════════════════════════════════════════════════════
    # CHAPTER 3 — DATASET
    # ══════════════════════════════════════════════════════════════════════════
    whead(doc, "Chapter 3 — Dataset: Collection, Cleaning & Evolution", color="1F3864")
    wbody(doc,
        "The dataset was never static. It was continuously collected, cleaned, and expanded "
        "across 10+ project phases. The chart below shows its growth and composition."
    )
    wimg(doc, charts[4], 6.2, "Figure 3 — Dataset Evolution: Images per version with Veg/Non-Veg split")
    doc.add_paragraph()

    whead(doc, "Dataset Versions", level=2, color="2E75B6")
    wtable(doc,
        ["Version", "Total", "Non-Veg", "Veg", "Key Change"],
        [
            ["V1 Baseline",   "~1,439", "758",   "681",   "Raw Arabic dishes, 25 dish classes"],
            ["V2 Binary",     "~1,439", "758",   "681",   "Re-labelled as Veg / Non-Veg"],
            ["V3 YOLO",       "~1,439", "758",   "681",   "YOLOv8 background removal added"],
            ["V4 German",     "2,191",  "1,172", "1,019", "Cross-domain: 30 German dish types"],
            ["V5 Shawarma+",  "1,319",  "615",   "704",   "Shawarma boosted 5 → 86 images (17×)"],
            ["V6 Rebalanced", "1,432",  "728",   "704",   "Non-veg rebalanced; Shish Tawook +80"],
            ["35-Dish",       "1,306",  "—",     "—",     "35-class hierarchical classifier"],
        ]
    )
    doc.add_paragraph()

    whead(doc, "Data Cleaning Protocol", level=2, color="2E75B6")
    wbullet(doc, "Corrupt & unsupported files:", "1 corrupt image + 8 .avif / .DS_Store files removed")
    wbullet(doc, "Size filtering:", "Images below minimum CNN-usable resolution discarded")
    wbullet(doc, "Duplicate leakage prevention:", "Perceptual hashing (threshold=5) ensures near-duplicates land in the same split")
    wbullet(doc, "Train / Val / Test split:", "70% / 15% / 15% stratified three-way split")
    wbullet(doc, "Image resolution:", "344×344 px (primary) — also tested 288 and 224")
    wbullet(doc, "Class weights:", "Inverse-frequency weights in CrossEntropyLoss to handle imbalance")
    doc.add_page_break()

    # ══════════════════════════════════════════════════════════════════════════
    # CHAPTER 4 — ARCHITECTURE & TRAINING
    # ══════════════════════════════════════════════════════════════════════════
    whead(doc, "Chapter 4 — Model Architecture & Training Strategy", color="1F3864")
    wbody(doc,
        "The classifier is built on MobileNetV3-Large — a lightweight but highly capable "
        "convolutional neural network pre-trained on ImageNet. It was chosen for its "
        "balance of accuracy, speed, and deployability (only ~5M parameters)."
    )
    doc.add_paragraph()
    wimg(doc, charts[5], 6.2, "Figure 4 — Two-Stage Transfer Learning Flow with all hyperparameters")
    doc.add_paragraph()

    whead(doc, "Hyperparameters", level=2, color="2E75B6")
    wtable(doc,
        ["Parameter", "Stage 1 (Frozen)", "Stage 2 (Fine-Tune)"],
        [
            ["Learning Rate",   "1e-3",          "1e-5"],
            ["Max Epochs",      "15",            "25 (early stop)"],
            ["Optimizer",       "AdamW",         "AdamW"],
            ["LR Schedule",     "None",          "Cosine Annealing"],
            ["Backbone",        "FROZEN",        "UNFROZEN (all layers)"],
            ["Batch Size",      "32",            "32"],
            ["Image Size",      "344×344",       "344×344"],
            ["Loss Function",   "CrossEntropy + Label Smoothing (0.1)",
                                "CrossEntropy + Label Smoothing (0.1)"],
            ["Early Stop Pat.", "7 epochs",      "7 epochs"],
            ["Random Seed",     "42",            "42"],
        ]
    )
    doc.add_paragraph()

    whead(doc, "Model Specifications", level=2, color="2E75B6")
    wtable(doc,
        ["Property", "MobileNetV3-Large", "EfficientNet-B2 (comparison)"],
        [
            ["Parameters",     "~5M",       "~9M"],
            ["Backbone",       "MobileNetV3","EfficientNet-B2"],
            ["Pre-training",   "ImageNet",   "ImageNet"],
            ["Fine-tuning",    "Full (2-stage)", "Stage 1 only (CPU limitation)"],
            ["Standard Acc.",  "92.70%",    "84.0%"],
            ["Restaurant Acc.","90.9%",     "90.9%"],
            ["Speed",          "Fast",      "Slow (~2× slower)"],
        ]
    )
    doc.add_page_break()

    # ══════════════════════════════════════════════════════════════════════════
    # CHAPTER 5 — PHASE HISTORY
    # ══════════════════════════════════════════════════════════════════════════
    whead(doc, "Chapter 5 — Phase-by-Phase Development History", color="1F3864")
    wbody(doc,
        "The project evolved through 10 distinct phases, each adding a new technique, "
        "dataset expansion, or evaluation. The chart below summarises accuracy across phases."
    )
    wimg(doc, charts[1], 6.2, "Figure 5 — Accuracy Evolution: Standard Test & Restaurant Accuracy Across All Phases")
    doc.add_paragraph()

    phases = [
        ("Phase 1", "Baseline 25-Class Dish Classifier", "MobileNetV3-Large", "~1,439 (25 classes)", "Qualitative", "N/A",
         "The first model tried to predict exact dish names across 25 Arabic categories. "
         "Visual ambiguity between similar-looking rice and meat dishes caused confusion, "
         "proving we needed to simplify the task."),
        ("Phase 2", "Binary Veg/Non-Veg — EfficientNet, No YOLO", "EfficientNet-B2", "~1,439 (binary)", "89.30%", "N/A",
         "Restructuring to 2 classes caused a dramatic accuracy jump (+~5pp). "
         "Proved that learning dietary categories is far easier than exact dish names."),
        ("Phase 2B", "MobileNetV3 + YOLO (First YOLO Run)", "MobileNetV3+YOLO", "~1,439 (binary)", "92.70%", "N/A",
         "Adding YOLO segmentation was the project's biggest single improvement: "
         "+3.4pp from 89.3% → 92.7% by removing background noise."),
        ("Phase 3", "Cross-Domain: German Restaurant Dataset", "MobileNetV3+YOLO", "2,191 German images", "95.53%", "N/A",
         "Fine-tuning on an entirely different cuisine (German) proved the pipeline generalizes "
         "across domains. Weakest dishes: Kartoffelpuffer (83.3%) and Bratkartoffeln (84.0%)."),
        ("Phase 4", "Real-World Arabic Restaurant Testing", "German-Tuned Model", "53 restaurant images", "N/A", "~77.36%",
         "First real-world restaurant test revealed a domain gap. The German-tuned model "
         "struggled with Arabic visual ambiguity: Shawarma vs flatbread, Mandi meat vs rice."),
        ("Phase 5", "Hierarchical 26-Class Dish Classifier", "MobileNetV3+YOLO", "~1,439 (26 classes)", "~68%", "N/A",
         "Returned to dish-level classification with YOLO preprocessing. "
         "67.6% val accuracy — strong for a 26-way task but limited by training data volume."),
        ("Phase 6", "Expanded 35-Class Dish Classifier", "MobileNetV3+YOLO", "1,306 (35 classes)", "67.01%", "N/A",
         "Expanding to 35 classes naturally dropped accuracy slightly. "
         "Binary Veg/Non-Veg accuracy remained at 98% via category aggregation."),
        ("Phase 7", "Refined Binary — V5 Dataset", "MobileNetV3+YOLO", "1,231 images (binary)", "89.43%", "N/A",
         "Cleaned back to core 26 dishes, retrained as binary. Early stopping at Epoch 8 Stage 2."),
        ("Phase 8", "Shawarma-Boosted — PEAK Restaurant Score", "MobileNetV3+YOLO", "1,319 (V5)", "91.20%", "92.73% ★",
         "17× Shawarma data boost pushed 8/10 restaurant dishes to 100%. "
         "This was the ALL-TIME BEST restaurant accuracy. Shawarma still at 57.1%."),
        ("Phase 9", "V6 Rebalance (1,432 images)", "MobileNetV3+YOLO", "1,432 (V6)", "92.0%", "90.9%",
         "Added 113 new images. Shish Tawook recovered to 100%. Shawarma still 57.1%. "
         "Standard test improved to 92.0% due to better class balance."),
        ("Phase 10", "Architecture Experiment: MobileNet vs EfficientNet", "Both", "V6 (1,432)", "92.0% / 84.0%", "90.9% / 90.9%",
         "Both architectures hit the same 90.9% restaurant ceiling — proving the bottleneck "
         "is data, not architecture. EfficientNet Stage 2 cancelled (>5h/epoch on CPU)."),
    ]

    for ph, title, model, data, std, rst, desc in phases:
        whead(doc, f"{ph} — {title}", level=2, color="2E75B6")
        wtable(doc,
            ["Attribute", "Value"],
            [
                ["Model",            model],
                ["Dataset",          data],
                ["Standard Test Acc.",std],
                ["Restaurant Acc.",  rst],
            ]
        )
        wbody(doc, desc)
        doc.add_paragraph()

    doc.add_page_break()

    # ══════════════════════════════════════════════════════════════════════════
    # CHAPTER 6 — RESULTS & METRICS
    # ══════════════════════════════════════════════════════════════════════════
    whead(doc, "Chapter 6 — Results & Performance Metrics", color="1F3864")
    wbody(doc, "Primary Model: MobileNetV3-Large + YOLOv8n-seg  |  Dataset: V6 (1,432 images)  |  Resolution: 344×344px", italic=True)
    doc.add_paragraph()

    whead(doc, "6.1  Standard Test Set Results (534 images)", level=2, color="2E75B6")
    wtable(doc,
        ["Metric", "Non-Veg Class", "Veg Class", "Overall (Weighted)"],
        [
            ["Precision",   "0.93", "0.92", "0.927"],
            ["Recall",      "0.93", "0.93", "0.927"],
            ["F1-Score",    "0.93", "0.93", "0.927"],
            ["Support",     "271",  "263",  "534"],
            ["Accuracy",    "—",    "—",    "92.70%"],
        ]
    )
    doc.add_paragraph()
    wimg(doc, rp("results_mobilenet_v3_large_yolo_seg/training_curves.png"), 6.0,
         "Figure 6 — Training & Validation Loss/Accuracy Curves (MobileNetV3 + YOLO)")
    wimg(doc, rp("results_mobilenet_v3_large_yolo_seg/confusion_matrix.png"), 5.5,
         "Figure 7 — Confusion Matrix: Standard Validation Set (MobileNetV3 + YOLO Seg)")
    doc.add_paragraph()

    whead(doc, "6.2  External Test Set (534 images — zero overlap with training)", level=2, color="2E75B6")
    wimg(doc, rp("results/test_external_mobilenet_v3_large_yolo_seg/confusion_matrix.png"), 5.5,
         "Figure 8 — Confusion Matrix: External Standard Test Set (no training overlap)")
    doc.add_page_break()

    # ══════════════════════════════════════════════════════════════════════════
    # CHAPTER 7 — RESTAURANT TESTING
    # ══════════════════════════════════════════════════════════════════════════
    whead(doc, "Chapter 7 — Real-World Restaurant Testing", color="1F3864")
    wbody(doc,
        "The ultimate benchmark for this project was not the standard test set but "
        "real-world images from Saudi restaurants — out-of-distribution photos with "
        "complex plating, restaurant lighting, and ambiguous presentations."
    )
    doc.add_paragraph()
    wimg(doc, charts[2], 6.2, "Figure 9 — Restaurant Test: Per-Dish Accuracy & Confidence Scores")
    doc.add_paragraph()

    whead(doc, "Per-Dish Results (Current Model)", level=2, color="2E75B6")
    wtable(doc,
        ["Dish", "Category", "Correct/Total", "Accuracy", "Avg Confidence", "Status"],
        [
            ["Okra Tagine",       "Veg",     "2/4",  "50.0%",  "0.730", "⚠️ Critical"],
            ["Za'atar Manaqish",  "Veg",     "4/5",  "80.0%",  "0.775", "⚠️ Needs Work"],
            ["Shawarma",          "Non-Veg", "6/7",  "85.7%",  "0.742", "⚠️ Needs Work"],
            ["Shish Tawook",      "Non-Veg", "6/7",  "85.7%",  "0.850", "⚠️ Needs Work"],
            ["Cheese Fatayer",    "Veg",     "5/5",  "100.0%", "0.775", "✅ Perfect"],
            ["Hummus",            "Veg",     "5/5",  "100.0%", "0.864", "✅ Perfect"],
            ["Mandi",             "Non-Veg", "7/7",  "100.0%", "0.866", "✅ Perfect"],
            ["Meat Madfoon",      "Non-Veg", "5/5",  "100.0%", "0.965", "✅ Perfect"],
            ["Mixed Grill",       "Non-Veg", "5/5",  "100.0%", "0.930", "✅ Perfect"],
            ["Tamees Bread",      "Veg",     "5/5",  "100.0%", "0.915", "✅ Perfect"],
            ["OVERALL",           "—",       "47/53","88.68%", "0.852", "—"],
        ]
    )
    doc.add_paragraph()

    wimg(doc, rp("results/restaurant_analysis/overall_accuracy_pie.png"), 4.5,
         "Figure 10 — Overall Restaurant Test Accuracy Pie Chart")
    wimg(doc, rp("results/restaurant_analysis/confidence_distribution.png"), 6.0,
         "Figure 11 — Confidence Score Distribution for Restaurant Predictions")
    wimg(doc, rp("results/restaurant_analysis/per_dish_accuracy.png"), 6.0,
         "Figure 12 — Per-Dish Accuracy Bar Chart (Restaurant Test Set)")
    doc.add_page_break()

    # ══════════════════════════════════════════════════════════════════════════
    # CHAPTER 8 — ARCHITECTURE COMPARISON
    # ══════════════════════════════════════════════════════════════════════════
    whead(doc, "Chapter 8 — Architecture Comparison: MobileNetV3 vs EfficientNet-B2", color="1F3864")
    wbody(doc,
        "In Phase 10, both architectures were trained on the identical V6 dataset "
        "at 344×344 resolution with the same hyperparameters. This was a controlled "
        "experiment to isolate the effect of architecture alone."
    )
    doc.add_paragraph()
    wimg(doc, charts[3], 6.0, "Figure 13 — Radar Chart: MobileNetV3-Large vs EfficientNet-B2 (Stage 1 only)")
    doc.add_paragraph()

    whead(doc, "Key Finding: The 90.9% Restaurant Ceiling", level=2, color="C00000")
    wbody(doc,
        "Both architectures hit exactly 90.9% restaurant accuracy despite very different "
        "standard test scores (92% vs 84%). This reveals a fundamental truth: the bottleneck "
        "is NOT the architecture — it is the training data. The same 4-5 images are "
        "always misclassified regardless of which model is used.",
        bold=False
    )
    doc.add_paragraph()

    wtable(doc,
        ["Dish", "MobileNetV3", "EfficientNet-B2", "Analysis"],
        [
            ["Cheese Fatayer",   "100%",        "100%",        "Perfect on both"],
            ["Hummus",           "100%",        "100%",        "Perfect on both"],
            ["Mandi",            "100%",        "100%",        "Perfect on both"],
            ["Meat Madfoon",     "100%",        "100%",        "Perfect on both"],
            ["Mixed Grill",      "100%",        "100%",        "Perfect on both"],
            ["Tamees Bread",     "100%",        "100%",        "Perfect on both"],
            ["Shawarma",         "71.4% (5/7)", "85.7% (6/7)", "EfficientNet found hidden meat texture"],
            ["Shish Tawook",     "85.7% (6/7)", "85.7% (6/7)", "Identical — garlic sauce fools both"],
            ["Za'atar Manaqish", "100% (5/5)",  "80.0% (4/5)", "EfficientNet over-fitted herb texture"],
            ["Okra Tagine",      "75.0% (3/4)", "50.0% (2/4)", "EfficientNet confused by stew texture"],
        ]
    )
    doc.add_page_break()

    wimg(doc, rp("results_efficientnet_b2/training_curves.png"), 6.0,
         "Figure 14 — EfficientNet-B2 Training Curves (Phase 2 — No YOLO)")
    wimg(doc, rp("results_efficientnet_b2/confusion_matrix.png"), 5.5,
         "Figure 15 — EfficientNet-B2 Confusion Matrix (Phase 2)")
    doc.add_page_break()

    # ══════════════════════════════════════════════════════════════════════════
    # CHAPTER 9 — FAILURE ANALYSIS
    # ══════════════════════════════════════════════════════════════════════════
    whead(doc, "Chapter 9 — Known Flaws & Failure Analysis", color="1F3864")
    wbody(doc,
        "Understanding where the model fails is as important as knowing where it succeeds. "
        "This chapter documents every known failure mode with root causes and attempted fixes."
    )
    wimg(doc, charts[7], 6.0, "Figure 16 — Failure Factor Heatmap: Risk Level by Dish and Issue Type")
    doc.add_paragraph()

    failures = [
        ("Failure 1 — Shawarma: The Persistent Problem", "C00000",
         "Shawarma is the most persistent failure across all 10 phases. The restaurant "
         "serves Shawarma wrapped in pita bread, so the exterior looks like a flatbread "
         "vegetarian dish. The hidden meat inside is invisible to the camera.",
         [("Root cause", "Pita wrapper hides all meat — model sees bread exterior"),
          ("Phase 8 fix", "17× data boost (5 → 86 images) — improved but not solved"),
          ("Phase 9 status", "Still at 57.1% — new images did not match restaurant pita-wrap style"),
          ("Fix needed", "50-100 photos specifically showing pita-wrapped Shawarma cross-sections")]),

        ("Failure 2 — Mandi: The Sea of Rice Problem", "C00000",
         "Mandi is visually dominated by yellow/brown rice (70-90% of pixels). The meat "
         "(chicken or lamb) is often small, buried under the rice, or camouflaged by "
         "similar-coloured roasted vegetables and nuts.",
         [("Root cause 1", "Large rice portion fools model into predicting Veg"),
          ("Root cause 2", "YOLO sometimes crops out the meat at plate edges — then classifier only sees rice"),
          ("Root cause 3", "Slow-roasted meat has golden-brown colour identical to roasted vegetables"),
          ("Status", "Variable — 100% in Phase 8 but dropped to 85.7% in Phase 9")]),

        ("Failure 3 — Okra Tagine: Label Ambiguity", "E67E22",
         "Okra Tagine can be prepared as a fully vegetarian stew or as a meat stew "
         "with lamb or beef. This creates inherent label noise in the dataset — the "
         "model cannot be expected to classify accurately when the ground truth itself is ambiguous.",
         [("Root cause", "Same visual appearance for Veg and Non-Veg versions of the dish"),
          ("EfficientNet result", "50.0% — worse than random chance"),
          ("Fix needed", "Consistent labelling policy OR a separate 'preparation-dependent' class")]),

        ("Failure 4 — The 90.9% Restaurant Ceiling", "E67E22",
         "Every model variation tested in Phases 9 and 10 hits the exact same 90.9% "
         "restaurant accuracy. This is a mathematical data ceiling, not a model limitation.",
         [("Variables already tested", "Image resolution (288 vs 344), YOLO vs raw, MobileNetV3 vs EfficientNet-B2"),
          ("Consistent failures", "Same 4-5 images always misclassified regardless of model"),
          ("Conclusion", "Training data is the bottleneck — more model tuning will not help"),
          ("Only fix", "Targeted data collection for Shawarma, Okra Tagine, Za'atar Manaqish")]),

        ("Failure 5 — EfficientNet Stage 2 Cancellation", "767676",
         "EfficientNet-B2 Stage 2 fine-tuning was cancelled because the estimated training "
         "time exceeded 5 hours per epoch on CPU. The Phase 10 comparison is therefore "
         "between a fully fine-tuned MobileNetV3 and an only half-trained EfficientNet.",
         [("Parameters", "EfficientNet-B2: ~9M vs MobileNetV3-Large: ~5M"),
          ("Impact", "Phase 10 results are not a fair architecture comparison"),
          ("Fix needed", "GPU training environment for EfficientNet Stage 2")]),
    ]

    for title, col, desc, bullets in failures:
        whead(doc, title, level=2, color=col)
        wbody(doc, desc)
        for label, value in bullets:
            wbullet(doc, label + ":", value, color=col)
        doc.add_paragraph()

    doc.add_page_break()

    # ══════════════════════════════════════════════════════════════════════════
    # CHAPTER 10 — LESSONS LEARNED
    # ══════════════════════════════════════════════════════════════════════════
    whead(doc, "Chapter 10 — Lessons Learned", color="1F3864")
    lessons = [
        ("Data Quality > Model Architecture",
         "The 90.9% restaurant ceiling held across MobileNetV3 and EfficientNet-B2. "
         "Investing in better targeted training data would outperform any architecture switch."),
        ("YOLO Pre-processing was Transformative",
         "A single preprocessing step (YOLOv8 background removal) gained +3.4 percentage points "
         "(89.3% → 92.7%). This had more impact than any hyperparameter or architecture change."),
        ("Binary Classification is 20-25pp Easier Than Multi-Class",
         "Veg/Non-Veg binary classification consistently outperforms 26/35-class dish-level "
         "classification by a wide margin, because the model learns discriminative dietary features "
         "rather than subtle visual differences between individual dishes."),
        ("Cross-Domain Transfer Learning Works",
         "Fine-tuning on the German dataset (entirely different cuisine) reached 95.53%, proving "
         "the YOLO + MobileNetV3 backbone generalizes well across food domains."),
        ("Targeted Augmentation Must Match Deployment Domain",
         "Shawarma's 17× boost improved standard test accuracy but restaurant accuracy "
         "did not recover because the new training images did not match the pita-wrapped "
         "style used in the actual restaurant test set."),
        ("YOLO Can Hurt as Well as Help",
         "For dishes like Mandi, YOLO's crop can accidentally remove the meat that sits on "
         "the plate edges, causing misclassification as Veg. Segmentation is not always optimal."),
        ("The 90.9% Ceiling is a Data Problem",
         "After testing resolution, architecture, and preprocessing variations, the conclusion "
         "is clear: the same 4-5 images are always misclassified. "
         "Only targeted data collection for Shawarma, Okra Tagine, and Za'atar Manaqish will break this ceiling."),
        ("Class Imbalance Correction is Critical",
         "Using inverse-frequency class weights in CrossEntropyLoss was essential for preventing "
         "the model from defaulting to the majority class in imbalanced datasets."),
    ]
    for i, (title, desc) in enumerate(lessons, 1):
        whead(doc, f"Lesson {i}: {title}", level=2, color="2E75B6")
        wbody(doc, desc)
        doc.add_paragraph()
    doc.add_page_break()

    # ══════════════════════════════════════════════════════════════════════════
    # CHAPTER 11 — HOW TO RUN
    # ══════════════════════════════════════════════════════════════════════════
    whead(doc, "Chapter 11 — How to Run the Project", color="1F3864")
    wbody(doc, "This chapter covers every command needed to reproduce the full pipeline from scratch.", italic=True)
    doc.add_paragraph()

    steps = [
        ("Step 1 — Clone the Repository", [
            "git clone https://github.com/Mehdidar/DishDetect-SA.git",
            "cd DishDetect-SA",
        ]),
        ("Step 2 — Install Dependencies", [
            "pip install -r requirements.txt",
        ]),
        ("Step 3 — Validate Your Dataset", [
            "python tools/dataset_check.py --dataset /path/to/dataset",
        ]),
        ("Step 4 — Crop Images with YOLO", [
            "python scripts/yolo_seg_crop.py --input /path/to/raw --output /path/to/cropped",
        ]),
        ("Step 5 — Train the Model", [
            "python scripts/train.py --dataset /path/to/cropped",
            "# Results saved to checkpoints/ and results/",
        ]),
        ("Step 6 — Evaluate on Standard Test", [
            "python scripts/evaluate.py --dataset /path/to/cropped",
        ]),
        ("Step 7 — Evaluate on External Test", [
            "python scripts/evaluate_external.py --test_dir /path/to/external_test",
        ]),
        ("Step 8 — Predict a Single Image", [
            "python scripts/predict.py --image /path/to/food.jpg --threshold 0.7",
        ]),
        ("Step 9 — Run Restaurant Analysis", [
            "python scripts/restaurant_analysis.py",
            "# Outputs charts to results/restaurant_analysis/",
        ]),
        ("Step 10 — Launch the GUI", [
            "python app/ui.py",
        ]),
        ("Step 11 — Generate All Reports", [
            "python scripts/generate_master_report.py",
            "python scripts/generate_architecture_report.py",
            "python scripts/generate_walkthrough_book.py",
        ]),
    ]
    for title, cmds in steps:
        whead(doc, title, level=2, color="2E75B6")
        for cmd in cmds:
            wcode(doc, "$ " + cmd)
        doc.add_paragraph()
    doc.add_page_break()

    # ══════════════════════════════════════════════════════════════════════════
    # CHAPTER 12 — NEXT STEPS
    # ══════════════════════════════════════════════════════════════════════════
    whead(doc, "Chapter 12 — Next Steps & Roadmap", color="1F3864")

    whead(doc, "Immediate Wins — Data Collection", level=2, color="2E75B6")
    wbullet(doc, "Collect 50-100 pita-wrapped Shawarma images",
            "specifically the cross-section view revealing the meat inside")
    wbullet(doc, "Collect diverse Mandi images",
            "where the chicken/lamb is prominently visible above the rice")
    wbullet(doc, "Standardise Okra Tagine labelling",
            "— either split into Veg/Non-Veg subcategories or set a clear policy")

    doc.add_paragraph()
    whead(doc, "Pipeline Improvements", level=2, color="2E75B6")
    wbullet(doc, "Disable YOLO crop for Mandi",
            "and test if full-image classification improves the meat/rice disambiguation")
    wbullet(doc, "Train EfficientNet-B2 Stage 2 on GPU",
            "for a fair architecture comparison")
    wbullet(doc, "Implement Test-Time Augmentation (TTA)",
            "to improve robustness on ambiguous low-confidence predictions")
    wbullet(doc, "Add confidence threshold to UI",
            "flag predictions below 65% as 'Uncertain — Please Verify'")
    wbullet(doc, "Add Grad-CAM visualisations",
            "to show which food region the model is focusing on for each prediction")

    doc.add_paragraph()
    whead(doc, "Scaling & Deployment", level=2, color="2E75B6")
    wbullet(doc, "Export model to ONNX / TFLite",
            "for mobile and edge device deployment")
    wbullet(doc, "Build a REST API",
            "using FastAPI or Flask for restaurant POS system integration")
    wbullet(doc, "Expand to additional Arabic cuisines",
            "— Jareesh, Kabsa, Saleeg, Mansaf, Machboos")
    wbullet(doc, "Publish model to Hugging Face Hub",
            "for community access and benchmarking")

    doc.add_page_break()

    # ══════════════════════════════════════════════════════════════════════════
    # CHAPTER 13 — VEGAN VS. VEGETARIAN (MEAT-FREE) INTELLIGENCE MODULE
    # ══════════════════════════════════════════════════════════════════════════
    whead(doc, "Chapter 13 — Vegan vs. Vegetarian (Meat-Free) Intelligence Module", color="1F3864")

    wbody(doc,
        "In commercial food apps, dietary screening tools, and restaurant POS systems, consumers frequently "
        "require a distinction beyond simply identifying meat. Vegetarians and strict vegans have distinct dietary "
        "boundaries: while vegetarians abstain from animal flesh, they may consume dairy (milk, cheese, yogurt), eggs, "
        "and clarified butter (samneh / ghee). In contrast, strict vegans consume 100% plant-based foods only. "
        "This chapter introduces DishDetect SA's specialized meat-free intelligence module, designed specifically "
        "to solve the subtle visual discrimination problem between Vegan and Lacto-Ovo Vegetarian dishes."
    )

    doc.add_paragraph()
    whead(doc, "13.1  Dietary Scope: Strict Meat Exclusion", level=2, color="2E75B6")
    wbody(doc,
        "All 10 meat, poultry, fish, and seafood categories (Mixed Grill, Shawarma, Shish Tawook, Grilled Kofta, "
        "Meat Maqluba, Meat Pie, Kofta Stew, Shrimp Sayadiyah, Stuffed Pigeon, and Meat Madfoon) were completely excluded. "
        "The model evaluates exclusively within the meat-free cohort of 15 Arabic dishes, eliminating trivial meat-vs-salad "
        "cues and forcing the neural network to learn fine-grained visual indicators of dairy, eggs, and animal fats."
    )

    doc.add_paragraph()
    whead(doc, "13.2  Culinary & Ingredient Audit of the 15 Meatless Dishes", level=2, color="2E75B6")
    wtable(doc,
        ["Dish Name", "Dietary Category", "Images", "Key Ingredients & Animal Byproduct Audit"],
        [
            ["Hummus",            "Vegan",      "61", "Chickpeas, tahini (sesame paste), olive oil, garlic. 100% plant-based."],
            ["Tamees Bread",      "Vegan",      "53", "Wheat flour, yeast, water, vegetable oil (tandoor-baked). No dairy."],
            ["Za’atar Manakish",  "Vegan",      "52", "Flatbread with wild thyme, sumac, sesame, extra virgin olive oil."],
            ["Okra Tagine",       "Vegan",      "45", "Bamya bil-Zayt: Baby okra pods, tomato gravy, coriander, olive oil."],
            ["Sesame Brittle",    "Vegan",      "45", "Simsimiya: Roasted sesame seeds bound with sugar syrup & lemon."],
            ["Spinach Fatayer",   "Vegan",      "44", "Levantine spinach pie: Spinach, onions, sumac, olive oil dough."],
            ["Onion Soup",        "Vegan",      "42", "Shorbat Basal: Caramelized onions, olive oil, vegetable broth, cumin."],
            ["Pomegranate Fattoush","Vegan",    "37", "Romaine, tomatoes, cucumbers, toasted pita chips, pomegranate molasses."],
            ["Spinach Stew",      "Vegan",      "32", "Sabanegh bil-Zayt: Fresh spinach, chickpeas, garlic, olive oil."],
            ["Fatayer Cheese",    "Vegetarian", "86", "Pastry filled with dairy cheese (Akkawi / Halloumi / Nabulsi) & butter."],
            ["Saffron Cake",      "Vegetarian", "59", "Sponge cake made with eggs, soaked in saffron condensed milk & cream."],
            ["Rice Pudding",      "Vegetarian", "57", "Roz Bel Laban: Short-grain rice cooked in whole cow's milk & ashta cream."],
            ["Pistachio Sweet",   "Vegetarian", "57", "Halawet el-Jibn / Mafroukeh: Contains sweet cheese, ashta, & samneh."],
            ["Pistachio Tamriya", "Vegetarian", "21", "Date confection pan-roasted in clarified butter (samneh / ghee)."],
            ["Tamriya",           "Vegetarian", "13", "Authentic Saudi date confection roasted in samneh (animal butterfat)."],
        ]
    )

    doc.add_paragraph()
    wimg(doc, c("09_vegan_intelligence.png"), width=6.2,
         caption="Figure 9 — Vegan vs. Vegetarian dataset composition (left) and benchmark performance across splits (right)")

    doc.add_paragraph()
    whead(doc, "13.3  Model Architecture & Training Schedule", level=2, color="2E75B6")
    wbullet(doc, "Architecture:", "MobileNetV3-Large (torchvision), ImageNet pretrained")
    wbullet(doc, "Input Resolution:", "224×224 pixels")
    wbullet(doc, "Transfer Learning Schedule:", "Stage 1 (Classifier Head, LR 1e-3, AdamW) for 10 epochs → Stage 2 (Backbone fine-tuning, LR 1e-5, Cosine Annealing) for 15 epochs")
    wbullet(doc, "Loss Function:", "Cross-Entropy with inverse-frequency class weights (Vegan: 0.86, Vegetarian: 1.20) to counter the mild 1.4:1 ratio")
    wbullet(doc, "Leakage Prevention:", "Three-way 70/15/15 split grouped by perceptual hash (phash <= 5) to isolate near-duplicates")

    doc.add_paragraph()
    whead(doc, "13.4  Experimental Results & Test Set Benchmarks", level=2, color="2E75B6")
    wtable(doc,
        ["Evaluation Split", "Total Images", "Accuracy", "Weighted Precision", "Weighted Recall", "Weighted F1-Score"],
        [
            ["Validation Split", "107", "88.79%", "0.8879", "0.8879", "0.8879"],
            ["Held-Out Test Split", "107", "87.85%", "0.8875", "0.8785", "0.8793"],
        ]
    )

    doc.add_paragraph()
    wbody(doc,
        "On the unbiased held-out test set (107 images), the model achieved 95% precision for the Vegan class "
        "(only 3 vegetarian images were mistakenly classified as vegan) and 93% recall for the Vegetarian class, "
        "demonstrating strong capability in identifying dairy cheese fillings and rich butter pastries."
    )

    doc.add_paragraph()
    whead(doc, "13.5  Real-World Restaurant Testing (Meat-Free Dishes)", level=2, color="2E75B6")
    wtable(doc,
        ["Restaurant Dish", "Ground Truth", "Correct / Total", "Accuracy", "Culinary Analysis"],
        [
            ["Hummus",          "Vegan",      "5 / 5", "100.0%", "Perfect plant-based legume dip recognition"],
            ["Okra Tagine",     "Vegan",      "4 / 4", "100.0%", "Correctly identified as olive-oil vegetable stew"],
            ["Tamees Bread",    "Vegan",      "5 / 5", "100.0%", "Perfect recognition of tandoor plain flatbread"],
            ["Cheese Fatayer",  "Vegetarian", "4 / 5", "80.0%",  "Successfully detects dairy cheese; recovered from Veg failure"],
            ["Za’atara Manaqish","Vegan",     "2 / 5", "40.0%",  "Heavy olive oil sheen sometimes mimics melted cheese glaze"],
            ["OVERALL",         "—",          "20 / 24", "83.33%", "Strong generalization to unseen restaurant photography"],
        ]
    )

    doc.add_paragraph()
    whead(doc, "13.6  Standalone Package Structure", level=2, color="2E75B6")
    wbody(doc,
        "A dedicated, self-contained package has been published under Vegan_vs_Vegetarian_Classifier/ with "
        "its own weights, dataset symlinks, inference tools, and evaluation scripts:"
    )
    wcode(doc,
        "# Run single-image inference:\n"
        ".venv/bin/python Vegan_vs_Vegetarian_Classifier/scripts/predict.py --image 'path/to/dish.jpg'\n\n"
        "# Run held-out test evaluation:\n"
        ".venv/bin/python Vegan_vs_Vegetarian_Classifier/scripts/evaluate.py --split test\n\n"
        "# Run real-world restaurant test:\n"
        ".venv/bin/python Vegan_vs_Vegetarian_Classifier/scripts/evaluate_restaurant.py"
    )

    doc.add_paragraph()
    wbody(doc,
        f"\n\nDishDetect SA — Walkthrough Book  ·  github.com/Mehdidar/DishDetect-SA  "
        f"·  Generated {datetime.now().strftime('%d %B %Y, %H:%M')}",
        italic=True, color="767676"
    )

    doc.save(OUTPUT)
    print(f"\n  Book saved → {OUTPUT}")

# ══════════════════════════════════════════════════════════════════════════════
# ENTRY POINT
# ══════════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    print("\nGenerating charts...")
    charts = [
        chart_pipeline(),
        chart_accuracy_evolution(),
        chart_restaurant_dishes(),
        chart_arch_comparison(),
        chart_dataset_timeline(),
        chart_training_flow(),
        chart_yolo_pipeline(),
        chart_failure_heatmap(),
        chart_vegan_intelligence(),
    ]
    print("\nBuilding walkthrough book...")
    build_book(charts)
    print("\nAll done!")
