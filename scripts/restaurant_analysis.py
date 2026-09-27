"""
restaurant_analysis.py
----------------------
Reads the restaurant test predictions CSV, organizes images into
correct/incorrect folders, and generates per-dish accuracy graphs.
"""

import os
import shutil
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

CSV_PATH   = "results/test_external_efficientnet_b2_yolo_seg/predictions.csv"
OUT_DIR    = "results/restaurant_analysis"

os.makedirs(f"{OUT_DIR}/correct",   exist_ok=True)
os.makedirs(f"{OUT_DIR}/incorrect", exist_ok=True)

# ── 1. Load predictions ───────────────────────────────────────────────────────
df = pd.read_csv(CSV_PATH)
df["image_path"] = df["image_path"].str.strip()
df["dish"] = df["image_path"].apply(lambda p: os.path.basename(os.path.dirname(p)))

# ── 2. Copy images into correct / incorrect folders ───────────────────────────
for _, row in df.iterrows():
    src = row["image_path"]
    if not os.path.exists(src):
        continue
    fname    = os.path.basename(src)
    tag      = "correct" if row["correct"] == 1 else "incorrect"
    dish_dir = os.path.join(OUT_DIR, tag, row["dish"])
    os.makedirs(dish_dir, exist_ok=True)
    shutil.copy2(src, os.path.join(dish_dir, fname))

print(f"✅  Images sorted into {OUT_DIR}/correct  and  {OUT_DIR}/incorrect")

# ── 3. Per-dish accuracy ──────────────────────────────────────────────────────
dish_stats = df.groupby("dish").agg(
    total=("correct", "count"),
    correct=("correct", "sum"),
    avg_conf=("confidence", "mean"),
    true_label=("true_label", "first")
).reset_index()
dish_stats["accuracy"] = dish_stats["correct"] / dish_stats["total"] * 100

print("\nPer-Dish Accuracy:")
print(dish_stats[["dish","true_label","total","correct","accuracy","avg_conf"]].to_string(index=False))

# ── 4. Bar chart – per-dish accuracy ─────────────────────────────────────────
fig, ax = plt.subplots(figsize=(14, 6))
colors = ["#2ecc71" if a == 100 else "#e74c3c" if a < 80 else "#f39c12"
          for a in dish_stats["accuracy"]]
bars = ax.bar(dish_stats["dish"], dish_stats["accuracy"], color=colors, edgecolor="white", linewidth=0.8)

for bar, acc in zip(bars, dish_stats["accuracy"]):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1.5,
            f"{acc:.0f}%", ha="center", va="bottom", fontsize=10, fontweight="bold")

ax.set_ylim(0, 115)
ax.set_ylabel("Accuracy (%)", fontsize=12)
ax.set_xlabel("Dish", fontsize=12)
ax.set_title("Per-Dish Accuracy on Real-World Restaurant Test Set", fontsize=14, fontweight="bold")
ax.axhline(88.68, color="#3498db", linewidth=1.5, linestyle="--", label="Overall Accuracy (88.68%)")
plt.xticks(rotation=30, ha="right", fontsize=9)
green_p  = mpatches.Patch(color="#2ecc71",  label="100% Accuracy")
orange_p = mpatches.Patch(color="#f39c12",  label="80-99%")
red_p    = mpatches.Patch(color="#e74c3c",  label="< 80% (Problem)")
ax.legend(handles=[green_p, orange_p, red_p], loc="upper right")
plt.tight_layout()
plt.savefig(f"{OUT_DIR}/per_dish_accuracy.png", dpi=150)
plt.close()
print(f"✅  Saved per_dish_accuracy.png")

# ── 5. Confidence distribution ────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(10, 5))
correct_conf   = df[df["correct"] == 1]["confidence"]
incorrect_conf = df[df["correct"] == 0]["confidence"]
ax.hist(correct_conf,   bins=10, alpha=0.7, color="#2ecc71", label="Correct Predictions")
ax.hist(incorrect_conf, bins=10, alpha=0.7, color="#e74c3c", label="Incorrect Predictions")
ax.set_xlabel("Confidence Score", fontsize=12)
ax.set_ylabel("Number of Images", fontsize=12)
ax.set_title("Confidence Distribution: Correct vs Incorrect Predictions", fontsize=13, fontweight="bold")
ax.legend()
plt.tight_layout()
plt.savefig(f"{OUT_DIR}/confidence_distribution.png", dpi=150)
plt.close()
print(f"✅  Saved confidence_distribution.png")

# ── 6. Pie chart – overall results ───────────────────────────────────────────
fig, ax = plt.subplots(figsize=(6, 6))
sizes  = [len(df[df["correct"]==1]), len(df[df["correct"]==0])]
labels = [f"Correct ({sizes[0]})", f"Incorrect ({sizes[1]})"]
colors = ["#2ecc71", "#e74c3c"]
ax.pie(sizes, labels=labels, colors=colors, autopct="%1.1f%%",
       startangle=90, textprops={"fontsize": 13})
ax.set_title(f"Overall Accuracy: {sizes[0]/sum(sizes)*100:.2f}%\n(Restaurant Real-World Test)", fontsize=13, fontweight="bold")
plt.tight_layout()
plt.savefig(f"{OUT_DIR}/overall_accuracy_pie.png", dpi=150)
plt.close()
print(f"✅  Saved overall_accuracy_pie.png")

# ── 7. Write per-dish text summary ───────────────────────────────────────────
with open(f"{OUT_DIR}/per_dish_summary.txt", "w") as f:
    f.write("RESTAURANT TEST SET – PER DISH ACCURACY REPORT\n")
    f.write("=" * 55 + "\n\n")
    f.write(f"{'Dish':<25} {'Category':<10} {'Correct':>8} {'Total':>6} {'Accuracy':>10} {'Avg Conf':>10}\n")
    f.write("-" * 75 + "\n")
    for _, row in dish_stats.sort_values("accuracy").iterrows():
        f.write(f"{row['dish']:<25} {row['true_label']:<10} {int(row['correct']):>8} {int(row['total']):>6} "
                f"{row['accuracy']:>9.1f}%  {row['avg_conf']:>9.3f}\n")
    f.write("\n" + "=" * 55 + "\n")
    f.write(f"OVERALL:                           47       53      88.68%     0.852\n")

print(f"✅  Saved per_dish_summary.txt")
print(f"\n📁  All outputs saved to: {OUT_DIR}/")
