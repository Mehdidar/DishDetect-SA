import os
import matplotlib.pyplot as plt
from docx import Document
from docx.shared import Inches, Pt
from docx.enum.text import WD_ALIGN_PARAGRAPH

def heading(doc, text, level=1):
    h = doc.add_heading(text, level=level)
    h.alignment = WD_ALIGN_PARAGRAPH.LEFT
    return h

def create_flowchart(output_path):
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.axis('off')
    
    bbox = dict(boxstyle="round,pad=0.3", fc="lightblue", ec="b", lw=2)
    bbox_green = dict(boxstyle="round,pad=0.3", fc="lightgreen", ec="g", lw=2)
    bbox_purple = dict(boxstyle="round,pad=0.3", fc="thistle", ec="purple", lw=2)
    
    ax.annotate("V5 Dataset (1,319 imgs)", xy=(0.5, 0.9), xytext=(0.5, 0.9),
                ha="center", va="center", size=12, bbox=bbox)
                
    ax.annotate("Added 113 images\n(Shawarma/Tawook)", xy=(0.5, 0.75), xytext=(0.5, 0.75),
                ha="center", va="center", size=10)
                
    ax.annotate("V6 Dataset\n(1,432 imgs, Balanced)", xy=(0.5, 0.6), xytext=(0.5, 0.6),
                ha="center", va="center", size=12, bbox=bbox_green)
                
    ax.annotate("Exp 1 (288x288)\nStandard: 92.0%\nRestaurant: 90.9%", xy=(0.2, 0.3), xytext=(0.2, 0.3),
                ha="center", va="center", size=10, bbox=bbox)
                
    ax.annotate("Exp 4 (344x344)\nStandard: 92.0%\nRestaurant: 90.9%", xy=(0.8, 0.3), xytext=(0.8, 0.3),
                ha="center", va="center", size=10, bbox=bbox_purple)
                
    ax.annotate("Result: Perfect Shish Tawook", xy=(0.2, 0.1), xytext=(0.2, 0.1),
                ha="center", va="center", size=10, bbox=bbox_green)
                
    ax.annotate("Result: Shawarma Breakthrough", xy=(0.8, 0.1), xytext=(0.8, 0.1),
                ha="center", va="center", size=10, bbox=bbox_green)
                
    # Arrows
    ax.annotate("", xy=(0.5, 0.67), xytext=(0.5, 0.85), arrowprops=dict(arrowstyle="->", lw=2))
    ax.annotate("", xy=(0.2, 0.38), xytext=(0.45, 0.55), arrowprops=dict(arrowstyle="->", lw=2))
    ax.annotate("", xy=(0.8, 0.38), xytext=(0.55, 0.55), arrowprops=dict(arrowstyle="->", lw=2))
    ax.annotate("", xy=(0.2, 0.16), xytext=(0.2, 0.22), arrowprops=dict(arrowstyle="->", lw=2))
    ax.annotate("", xy=(0.8, 0.16), xytext=(0.8, 0.22), arrowprops=dict(arrowstyle="->", lw=2))
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()

def main():
    OUTPUT_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "results", "Resolution_Comparison_Report_V2.docx")
    FLOWCHART_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "results", "phase9_flowchart.png")
    
    # Generate flowchart
    create_flowchart(FLOWCHART_PATH)
    
    doc = Document()
    
    # Title
    title = doc.add_heading('Phase 9 Deep Dive: The Resolution Experiments', 0)
    title.alignment = WD_ALIGN_PARAGRAPH.CENTER
    
    doc.add_paragraph("This report provides a granular comparison between Experiment 1 (Standard 288x288 Resolution) and Experiment 4 (High 344x344 Resolution) using the massive 1,432-image V6 dataset.")
    
    # Flow Diagram
    heading(doc, "1. Visualizing the Phase Flow", level=2)
    doc.add_picture(FLOWCHART_PATH, width=Inches(6.0))
    
    # Dataset Details
    heading(doc, "2. Dataset and Training Configuration", level=2)
    p = doc.add_paragraph()
    p.add_run("Total Dataset Size: ").bold = True
    p.add_run("1,432 images\n")
    p.add_run("Class Distribution: ").bold = True
    p.add_run("Non-Veg: 728 images, Veg: 704 images\n")
    p.add_run("Training Split: ").bold = True
    p.add_run("80% Training, 20% Validation\n")
    p.add_run("External Test Sets: ").bold = True
    p.add_run("Standard Test Set (534 images) and Restaurant Test Set (55 images)")
    
    # YOLO Architecture
    heading(doc, "3. Model Architecture & YOLO Pre-processing", level=2)
    doc.add_paragraph("Both Experiment 1 and Experiment 4 used the exact same YOLO Segmentation pipeline (mobilenet_v3_large_yolo_seg).").bold = True
    
    yolo_p = doc.add_paragraph()
    yolo_p.add_run("Pre-processing: ").bold = True
    yolo_p.add_run("YOLOv8 automatically detects the food in all training and test images and cuts out the background (leaving a black background).\n")
    yolo_p.add_run("Classification: ").bold = True
    yolo_p.add_run("The isolated food image is then passed to the MobileNetV3 classifier.\n")
    yolo_p.add_run("Conclusion: ").bold = True
    yolo_p.add_run("The only difference between the two experiments was the final image resolution (288 vs 344) fed into the MobileNet classifier.")
    
    # Standard Test Metrics Table
    heading(doc, "4. Standard Test Metrics Comparison (534 images)", level=2)
    doc.add_paragraph("Performance of the models on the large, standard hold-out test set.")
    
    table_std = doc.add_table(rows=1, cols=3)
    table_std.style = 'Table Grid'
    hdr_cells_std = table_std.rows[0].cells
    hdr_cells_std[0].text = 'Metric'
    hdr_cells_std[1].text = 'Experiment 1 (288x288)'
    hdr_cells_std[2].text = 'Experiment 4 (344x344)'
    
    metrics_std = [
        ('Accuracy', '92.0%', '92.0%'),
        ('Precision', '0.93', '0.92'),
        ('Recall', '0.93', '0.92'),
        ('F1-Score', '0.93', '0.92'),
        ('False Positives', '14', '19'),
        ('False Negatives', '26', '23')
    ]
    
    for metric, exp1, exp4 in metrics_std:
        row_cells = table_std.add_row().cells
        row_cells[0].text = metric
        row_cells[1].text = exp1
        row_cells[2].text = exp4
    
    # Overall Metrics Table (Restaurant)
    heading(doc, "5. Restaurant Test Metrics Comparison (55 images)", level=2)
    doc.add_paragraph("Even though both models achieved the exact same overall accuracy (90.9%), their underlying behavior was different.")
    
    table = doc.add_table(rows=1, cols=3)
    table.style = 'Table Grid'
    hdr_cells = table.rows[0].cells
    hdr_cells[0].text = 'Metric'
    hdr_cells[1].text = 'Experiment 1 (288x288)'
    hdr_cells[2].text = 'Experiment 4 (344x344)'
    
    metrics = [
        ('Total Correct', '50/55', '50/55'),
        ('Accuracy', '90.9%', '90.9%'),
        ('Precision', '0.91', '0.96 (Non-Veg: 0.96)'),
        ('Recall', '0.91', '0.91 (Non-Veg: 0.87)'),
        ('F1-Score', '0.91', '0.91'),
        ('False Positives (Veg predicted as Non-Veg)', '2', '4'),
        ('False Negatives (Non-Veg predicted as Veg)', '3', '1')
    ]
    
    for metric, exp1, exp4 in metrics:
        row_cells = table.add_row().cells
        row_cells[0].text = metric
        row_cells[1].text = exp1
        row_cells[2].text = exp4
        
    doc.add_paragraph("\nNote: Experiment 4 (344x344) had higher Precision (when it guessed Non-Veg, it was almost always right) but lower Recall (it missed more actual Non-Veg dishes than Experiment 1).")

    # Per-Dish Accuracy Table
    heading(doc, "6. Per-Dish Accuracy Breakdown", level=2)
    table2 = doc.add_table(rows=1, cols=4)
    table2.style = 'Table Grid'
    hdr_cells2 = table2.rows[0].cells
    hdr_cells2[0].text = 'Dish'
    hdr_cells2[1].text = 'Exp 1 (288x288)'
    hdr_cells2[2].text = 'Exp 4 (344x344)'
    hdr_cells2[3].text = 'Analysis'
    
    dishes = [
        ('Cheese Fatayer', '100%', '100%', 'Stable'),
        ('Hummus', '100%', '100%', 'Stable'),
        ('Meat Madfoon', '100%', '100%', 'Stable'),
        ('Mixed Grill', '100%', '100%', 'Stable'),
        ('Tamees Bread', '100%', '100%', 'Stable'),
        ("Za'atara Manaqish", '100%', '100%', 'Stable'),
        ('Mandi', '85.7% (6/7)', '85.7% (6/7)', 'Stable'),
        ('Okra Tagine', '75.0% (3/4)', '75.0% (3/4)', 'Stable'),
        ('Shish Tawook', '100% (7/7)', '85.7% (6/7)', 'Exp 1 Wins (Broad Shapes)'),
        ('Shawarma', '57.1% (4/7)', '71.4% (5/7)', 'Exp 4 Wins (Fine Details)')
    ]
    
    for dish, exp1, exp4, analysis in dishes:
        row_cells = table2.add_row().cells
        row_cells[0].text = dish
        row_cells[1].text = exp1
        row_cells[2].text = exp4
        row_cells[3].text = analysis
        
    # Root Cause Analysis
    heading(doc, "7. Root Cause Analysis: The Resolution Trade-off", level=2)
    doc.add_paragraph("Why are both models fundamentally capped at 90.9% on the restaurant test?")
    
    doc.add_paragraph("Experiment 1 (288x288) relies on Broad Shapes:").bold = True
    doc.add_paragraph("Because the image is blurrier, the model doesn't overthink textures. It sees chunks on a stick and immediately guesses 'Shish Tawook' (100% accuracy). But because it's blurry, it can't see the tiny slivers of meat in a wrapped Shawarma sandwich, so it defaults to guessing 'Bread/Veg' (57.1% accuracy).")
    
    doc.add_paragraph("Experiment 4 (344x344) relies on Fine Details:").bold = True
    doc.add_paragraph("Because the image is sharper, it CAN see the tiny pieces of meat inside the pita wrap! This finally pushed Shawarma to 71.4%. However, because it's looking so closely at textures, it got confused by the specific white garlic sauce on the restaurant Shish Tawook (which didn't perfectly match its training data), causing it to fail (85.7% accuracy).")
    
    # Visuals
    heading(doc, "8. Supplementary Graphs", level=2)
    
    base_dir = os.path.dirname(os.path.abspath(__file__))
    brain_dir = "/Users/mehdidar/.gemini/antigravity-ide/brain/34292ba8-ebe9-423e-8e59-2ef9f6b6a4dd"
    
    doc.add_paragraph("Experiment 4 (344x344) Training Curves:")
    train_curves = os.path.join(base_dir, "..", "results_mobilenet_v3_large_yolo_seg", "training_curves.png")
    if os.path.exists(train_curves):
        doc.add_picture(train_curves, width=Inches(5.5))
        
    doc.add_paragraph("Experiment 4 (344x344) Restaurant Evaluation Confusion Matrix:")
    rest_cm = os.path.join(base_dir, "..", "results", "test_external_mobilenet_v3_large_yolo_seg", "confusion_matrix.png")
    if os.path.exists(rest_cm):
        doc.add_picture(rest_cm, width=Inches(5.0))
    
    doc.add_paragraph("Experiment 4 (344x344) Per-Dish Accuracy:")
    exp4_per_dish = os.path.join(brain_dir, "exp4_per_dish.png")
    if os.path.exists(exp4_per_dish):
        doc.add_picture(exp4_per_dish, width=Inches(5.5))
        
    doc.add_paragraph("Experiment 4 (344x344) Overall Accuracy:")
    exp4_pie = os.path.join(brain_dir, "exp4_pie.png")
    if os.path.exists(exp4_pie):
        doc.add_picture(exp4_pie, width=Inches(5.5))
        
    doc.add_paragraph("Experiment 1 (288x288) Per-Dish Accuracy:")
    exp1_per_dish = os.path.join(brain_dir, "phase9_per_dish.png")
    if os.path.exists(exp1_per_dish):
        doc.add_picture(exp1_per_dish, width=Inches(5.5))
        
    doc.save(OUTPUT_PATH)
    print(f"Generated document at: {OUTPUT_PATH}")

if __name__ == "__main__":
    main()
