# Train MobileNet V3 Large with YOLO Segmentation — Fixed Test Set

## What We're Doing

**New pipeline:**
1. **YOLO segment** the training data (`results/categorized_dataset`) → crop to food region → save to `results/yolo_seg_dataset`
2. **YOLO segment** the test data (`test_dataset_dishes`) → crop → save to `results/yolo_seg_test`
3. **Train** MobileNet V3 Large on `yolo_seg_dataset` (80% train / 20% val split — no test from this data)
4. **Evaluate** on the fully held-out `yolo_seg_test` (your manually added images, never seen during training)

## Key Difference From Previous Runs

| | Old (yolo) | New (this plan) |
|--|--|--|
| Training data | yolo_cropped_dataset (~1203 imgs) | categorized_dataset (~1203 imgs) |
| Test data | 15% of same dataset | **Separate test_dataset_dishes** (538 imgs) |
| Overfitting risk | High (tiny dataset) | Lower (fresh test set) |
| YOLO mode | Detection (bounding box) | **Segmentation mask → tighter crop** |

## Dataset Summary

### Training Data (`results/categorized_dataset`) — ~1,203 images
- **veg** (15 dishes): cheese_pie, hummus, okra_tagine, onion_soup, pistachio_sweet, pistachio_tamriya, pomegranate_fattoush, rice_pudding, saffron_cake, sesame_brittle, spinach_fatayer, spinach_stew, tamees_bread, tamriya, za'atar_manakish
- **non_veg** (11 dishes): grilled_kofta, kofta_stew, mandi, meat-madfoon, meat_maqluba, meat_pie, mixed_grill, shawarma, shish tawook, shrimp_sayadiyah, stuffed_pigeon

### Test Data (`test_dataset_dishes`) — ~538 images
- Same 26 dishes, completely separate images added manually

## Proposed Changes

---

### Step 1: New YOLO Segmentation Script

#### [NEW] [yolo_seg_crop.py](file:///Users/mehdidar/Desktop/food_classifier/scripts/yolo_seg_crop.py)
- Uses `YOLOv8n-seg` (segmentation model) instead of `yolov8n` (detection)
- Extracts the **segmentation mask** polygon → tight bounding box around it
- Falls back to detection box, then center crop if no mask found
- Accepts `--input` and `--output` paths
- Runs on BOTH training data and test data (two separate runs)

---

### Step 2: New Training Script (or flag in train.py)

#### [MODIFY] [train.py](file:///Users/mehdidar/Desktop/food_classifier/scripts/train.py)
- Use **80/20 train/val split** on the YOLO-seg training data (no internal test split — test comes from external folder)
- Results saved to `results_mobilenet_v3_large_yolo_seg/`
- Checkpoint saved as `checkpoints/mobilenet_v3_large_yolo_seg_best.pth`

---

### Step 3: Evaluate on External Test Set

#### [NEW] [evaluate_external.py](file:///Users/mehdidar/Desktop/food_classifier/scripts/evaluate_external.py)
- Loads the trained MobileNet V3 Large checkpoint
- Loads images directly from `results/yolo_seg_test/` (no split_info.json needed)
- Infers class (veg/non_veg) from folder structure
- Outputs: accuracy, precision, recall, F1, confusion matrix, predictions CSV
- Saves results to `results/test_external/`

---

## Run Commands (in order)

```bash
# 1. Segment & crop training data
.venv/bin/python scripts/yolo_seg_crop.py \
    --input  results/categorized_dataset \
    --output results/yolo_seg_dataset \
    --model  yolov8n-seg.pt

# 2. Segment & crop test data
.venv/bin/python scripts/yolo_seg_crop.py \
    --input  test_dataset_dishes \
    --output results/yolo_seg_test \
    --model  yolov8n-seg.pt

# 3. Train MobileNet V3 Large
.venv/bin/python scripts/train.py \
    --dataset results/yolo_seg_dataset \
    --model   mobilenet_v3_large_yolo_seg \
    --img_size 288 \
    --stage1_epochs 15 \
    --stage2_epochs 25

# 4. Evaluate on external test set
.venv/bin/python scripts/evaluate_external.py \
    --model      mobilenet_v3_large_yolo_seg \
    --test_dir   results/yolo_seg_test \
    --img_size   288
```

## Verification Plan

- Check YOLO seg crop output visually on a few images
- Monitor train/val loss curves for overfitting
- Final accuracy reported on the 538-image external test set

## Open Questions

> [!IMPORTANT]
> The `yolov8n-seg.pt` model will be auto-downloaded (~7MB) on first run if not present. Is that OK?
