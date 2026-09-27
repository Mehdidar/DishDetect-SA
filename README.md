# Food Image Classification System

A complete PyTorch-based image classification system using EfficientNet-B2 to classify food images into `veg` and `non_veg`.

## Setup

1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

2. Validate your dataset (IMPORTANT):
   ```bash
   python dataset_check.py --dataset /path/to/dataset
   ```
   This generates `results/dataset_report.json`.

3. Train the model:
   ```bash
   python train.py --dataset /path/to/dataset
   ```
   This will train in two stages (backbone frozen, then unfrozen) and save the best model to `checkpoints/`.

4. Evaluate the model:
   ```bash
   python evaluate.py --dataset /path/to/dataset
   ```
   This saves the classification report and confusion matrix to `results/`.

5. Predict a single image:
   ```bash
   python predict.py --image /path/to/image.jpg --threshold 0.7
   ```
