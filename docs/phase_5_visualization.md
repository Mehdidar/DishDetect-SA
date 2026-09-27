# Phase 5: Dish-Wise Hierarchical Classifier Visualization

This dashboard visualizes the data flow, architecture, and training results for the final hierarchical classification phase.

## 1. System Architecture Flow
The following flowchart illustrates how raw datasets are processed, segmented, and fed into the deep learning pipeline, ultimately being sorted into a logical hierarchical structure.

```mermaid
graph TD
    A[Raw 1,439 Images<br/>Arabic Dishes Dataset] --> B(Perceptual Hashing<br/>Leakage Prevention)
    B --> C[YOLOv8 Segmentation<br/>Background Removal]
    C --> D[Segmented 26-Class Dataset]
    
    D --> E{MobileNetV3<br/>Fine-Tuned Classifier}
    
    E --> F[Dish Prediction<br/>e.g., 'shawarma']
    
    F --> G(Hierarchical Mapping<br/>dish_to_category.json)
    
    G --> H[Output Folder: non_veg/shawarma/]
    G --> I[Output Folder: veg/falafel/]
    
    style E fill:#1f77b4,stroke:#333,stroke-width:2px,color:#fff
    style C fill:#ff7f0e,stroke:#333,stroke-width:2px,color:#fff
    style H fill:#2ca02c,stroke:#333,stroke-width:2px,color:#fff
    style I fill:#2ca02c,stroke:#333,stroke-width:2px,color:#fff
```

> [!NOTE]
> The **YOLOv8 Segmentation** step is critical for removing noisy backgrounds (like tables or hands), forcing the MobileNetV3 model to learn only the texture and shape of the food itself.

## 2. Model Training Curves
The following graph visualizes the Loss and Accuracy during the 24 epochs of Stage 2 fine-tuning. 

![Phase 5 Training Curves](/Users/mehdidar/.gemini/antigravity-ide/brain/34292ba8-ebe9-423e-8e59-2ef9f6b6a4dd/phase5_training_curves.png)

> [!TIP]
> Notice how the **Validation Loss (orange line)** begins to plateau and slightly rise near Epoch 15-20. The training script automatically triggered **Early Stopping** at Epoch 24 to prevent the model from overfitting to the training data.

## 3. Class Confusion Matrix
This matrix details the model's exact predictions across all 26 classes. The bright yellow diagonal represents correct predictions, while scattered points off the diagonal represent where the model confused one dish for another.

![Phase 5 Confusion Matrix](/Users/mehdidar/.gemini/antigravity-ide/brain/34292ba8-ebe9-423e-8e59-2ef9f6b6a4dd/phase5_confusion_matrix.png)

> [!IMPORTANT]
> The heavy cluster of errors (the lighter squares off the main diagonal) usually occurs between visually indistinguishable dishes like **Mandi**, **Madfoon**, and **Kabsa** — which all consist of roasted meat over spiced rice.
