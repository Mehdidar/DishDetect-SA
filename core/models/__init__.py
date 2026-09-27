from .efficientnet import EfficientNetClassifier
from .mobilenet import MobileNetClassifier

def FoodClassifier(num_classes=2, freeze_backbone=True, architecture='efficientnet_b2'):
    if architecture in ('efficientnet_b2', 'efficientnet_b2_yolo_seg'):
        return EfficientNetClassifier(num_classes=num_classes, freeze_backbone=freeze_backbone)
    elif architecture in ('mobilenet_v3_large', 'mobilenet_v3_large_yolo', 'mobilenet_v3_large_yolo_seg'):
        return MobileNetClassifier(num_classes=num_classes, freeze_backbone=freeze_backbone)
    else:
        raise ValueError(f"Unsupported architecture: {architecture}")
