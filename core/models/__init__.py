from .efficientnet import EfficientNetClassifier
from .mobilenet import MobileNetClassifier

def FoodClassifier(num_classes=2, freeze_backbone=True, architecture='efficientnet_b2'):
    if 'efficientnet' in architecture:
        return EfficientNetClassifier(num_classes=num_classes, freeze_backbone=freeze_backbone)
    elif 'mobilenet' in architecture:
        return MobileNetClassifier(num_classes=num_classes, freeze_backbone=freeze_backbone)
    else:
        raise ValueError(f"Unsupported architecture: {architecture}")
