import torch
import torch.nn as nn
from torchvision import models

class VeganFoodClassifier(nn.Module):
    """
    MobileNetV3-Large classifier tailored for discriminating between:
        - 0: vegan (100% plant-based)
        - 1: vegetarian (contains dairy, eggs, butter, or ghee)
    """
    def __init__(self, num_classes=2, freeze_backbone=True):
        super(VeganFoodClassifier, self).__init__()
        weights = models.MobileNet_V3_Large_Weights.IMAGENET1K_V1
        self.model = models.mobilenet_v3_large(weights=weights)
        num_ftrs = self.model.classifier[3].in_features
        self.model.classifier[3] = nn.Linear(num_ftrs, num_classes)
        
        if freeze_backbone:
            self.freeze_backbone()

    def freeze_backbone(self):
        for param in self.model.parameters():
            param.requires_grad = False
        for param in self.model.classifier.parameters():
            param.requires_grad = True

    def unfreeze_backbone(self):
        for param in self.model.parameters():
            param.requires_grad = True

    def forward(self, x):
        return self.model(x)
