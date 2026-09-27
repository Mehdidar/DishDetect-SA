import torch
import torch.nn as nn
from torchvision import models

class EfficientNetClassifier(nn.Module):
    def __init__(self, num_classes=2, freeze_backbone=True):
        super(EfficientNetClassifier, self).__init__()
        
        weights = models.EfficientNet_B2_Weights.IMAGENET1K_V1
        self.model = models.efficientnet_b2(weights=weights)
        num_ftrs = self.model.classifier[1].in_features
        self.model.classifier[1] = nn.Linear(num_ftrs, num_classes)
        
        if freeze_backbone:
            self.freeze_backbone()

    def freeze_backbone(self):
        # Freeze all layers
        for param in self.model.parameters():
            param.requires_grad = False
        # Unfreeze only the classifier
        for param in self.model.classifier.parameters():
            param.requires_grad = True
            
    def unfreeze_backbone(self):
        # Unfreeze all layers
        for param in self.model.parameters():
            param.requires_grad = True

    def forward(self, x):
        return self.model(x)
