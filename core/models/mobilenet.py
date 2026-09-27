import torch
import torch.nn as nn
from torchvision import models

class MobileNetClassifier(nn.Module):
    def __init__(self, num_classes=2, freeze_backbone=True):
        super(MobileNetClassifier, self).__init__()
        
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
