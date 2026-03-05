import torch.nn as nn
from torchvision import models


def build_model(num_classes=2):
    """Build a ResNet-18 classifier with a 2-class head."""
    model = models.resnet18(weights=None)
    model.fc = nn.Linear(512, num_classes)
    return model
