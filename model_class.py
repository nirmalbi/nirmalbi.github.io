import torch.nn as nn
from torchvision import models


def build_model(num_classes=2):
    """Build an enhanced ResNet-18 classifier with a 2-class head.

    Modifications (keeping ResNet-18 backbone intact):
    - Replaces the default FC layer with a stronger classification head
      (dropout + linear) for better generalisation.
    """
    model = models.resnet18(weights=None)

    in_features = model.fc.in_features  # 512

    model.fc = nn.Sequential(
        nn.Dropout(p=0.3),
        nn.Linear(in_features, num_classes),
    )

    # Better weight initialisation for the new head
    for m in model.fc.modules():
        if isinstance(m, nn.Linear):
            nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            if m.bias is not None:
                nn.init.zeros_(m.bias)

    return model
