import torch.nn as nn
import torchvision.models as models


def get_model(num_classes=2, dropout=0.5):
    model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
    in_feats = model.fc.in_features
    model.fc = nn.Sequential(
        nn.Dropout(dropout),
        nn.Linear(in_feats, num_classes)
    )
    return model