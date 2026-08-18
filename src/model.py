"""
model.py
ImageNet üzerinde ön eğitilmiş ResNet18 kullanarak transfer learning modeli.
"""

import torch.nn as nn
from torchvision import models


def build_model(num_classes=3, freeze_backbone=True, pretrained=True):
    """ResNet18 tabanlı sınıflandırma modeli oluşturur.

    Args:
        num_classes: Çıkış sınıf sayısı (varsayılan 3: Malignant/Benign/No Tumor)
        freeze_backbone: True ise sadece son katman eğitilir (hızlı, az veriyle iyi çalışır)
        pretrained: True ise ImageNet ön eğitimli ağırlıklar indirilir (eğitim için gerekli).
            Kendi eğitilmiş ağırlıklarınızı (best_model.pth) yükleyecekseniz False verin;
            bu ağırlıklar zaten üzerine yazılacağı için indirme gereksizdir.
    """
    weights = models.ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
    model = models.resnet18(weights=weights)

    if freeze_backbone:
        for param in model.parameters():
            param.requires_grad = False

    in_features = model.fc.in_features
    model.fc = nn.Sequential(
        nn.Linear(in_features, 128),
        nn.ReLU(),
        nn.Dropout(0.3),
        nn.Linear(128, num_classes),
    )

    return model
