# fashion_mnist_utils/models/resnet_finetune.py

import torch.nn as nn
from torchvision import models

def build_resnet18(num_classes=10, pretrained=True):
    model = models.resnet18(weights="DEFAULT" if pretrained else None)
    
    # 入力チャネルが1の場合（FashionMNISTなど）
    if model.conv1.in_channels != 1:
        model.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)

    # 最終層を置き換え
    in_features = model.fc.in_features
    model.fc = nn.Linear(in_features, num_classes)
    return model