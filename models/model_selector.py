import torch.nn as nn
import torchvision

def load_model(in_channels: int, model_name: str, num_classes: int, pretrained: bool = True, feature_extract: bool = False):
    model_name = model_name.lower()
    
    if model_name == 'resnet18':
        model = torchvision.models.resnet18(weights='DEFAULT' if pretrained else None)
        # 入力チャンネルが3以外の場合のみ、最初のConvを上書き
        if in_channels != 3:
            model.conv1 = nn.Conv2d(in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)
        # 最終層の置き換え
        model.fc = nn.Linear(model.fc.in_features, num_classes)
        # 転移学習モード（最終層だけ学習）
        if feature_extract:
            for param in model.parameters():
                param.requires_grad = False
            model.fc = nn.Linear(model.fc.in_features, num_classes)  # ←この書き換えでfcだけrequires_grad=Trueになる

    elif model_name == 'vgg16':
        model = torchvision.models.vgg16(weights='DEFAULT' if pretrained else None)
        if in_channels != 3:
            model.features[0] = nn.Conv2d(in_channels, 64, kernel_size=3, padding=1)
        model.classifier[6] = nn.Linear(model.classifier[6].in_features, num_classes)
        if feature_extract:
            for param in model.parameters():
                param.requires_grad = False
            model.classifier[6] = nn.Linear(model.classifier[6].in_features, num_classes)  # ここでclassifier

    else:
        raise ValueError(f"Unknown model name: {model_name}")

    return model
