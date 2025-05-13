from models.model_cnn import SimpleCNN
from models.resnet_finetune import build_resnet18

def get_model(name, num_classes=10):
    if name.lower() == 'simple_cnn':
        return SimpleCNN(num_classes=num_classes)
    elif name.lower() == 'resnet18':
        return build_resnet18(num_classes=num_classes)
    else:
        raise ValueError(f"Unknown model name: {model}")