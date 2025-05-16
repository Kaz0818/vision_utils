from vision_utils.models.model_cnn import SimpleCNN
from vision_utils.models.resnet_finetune import build_resnet18

def get_model(name, num_classes=10):
    """
    name: str: configs/config.yamlに書いてある。nameを入れる。例: 'simple_cnn'

    num_classes: int: outputのクラス数を指定。
    """
    if name.lower() == 'simple_cnn':
        return SimpleCNN(num_classes=num_classes)
    elif name.lower() == 'resnet18':
        return build_resnet18(num_classes=num_classes)
    else:
        raise ValueError(f"Unknown model name: {model}")