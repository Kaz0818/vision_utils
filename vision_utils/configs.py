from dataclasses import dataclass
from typing import Tuple

@dataclass
class TrainConfig:
    data_root: str                    
    img_size: int = 224
    batch_size: int = 32
    num_workers: int = 0
    pin_memory: bool = True
    valid_ratio: float = 0.2
    seed: int = 42

@dataclass
class AugConfig:
    img_size: int = 224
    hflip_p: float = 0.5
    rotate_limit: int = 10
    scale_limit: float = 0.10
    shift_limit: float = 0.05
    color_jitter: Tuple[float, float, float, float] = (0.15, 0.15, 0.15, 0.02)
    color_p: float = 0.3
    mean: Tuple[float, float, float] = (0.485, 0.456, 0.406)
    std:  Tuple[float, float, float] = (0.229, 0.224, 0.225)