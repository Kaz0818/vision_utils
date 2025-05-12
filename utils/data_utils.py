# fashion_mnist_utils/utils/split_utils.py

import numpy as np
import json
import os
from torch.utils.data import Subset

def split_dataset(dataset, split_ratio=0.8, seed=42, save_dir="./data/split_indices"):
    # 🔴 ここバグ修正：os.makedirs(dataset, exist_ok=True) → save_dir に変更
    os.makedirs(save_dir, exist_ok=True)

    num_samples = len(dataset)
    indices = np.arange(num_samples)
    np.random.seed(seed)
    np.random.shuffle(indices)

    split = int(split_ratio * num_samples)
    train_idx, val_idx = indices[:split], indices[split:]

    # 保存（再現性のため）
    with open(os.path.join(save_dir, "train_idx.json"), "w") as f:
        json.dump(train_idx.tolist(), f)
    with open(os.path.join(save_dir, "val_idx.json"), "w") as f:
        json.dump(val_idx.tolist(), f)

    return Subset(dataset, train_idx), Subset(dataset, val_idx)