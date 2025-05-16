
import numpy as np
import json
import os
from torch.utils.data import Subset

def split_dataset(dataset, split_ratio=0.8, seed=42, save_dir="./data/split_indices"):
    """
    dataset: torchvision.datasets.[detaset name:例:FashionMNIST],train=True
    split_ratio:   float:分割する割合
    seed: int: 再現性を保つためこれでtrain_idx, val_idxが同じになる。結果のバラツキ防止
    save_dir: int: saveするpathを指定、ディレクトリなくても関数内で作成.pathは忘れないように

    return : train_loader, val_loader, Subset(datasetからtrain_idx, val_idxが２つreturnされる。torchvision.datasetsと同じ役割) jsonで保存される。使用したindexを確認できる。
    """
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