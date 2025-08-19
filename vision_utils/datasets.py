from pathlib import Path
from typing import Tuple
from torchvision.datasets import ImageFolder
from torch.utils.data import Subset, DataLoader
from sklearn.model_selection import StratifiedShuffleSplit
import numpy as np

def make_imagefolder(root: Path, train_transform, valid_transform):
    ds_train_full = ImageFolder(root, transform=train_transform)
    ds_valid_full = ImageFolder(root, transform=valid_transform)
    classes = ds_train_full.classes
    labels = np.array(ds_train_full.targets)
    return ds_train_full, ds_valid_full, classes, labels

def split_data(
    ds_full_train, ds_full_valid, labels: np.ndarray,
    *, valid_ratio: float, batch_size: int, num_workers: int,
    pin_memory: bool, seed: int
):
    splitter = StratifiedShuffleSplit(n_splits=1, test_size=valid_ratio, random_state=seed)
    train_indices, val_indices = next(splitter.split(np.arange(len(labels)), labels))

    train_ds = Subset(ds_full_train, train_indices)
    val_ds   = Subset(ds_full_valid,   val_indices)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                              num_workers=num_workers, pin_memory=pin_memory,
                              persistent_workers=(num_workers > 0))
    val_loader   = DataLoader(val_ds, batch_size=batch_size, shuffle=False,
                              num_workers=num_workers, pin_memory=pin_memory,
                              persistent_workers=(num_workers > 0))
    return train_ds, val_ds, train_loader, val_loader, train_indices, val_indices