from pathlib import Path
import numpy as np
from vision_utils import TrainConfig, AugConfig
from vision_utils.transforms import build_transforms
from vision_utils.datasets import make_imagefolder, split_data

def test_smoke_loader():
    cfg = TrainConfig(data_root="/kaggle/input/animals/animals", img_size=128, batch_size=4)
    aug = AugConfig(img_size=cfg.img_size)
    train_tf, valid_tf = build_transforms(aug)
    root = Path(cfg.data_root)
    ds_tr_full, ds_va_full, classes, labels = make_imagefolder(root, train_tf, valid_tf)
    tr_ds, va_ds, tr_dl, va_dl, *_ = split_data(
        ds_tr_full, ds_va_full, labels,
        valid_ratio=cfg.valid_ratio, batch_size=cfg.batch_size,
        num_workers=cfg.num_workers, pin_memory=cfg.pin_memory, seed=cfg.seed
    )
    xb, yb = next(iter(tr_dl))
    assert xb.ndim == 4 and xb.shape[1] == 3 and yb.ndim == 1