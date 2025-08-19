from pathlib import Path
import numpy as np
from vision_utils import TrainConfig, AugConfig
from vision_utils.transforms import build_transforms
from vision_utils.datasets import make_imagefolder, split_data

def main():
    # ★ 最初に CFG を定義（ここから始める運用）
    train_cfg = TrainConfig(
        data_root="/kaggle/input/animals/animals",  # ←ローカルに合わせて書き換え
        img_size=224,
        batch_size=32,
        num_workers=0,
        pin_memory=True,
        valid_ratio=0.2,
        seed=42,
    )
    aug_cfg = AugConfig(img_size=train_cfg.img_size)

    # 前処理
    train_tf, valid_tf = build_transforms(aug_cfg)

    # Dataset 構築（同じROOTから transform違いで2つ）
    root = Path(train_cfg.data_root)
    ds_tr_full, ds_va_full, classes, labels = make_imagefolder(root, train_tf, valid_tf)

    # 層化split → Subset → DataLoader
    train_ds, val_ds, train_loader, val_loader, tr_idx, va_idx = split_data(
        ds_tr_full, ds_va_full, labels,
        valid_ratio=train_cfg.valid_ratio, batch_size=train_cfg.batch_size,
        num_workers=train_cfg.num_workers, pin_memory=train_cfg.pin_memory, seed=train_cfg.seed
    )

    # スモーク（1バッチ確認）
    xb, yb = next(iter(train_loader))
    print("OK:", xb.shape, yb.shape, "num_classes:", len(classes))

if __name__ == "__main__":
    main()