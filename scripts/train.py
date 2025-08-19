# scripts/train.py
from __future__ import annotations
from pathlib import Path
import argparse
import json
import random
import numpy as np
import torch
import torch.nn as nn
import torchvision.models as tvm
from PIL import Image, ImageDraw

# 自作ライブラリ（vision_utils/）から読み込み
from vision_utils import (
    TrainConfig, AugConfig,
    build_transforms, make_imagefolder, split_data,
)

# --------- 小ユーティリティ ---------
def save_json(path: Path, obj: dict):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False))

def make_toy_imagefolder(root: Path, classes=("cat", "dog"),
                         samples_per_class=24, img_size=224, seed=42):
    """クラスごとにランダム画像を生成して ImageFolder 形式を自動作成"""
    random.seed(seed); np.random.seed(seed)
    root.mkdir(parents=True, exist_ok=True)
    for cname in classes:
        cdir = root / cname
        cdir.mkdir(parents=True, exist_ok=True)
        for i in range(samples_per_class):
            img = Image.new("RGB", (img_size, img_size), color=(
                random.randint(0, 255), random.randint(0, 255), random.randint(0, 255)
            ))
            draw = ImageDraw.Draw(img)
            # ちょい模様
            for _ in range(10):
                x0, y0 = random.randint(0, img_size - 10), random.randint(0, img_size - 10)
                x1, y1 = x0 + random.randint(5, 40), y0 + random.randint(5, 40)
                draw.rectangle([x0, y0, x1, y1], outline=(255, 255, 255))
            img.save(cdir / f"{cname}_{i:03d}.png")

def run_one_epoch(model, loader, criterion, optimizer, device, train: bool):
    if train:
        model.train()
    else:
        model.eval()
    total_loss, total, correct = 0.0, 0, 0
    for xb, yb in loader:
        xb, yb = xb.to(device), yb.to(device)
        if train:
            optimizer.zero_grad(set_to_none=True)
        with torch.set_grad_enabled(train):
            logits = model(xb)
            loss = criterion(logits, yb)
            if train:
                loss.backward()
                optimizer.step()
        total_loss += loss.item() * xb.size(0)
        pred = logits.argmax(1)
        correct += (pred == yb).sum().item()
        total += xb.size(0)
    return total_loss / max(total, 1), correct / max(total, 1)

# --------- メイン ---------
def main():
    # 1) 引数
    parser = argparse.ArgumentParser(description="Smoke-ready training pipeline")
    parser.add_argument("--data_root", type=str, required=True,
                        help="クラス名フォルダが並ぶ親ディレクトリ（ImageFolder想定）")
    parser.add_argument("--img_size", type=int, default=224)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--valid_ratio", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--make_toy", action="store_true",
                        help="指定先にcat/dogの玩具データセットを自動生成してから実行")
    args = parser.parse_args()

    # 2) CFG 構築
    train_cfg = TrainConfig(
        data_root=args.data_root,
        img_size=args.img_size,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=True,
        valid_ratio=args.valid_ratio,
        seed=args.seed,
    )
    aug_cfg = AugConfig(img_size=train_cfg.img_size)

    # 3) データルート検証＋必要なら玩具データ生成
    root = Path(train_cfg.data_root)
    if args.make_toy:
        print(f"[info] Generating toy dataset under: {root}")
        make_toy_imagefolder(root, classes=("cat", "dog"),
                             samples_per_class=24, img_size=train_cfg.img_size, seed=train_cfg.seed)
    if not root.exists():
        raise FileNotFoundError(f"--data_root not found: {root}")
    class_dirs = [p for p in root.iterdir() if p.is_dir()]
    if not class_dirs:
        raise RuntimeError(f"No class folders under {root}. e.g. {root/'class_a'}, {root/'class_b'}")

    # 4) 前処理（Albumentations）
    # ＊警告が気になる場合は ShiftScaleRotate を A.Affine に置換してOK
    train_tf, valid_tf = build_transforms(aug_cfg)

    # 5) 同じ ROOT から transform だけ違う ImageFolder を2つ作成
    ds_tr_full, ds_va_full, classes, labels = make_imagefolder(root, train_tf, valid_tf)

    # 6) StratifiedShuffleSplit → Subset → DataLoader
    tr_ds, va_ds, tr_dl, va_dl, tr_idx, va_idx = split_data(
        ds_tr_full, ds_va_full, labels,
        valid_ratio=train_cfg.valid_ratio,
        batch_size=train_cfg.batch_size,
        num_workers=train_cfg.num_workers,
        pin_memory=train_cfg.pin_memory,
        seed=train_cfg.seed,
    )

    # 7) 使った設定と分割を保存（再現性）
    runs_dir = Path("runs") / "smoke"
    save_json(runs_dir / "config.json", {
        "train_cfg": train_cfg.__dict__,
        "aug_cfg": aug_cfg.__dict__,
        "num_classes": len(classes),
        "classes": classes,
    })
    Path("splits").mkdir(parents=True, exist_ok=True)
    np.save(Path("splits") / "train_idx.npy", tr_idx)
    np.save(Path("splits") / "val_idx.npy", va_idx)

    # 8) スモーク：1バッチ取り出し
    xb, yb = next(iter(tr_dl))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"OK: batch={tuple(xb.shape)} labels={tuple(yb.shape)} classes={len(classes)} device={device.type}")

    # 9) モデル定義（ResNet18）＋学習ループ
    num_classes = len(classes)
    model = tvm.resnet18(weights=tvm.ResNet18_Weights.IMAGENET1K_V1)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)

    for epoch in range(3):
        tr_loss, tr_acc = run_one_epoch(model, tr_dl, criterion, optimizer, device, train=True)
        va_loss, va_acc = run_one_epoch(model, va_dl, criterion, optimizer, device, train=False)
        print(f"[{epoch+1}] train loss={tr_loss:.4f} acc={tr_acc:.3f} | "
              f"val loss={va_loss:.4f} acc={va_acc:.3f}")

    # 10) モデル保存
    out = Path("runs") / "smoke"
    out.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), out / "model.pt")
    print(f"[info] saved model to {out / 'model.pt'}")

if __name__ == "__main__":
    main()