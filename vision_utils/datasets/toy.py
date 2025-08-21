import random
from pathlib import Path
from PIL import Image, ImageDraw
import numpy as np

def make_toy_imagefolder(root: Path, classes=("cat", "dog"),
                         samples_per_class=24, img_size=224, seed=42):
    """クラスごとにランダム画像を生成して ImageFolder 形式を自動作成"""
    random.seed(seed); np.random.seed(seed)
    root.mkdir(parents=True, exist_ok=True)
    for class_name in classes:
        class_dir = root / class_name
        class_dir.mkdir(parents=True, exist_ok=True)
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
            img.save(class_dir / f"{class_name}_{i:03d}.png")