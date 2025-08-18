from pathlib import Path
import cv2, albumentations as A, torch, random, numpy as np
from albumentations.pytorch import ToTensorV2
from torch.utils.data import Dataset, DataLoader

# ===== 2) クラス辞書（決定論の要） =====
def build_class_maps(root: Path):
    classes = sorted([d.name for d in root.iterdir() if d.is_dir()])
    class_to_id = {c:i for i,c in enumerate(classes)}
    id_to_class = {i:c for c,i in class_to_id.items()}
    return classes, class_to_id, id_to_class

# ===== 3) 画像パス収集（決定論的にソート） =====
ALLOWED_EXT = frozenset({".jpg",".jpeg",".png",".bmp",".webp"})
def collect_image_paths(root: Path, class_to_id: dict):
    paths, targets = [], []
    for cname in sorted(class_to_id):
        cid = class_to_id[cname]
        cdir = root / cname
        for p in sorted((q for q in cdir.rglob("*") if q.is_file()), key=lambda x: x.as_posix()):
            if p.suffix.lower() in ALLOWED_EXT:
                paths.append(p); targets.append(cid)
    return paths, targets

# ===== 4) 変換（理解済みとのこと・最小構成） =====
def build_transforms(img_size=int):
    train_transform = A.Compose([
        A.Resize(img_size, img_size), A.HorizontalFlip(p=0.5),
        A.Normalize(mean=(0.485,0.456,0.406), std=(0.229,0.224,0.225)),
        ToTensorV2(),
    ])
    valid_transform = A.Compose([
        A.Resize(img_size, img_size),
        A.Normalize(mean=(0.485,0.456,0.406), std=(0.229,0.224,0.225)),
        ToTensorV2(),
    ])
    return train_transform, valid_transform

# ===== 5) Dataset =====
class PathDataset(Dataset):
    
    def __init__(self, paths, targets, transform=None):
        self.paths = list(paths)
        self.targets = list(targets)
        self.transform = transform
        
    def __len__(self):
        return len(self.paths)
    
    def __getitem__(self, idx):
        path = self.paths[idx]
        image = cv2.imread(str(path))
        if image is None: raise FileNotFoundError(path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        if self.transform:
            img = self.transform(image=image)['image']
            label = int(self.targets[idx])
        
        return img, label
            

# ===== 6) Loader 構築 ＆ 煙テスト =====
def build_loaders(TRAIN_ROOT, VAL_ROOT, IMG_SIZE, BATCH_SIZE, build_transforms):
    # set_seed()
    classes, class_to_id, id_to_class = build_class_maps(TRAIN_ROOT)
    train_paths, train_targets = collect_image_paths(TRAIN_ROOT, class_to_id)
    val_paths,   val_targets   = collect_image_paths(VAL_ROOT,   class_to_id)
    train_transform, valid_transform = build_transforms(IMG_SIZE)

    train_dataset = PathDataset(train_paths, train_targets, transform=train_transform)
    val_dataset = PathDataset(val_paths, val_targets, transform=valid_transform)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True,  num_workers=0, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0, pin_memory=True)
    
    xb, yb = next(iter(train_loader))
    assert xb.ndim==4 and xb.shape[1]==3 and xb.dtype==torch.float32
    return train_loader, val_loader, classes