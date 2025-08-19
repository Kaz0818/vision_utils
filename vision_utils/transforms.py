import albumentations as A
from albumentations.pytorch import ToTensorV2
import cv2, numpy as np
from PIL import Image
from .configs import AugConfig

class AlbuAdapter:
    def __init__(self, aug: A.BasicTransform):
        self.aug = aug
    def __call__(self, img: Image.Image):
        x = np.array(img)
        if x.ndim == 2:
            x = np.stack([x, x, x], axis=-1)
        x = np.ascontiguousarray(x[..., :3])
        return self.aug(image=x)["image"]

def build_transforms(cfg: AugConfig):
    train_tf = A.Compose([
        A.Resize(cfg.img_size, cfg.img_size),
        A.HorizontalFlip(p=cfg.hflip_p),
        A.ShiftScaleRotate(
            shift_limit=cfg.shift_limit,
            scale_limit=cfg.scale_limit,
            rotate_limit=cfg.rotate_limit,
            border_mode=cv2.BORDER_REFLECT101,
            p=0.5
        ),
        A.ColorJitter(*cfg.color_jitter, p=cfg.color_p),
        A.Normalize(mean=cfg.mean, std=cfg.std),
        ToTensorV2(),
    ])
    valid_tf = A.Compose([
        A.Resize(cfg.img_size, cfg.img_size),
        A.Normalize(mean=cfg.mean, std=cfg.std),
        ToTensorV2(),
    ])
    return AlbuAdapter(train_tf), AlbuAdapter(valid_tf)