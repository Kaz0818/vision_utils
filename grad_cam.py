import os
from typing import List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt

from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget

# ---------------------------
# ユーティリティ
# ---------------------------
def is_vit_model(model: nn.Module) -> bool:
    """timmのViTをざっくり判定（patch_embed と blocks を持つ）"""
    m = unwrap_model(model)
    return hasattr(m, "patch_embed") and hasattr(m, "blocks")


def unwrap_model(model: nn.Module) -> nn.Module:
    """DataParallel / DistributedDataParallel を外す"""
    return getattr(model, "module", model)


def vit_reshape_transform(tensor: torch.Tensor, model: nn.Module) -> torch.Tensor:
    """
    ViT用: トークン列 [B, N, C] -> 2次元特徴 [B, C, H, W] へ
    CLSトークンを除外して grid_size から (H, W) を推定
    """
    if tensor.dim() != 3:
        return tensor

    B, N, C = tensor.shape
    m = unwrap_model(model)

    try:
        gh, gw = m.patch_embed.grid_size
    except Exception:
        # grid_sizeが無い場合は N-1 = H*W として平方根から推定
        gh = gw = int((N - 1) ** 0.5)

    x = tensor[:, 1:, :]  # CLS除去
    x = x.reshape(B, gh, gw, C).permute(0, 3, 1, 2).contiguous()  # [B, C, H, W]
    return x


def denormalize_image(tensor: torch.Tensor, mean: List[float], std: List[float]) -> np.ndarray:
    """
    正規化されたTensor [C,H,W] を denormalize して [H,W,3] の numpy(float32, 0-1) に変換
    """
    tensor = tensor.detach().cpu().clone()
    for t, m, s in zip(tensor, mean, std):
        t.mul_(s).add_(m)
    image_np = tensor.permute(1, 2, 0).numpy()
    return np.clip(image_np, 0, 1).astype(np.float32)


def find_last_conv_layer(model: nn.Module) -> Optional[nn.Module]:
    """
    CNNモデルの「最後のConv2d層」を探す（簡易）
    """
    last_conv = None
    for m in unwrap_model(model).modules():
        if isinstance(m, nn.Conv2d):
            last_conv = m
    return last_conv


# ---------------------------
# メインクラス
# ---------------------------
class GradCAMVisualizer:
    """
    Grad-CAM の計算と可視化を行うクラス（ViT/CNN対応、読みやすさ優先のシンプル版）
    - pytorch-grad-cam を内部で利用
    """

    def __init__(
        self,
        model: nn.Module,
        class_names: Optional[List[str]] = None,
        device: Optional[torch.device] = None,
        mean: Optional[List[float]] = None,
        std: Optional[List[float]] = None,
        target_layer: Optional[nn.Module] = None,
    ):
        """
        Args:
            model: 学習済みモデル（PyTorch）
            class_names: クラス名リスト（index -> name）。Noneなら index を使う
            device: 使用デバイス。Noneなら自動判定
            mean, std: 画像正規化の平均/標準偏差（デフォルトはImageNet）
            target_layer: Grad-CAMのターゲット層。Noneなら自動選択
        """
        self.model = model
        self.model.eval()
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)

        self.class_names = class_names
        self.mean = mean or [0.485, 0.456, 0.406]
        self.std = std or [0.229, 0.224, 0.225]

        # ターゲットレイヤの自動選択（ViTなら blocks[-1].norm1, CNNなら最後のConv2d）
        self.is_vit = is_vit_model(self.model)
        self.target_layer = target_layer or self._select_default_target_layer()

        # ViTの場合は reshape_transform を渡す
        reshape = (lambda t: vit_reshape_transform(t, self.model)) if self.is_vit else None

        self.cam = GradCAM(
            model=self.model,
            target_layers=[self.target_layer],
            reshape_transform=reshape,
        )

    def _select_default_target_layer(self) -> nn.Module:
        m = unwrap_model(self.model)

        if self.is_vit:
            # よく使われる候補のうち存在するものを選択
            # 優先: blocks[-1].norm1 -> blocks[-1].mlp.fc2 -> blocks[-1]
            try:
                return m.blocks[-1].norm1
            except Exception:
                pass
            try:
                return m.blocks[-1].mlp.fc2
            except Exception:
                pass
            # 最後のブロックでも可
            return m.blocks[-1]
        else:
            last_conv = find_last_conv_layer(m)
            if last_conv is None:
                raise ValueError("ターゲットレイヤが見つかりませんでした。target_layer を明示指定してください。")
            return last_conv

    # ---------------------------
    # 可視化メソッド
    # ---------------------------
    def plot_random_samples(
        self,
        dataloader: torch.utils.data.DataLoader,
        num_images: int = 16,
        cols: int = 4,
        save_dir: Optional[str] = None,
        cmap_title: bool = True,
    ):
        """
        検証用データローダから先頭バッチを取り出してランダムに num_images 枚可視化
        """
        print(f"ランダムな{num_images}枚の画像でGrad-CAMを可視化します...")

        # 先頭バッチを使う（シンプルに）
        batch = next(iter(dataloader))
        images_tensor, labels = batch[0], batch[1]

        images_tensor = images_tensor[:num_images].to(self.device)
        labels = labels[:num_images]

        with torch.no_grad():
            preds = self.model(images_tensor).argmax(dim=1)

        visualized_images, titles = [], []
        for i in range(len(images_tensor)):
            input_tensor = images_tensor[i].unsqueeze(0)

            # 予測クラスに対してCAMを計算
            pred_idx = preds[i].item()
            targets = [ClassifierOutputTarget(pred_idx)]

            grayscale_cam = self.cam(input_tensor=input_tensor, targets=targets)[0, :]

            rgb_img = denormalize_image(images_tensor[i], self.mean, self.std)
            visualization = show_cam_on_image(rgb_img, grayscale_cam, use_rgb=True)

            true_name = self._idx_to_name(labels[i].item())
            pred_name = self._idx_to_name(pred_idx)
            title = f"True: {true_name}\nPred: {pred_name}"

            visualized_images.append(visualization)
            titles.append(title)

        self._plot_grid(visualized_images, titles, cols=cols, save_dir=save_dir, prefix="random", cmap_title=cmap_title)

    def plot_misclassified_samples(
        self,
        dataloader: torch.utils.data.DataLoader,
        max_images: int = 16,
        cols: int = 4,
        save_dir: Optional[str] = None,
        cmap_title: bool = True,
    ):
        """
        誤分類した画像を探して最大 max_images 枚を可視化
        """
        print(f"誤分類した画像を最大{max_images}枚探し、Grad-CAMを可視化します...")

        mis_images, titles = [], []
        found = 0

        for images_tensor, labels in dataloader:
            if found >= max_images:
                break

            images_tensor = images_tensor.to(self.device)
            labels = labels.to(self.device)

            with torch.no_grad():
                preds = self.model(images_tensor).argmax(dim=1)

            mis_idx = torch.where(preds != labels)[0]
            for idx in mis_idx:
                if found >= max_images:
                    break

                input_tensor = images_tensor[idx].unsqueeze(0)
                pred_idx = preds[idx].item()
                targets = [ClassifierOutputTarget(pred_idx)]

                grayscale_cam = self.cam(input_tensor=input_tensor, targets=targets)[0, :]

                rgb_img = denormalize_image(images_tensor[idx], self.mean, self.std)
                visualization = show_cam_on_image(rgb_img, grayscale_cam, use_rgb=True)

                true_name = self._idx_to_name(labels[idx].item())
                pred_name = self._idx_to_name(pred_idx)
                title = f"True: {true_name}\nPred: {pred_name}"

                mis_images.append(visualization)
                titles.append(title)
                found += 1

        if not mis_images:
            print("誤分類された画像は見つかりませんでした。")
            return

        self._plot_grid(mis_images, titles, cols=cols, save_dir=save_dir, prefix="miscls", cmap_title=cmap_title)

    # ---------------------------
    # 内部ヘルパー
    # ---------------------------
    def _plot_grid(
        self,
        images: List[np.ndarray],
        titles: List[str],
        cols: int = 4,
        save_dir: Optional[str] = None,
        prefix: str = "cam",
        cmap_title: bool = True,
    ):
        rows = (len(images) + cols - 1) // cols
        plt.figure(figsize=(cols * 4, rows * 4))

        for i, (img, title) in enumerate(zip(images, titles)):
            ax = plt.subplot(rows, cols, i + 1)
            ax.imshow(img)
            if cmap_title:
                ax.set_title(title, fontsize=10)
            ax.axis("off")

        plt.tight_layout()
        plt.show()

        if save_dir is not None:
            os.makedirs(save_dir, exist_ok=True)
            for i, (img, title) in enumerate(zip(images, titles)):
                save_path = os.path.join(save_dir, f"{prefix}_{i:03d}.jpg")
                # 画像は uint8 BGR でなく RGB のまま保存（matplotlib では RGB でOK）
                plt.imsave(save_path, img)
            print(f"画像を保存しました: {save_dir}")

    def _idx_to_name(self, idx: int) -> str:
        if self.class_names is None:
            return str(idx)
        if 0 <= idx < len(self.class_names):
            return self.class_names[idx]
        return str(idx)



# 使い方
# 例: クラス名（あれば）
# class_names = ["cat", "dog", "bird", "..."]
# モデルとデータローダ（例）
# model = ...
# val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, num_workers=2)


# Visualizer を作成（target_layerは省略可。自動選択されます）
# visualizer = GradCAMVisualizer(
#     model=model,
#     class_names=class_names,
#     device=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
#     # mean/std を変えているならここで指定
#     # mean=[0.485, 0.456, 0.406],
#     # std=[0.229, 0.224, 0.225],
#     # target_layer=model.blocks[-1].norm1  # timm ViT の場合は明示もOK
# )

# # ランダムサンプル（先頭バッチから）を可視化
# visualizer.plot_random_samples(
#     dataloader=val_loader,
#     num_images=16,
#     cols=4,
#     save_dir=None,  # 保存したい場合は "./cams_random" などを指定
# )

# # 誤分類サンプルを可視化
# visualizer.plot_misclassified_samples(
#     dataloader=val_loader,
#     max_images=16,
#     cols=4,
#     save_dir=None,  # 保存したい場合は "./cams_miscls" などを指定
# )