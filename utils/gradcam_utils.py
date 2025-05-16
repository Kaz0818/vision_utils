# fashion_mnist_utils/utils/gradcam_utils.py

import torch
import numpy as np
import cv2
import matplotlib.pyplot as plt

def apply_gradcam(model, image_tensor, target_layer, class_idx=None, device="cpu"):
    model.eval()
    image_tensor = image_tensor.unsqueeze(0).to(device)  # [1, 1, 28, 28]
    
    features = []
    grads = []

    def forward_hook(module, input, output):
        features.append(output.detach())

    def backward_hook(module, grad_input, grad_output):
        grads.append(grad_output[0].detach())

    # hook 登録
    handle_f = target_layer.register_forward_hook(forward_hook)
    handle_b = target_layer.register_backward_hook(backward_hook)

    # 予測と逆伝播
    output = model(image_tensor)
    if class_idx is None:
        class_idx = torch.argmax(output, dim=1).item()
    loss = output[0, class_idx]
    model.zero_grad()
    loss.backward()

    # hook解除
    handle_f.remove()
    handle_b.remove()

    # Grad-CAM 計算
    feature_map = features[0][0]  # [C, H, W]
    gradient = grads[0][0]        # [C, H, W]

    weights = torch.mean(gradient, dim=(1, 2))  # グローバル平均プーリング
    cam = torch.zeros(feature_map.shape[1:], dtype=torch.float32).to(device)

    for i, w in enumerate(weights):
        cam += w * feature_map[i]

    cam = torch.relu(cam)
    cam -= cam.min()
    if cam.max() != 0:
        cam /= cam.max()
    else:
        cam = torch.zeros_like(cam)  # fallbackとして全部ゼロにする
    cam = cam.cpu().numpy()
    cam = cv2.resize(cam, (28, 28))
    return cam, class_idx


def show_batch_gradcam(
    model, images, labels, target_layer,
    device="cpu", nrow=8, alpha=0.4, save_path=None, apply_gradcam_fn=None,
    class_names=None
):
    """
        バッチ画像にGrad-CAMを適用してグリッド表示＋（任意で保存）

        Args:
            model: PyTorchモデル
            images: [B, C, H, W] バッチ画像Tensor
            labels: ラベルTensor or list
            target_layer: Grad-CAM対象層 (例: model.layer4[-1])
            device: "cpu" or "cuda"
            nrow: グリッド1行の画像数
            alpha: ヒートマップ重ね透明度
            save_path: 画像保存先 (Noneなら保存しない)
            apply_gradcam_fn: apply_gradcam関数（引数：model, img, target_layer, device）

        Returns:
            なし
        """
    model.eval()
    num_imgs = images.shape[0]
    ncol = (num_imgs + nrow - 1) // nrow  # 必要な行数だけ計算

    fig, axes = plt.subplots(ncol, nrow, figsize=(nrow * 3, ncol * 3))
    axes = axes.flatten()  # 2次元でも1次元でも使える

    for idx in range(num_imgs):
        img = images[idx].to(device)
    
        # ----- GT（正解）の名前と番号 -----
        gt_id = labels[idx].item() if torch.is_tensor(labels[idx]) else int(labels[idx])
        gt_name = class_names[gt_id] if class_names else str(gt_id)
    
        # ----- 予測クラス名と番号 -----
        cam, pred_id = apply_gradcam_fn(model, img, target_layer=target_layer, device=device)
        pred_name = class_names[pred_id] if class_names else str(pred_id)
    
        # ----- 画像合成 -----
        img_np = img.detach().cpu().permute(1, 2, 0).numpy()
        img_np = (img_np * 255).astype(np.uint8)
        if img_np.shape[2] == 1:
            img_np = np.repeat(img_np, 3, axis=2)
    
        cam = cv2.resize(cam, (img_np.shape[1], img_np.shape[0]))
        heatmap = cv2.applyColorMap(np.uint8(255 * cam), cv2.COLORMAP_JET)
        heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
        overlay = cv2.addWeighted(heatmap, 0.4, img_np, 0.6, 0)
    
        ax = axes[idx]
        # ----- タイトルを希望通りに -----
        title = f"GT: {gt_name}|{gt_id} / Pred: {pred_name}|{pred_id}"
        ax.imshow(overlay)
        ax.set_title(title, fontsize=10)
        ax.axis('off')

    # 余白のaxesは消す
    for i in range(num_imgs, len(axes)):
        axes[i].axis('off')

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path)
    plt.show()