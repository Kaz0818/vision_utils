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