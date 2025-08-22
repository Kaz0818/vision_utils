import os
import io
import numpy as np
import matplotlib.pyplot as plt
import torch
from PIL import Image
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, classification_report
from tqdm import tqdm # データローダーをループするなら tqdm もあると便利

class Visualizer: 
    def __init__(self, writer=None):
        self.writer = writer
        
# ---------Train Loss VS Validation Loss Plot--------------------------
    def metrics_plot(self, train_losses, val_losses, train_accuracies, val_accuracies):
        fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(12, 5))
        axes[0].plot(train_losses, label='Train', lw=3)
        axes[0].plot(val_losses, label='Val', lw=3)
        axes[0].set_xlabel('Epoch')
        axes[0].set_ylabel('Loss')
        axes[0].legend()
        axes[0].set_title('Loss Curve')
        axes[1].plot(train_accuracies, label='Train Acc', lw=3)
        axes[1].plot(val_accuracies, label='Val Acc', lw=3)
        axes[1].set_xlabel('Epoch')
        axes[1].set_ylabel('Accuracy')
        axes[1].legend()
        axes[1].set_title('Accuracy Curve')
        plt.tight_layout()
        plt.show()

    
      
    
    # -----------------Confusion Matrix Plot----------------------------                                 
    # インデントを修正して、他のメソッドと同じレベルにする
    def plot_confusion_matrix_display(self, model, dataloader, class_names, device,
                                    normalize=True, cm_save_path=None, epoch=None):
        
        model.to(device) # <--- model を device に送るのを追加
        model.eval()
        y_true = []
        y_pred = []

        with torch.no_grad():
            for X, y in tqdm(dataloader, desc="Generating CM", leave=False):
                X, y = X.to(device), y.to(device)
                outputs = model(X)
                preds = torch.argmax(outputs, dim=1)
                y_true.extend(y.cpu().tolist())
                y_pred.extend(preds.cpu().tolist())

        
        cm = confusion_matrix(y_true, y_pred, normalize='true' if normalize else None)
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)
        fig, ax = plt.subplots(figsize=(12, 8))
        disp.plot(ax=ax, cmap=plt.cm.Blues, colorbar=True)
        # タイポを修正
        plt.title("Confusion Matrix (Normalized)" if normalize else "Confusion Matrix")
        plt.xticks(rotation=45)
        plt.tight_layout()

        # --- TensorBoardに画像として追加 ---
        if self.writer:
            buf = io.BytesIO()
            plt.savefig(buf, format='png')
            buf.seek(0)
            image = Image.open(buf).convert("RGB")
            image_np = np.array(image)
            image_tensor = torch.tensor(image_np).permute(2, 0, 1).float() / 255.0
            tag = 'Confusion_Matrix_Display'
            step = epoch if epoch is not None else 0
            self.writer.add_image(tag, image_tensor, global_step=step)
        
        if cm_save_path:
            os.makedirs('results', exist_ok=True)
            # image変数はTensorBoardの処理の中で作られるので、
            # self.writerがない場合でも作られるように調整するか、
            # 保存時にも作る必要があります。ここでは保存時にも作るようにします。
            if not self.writer: 
                buf = io.BytesIO()
                plt.savefig(buf, format='png')
                buf.seek(0)
                image = Image.open(buf).convert("RGB")

            image.save(cm_save_path)
            print(f"[INFO] Saved confusion matrix image to {cm_save_path}")
        else:
            plt.show()
    
        plt.close()
        
    # ---------------検証データで間違えたものだけPlotする=------------------------
    def plot_misclassified_images(self, model, dataloader, class_names, device, # model, dataloader, device を追加
                                  max_images=25):
        model.to(device)
        model.eval()
        misclassified_images, misclassified_preds, correct_labels = [], [], []
    
        with torch.no_grad():
            # tqdm を追加, dataloader を使用
            for X_val, y_val in tqdm(dataloader, desc="Finding Misclassified", leave=False):
                X_val, y_val = X_val.to(device), y_val.to(device)
                preds = model(X_val).argmax(dim=1)
                wrong = preds != y_val
                if wrong.any():
                    misclassified_images.extend(X_val[wrong].cpu())
                    misclassified_preds.extend(preds[wrong].cpu())
                    correct_labels.extend(y_val[wrong].cpu())
                    if len(misclassified_images) >= max_images:
                        break
    
        n_images = min(len(misclassified_images), max_images)
        if n_images == 0:
            print("[INFO] No misclassified images found.")
            return # 間違えた画像がなければ終了
            
        plt.figure(figsize=(12, 12))
        n_rowcol = int(n_images ** 0.5) + 1
        for i in range(n_images):
            image = misclassified_images[i]
            pred_label = misclassified_preds[i].item()
            true_label = correct_labels[i].item()
            plt.subplot(n_rowcol, n_rowcol, i + 1)
            img = image.permute(1, 2, 0)
            # 標準化を戻す処理などが必要ならここに入れる
            plt.imshow(img.squeeze(), cmap='gray' if img.shape[2] == 1 else None)
            title = f"Pred: {class_names[pred_label] if class_names else pred_label}\nTrue: {class_names[true_label] if class_names else true_label}"
            plt.title(title)
            plt.axis('off')
        plt.tight_layout()
        plt.show() # <- メソッド内に移動
        plt.close() # 表示後には close するのが良い
        

# ---------------Classification Report ------------------------

def result_classification_report(model, data_loader, target_names, device):
    """ 
    Args:
        model: 学習済みのmodel
        data_loader: resultに使うdata
        target_names: class_names
        device: GPUで学習する場合
    
    example: result_classification_report(model=vit, data_loader=val_loader, target_names=class_names, device)
    """
    
    model.eval()  # 評価モード
    all_preds = []
    all_targets = []
    
    with torch.no_grad():
        for X, y in tqdm(data_loader, desc='Validation', total=len(data_loader), leave=False):
            X = X.to(device)
            y = y.to(device)
            logits = model(X)
            preds = logits.argmax(dim=1)  # softmaxは不要
    
            # CPU に戻してバッチごとに蓄積
            all_preds.append(preds.detach().cpu())
            all_targets.append(y.detach().cpu())
    
    # 1本のベクトルに結合して numpy へ
    y_pred = torch.cat(all_preds).numpy()
    y_true = torch.cat(all_targets).numpy()
    
    # クラス名があるなら渡す（len が一致すること）
    # 例: target_names = class_names
    if target_names:
        target_names = target_names
    else:
        target_names = None  # あるなら class_names に置き換え
    print(classification_report(y_true, y_pred, target_names=target_names, digits=4, zero_division=0))

