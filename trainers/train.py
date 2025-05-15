import os
from datetime import datetime
from tqdm import tqdm
import matplotlib.pyplot as plt
import torch
from torchvision import transforms
from torch.utils.tensorboard import SummaryWriter
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from PIL import Image
import io
import numpy as np

from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
class Trainer:
    def __init__(self, train_loader, val_loader, model,
                 criterion, optimizer, num_epochs, device=None,
                 early_stopping=None, scheduler=None, save_path=None,
                 log_dir=None, writer=None):

        self.train_loader = train_loader
        self.val_loader = val_loader
        self.model = model
        self.criterion = criterion
        self.optimizer = optimizer
        self.num_epochs = num_epochs
        self.device = device if device else torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.early_stopping = early_stopping
        self.scheduler = scheduler
        self.save_path = save_path
        self.log_dir = log_dir
        self.writer = writer
        self.model_name = self.model.__class__.__name__
        self.timestamp = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
        self.model.to(self.device)

        if self.writer is None:
            self.log_dir = log_dir or f"logs/{self.timestamp}_{self.model_name}"
            self.writer = SummaryWriter(self.log_dir)
            
    def train(self):
        # --------初期化--------
        self.train_losses, self.train_accuracies = [], []
        self.val_losses, self.val_accuracies = [], []
        best_val_loss = float('inf')
        no_improve = 0

        try:
            for epoch in range(self.num_epochs):
                # --------Training-------
                self.model.train()
                running_train_loss = 0.0
                running_train_correct = 0
                running_train_samples = 0

                for batch_idx, (X, y) in enumerate(tqdm(self.train_loader,
                                                        desc='Train',
                                                        total=len(self.train_loader),
                                                        leave=False)):
                    # X, yをGPUに移動 
                    X, y = X.to(self.device), y.to(self.device)
                    # batch sizeの個数を取得   
                    batch_size = y.size(0)

                    # グラフは1バッチだけ
                    if epoch == 0 and batch_idx == 0:
                        self.writer.add_graph(self.model, X)

                    # forward~backforward
                    self.optimizer.zero_grad()
                    preds = self.model(X)
                    loss = self.criterion(preds, y)
                    loss.backward()
                    self.optimizer.step()

                    # ここの * batch_sizeをなぜしているかわからない
                    running_train_loss += loss.item() * batch_size
                    # accuracyを累積
                    running_train_correct += (preds.argmax(dim=1) == y).sum().item()
                    # batch sizeを累積して平均を取る時にこれを使って割る
                    running_train_samples += batch_size

                # epoch平均(サンプル数で割る)
                avg_train_loss = running_train_loss / running_train_samples
                avg_train_acc = running_train_correct / running_train_samples

                # epoch毎の損失を保管する
                self.train_losses.append(avg_train_loss)
                self.train_accuracies.append(avg_train_acc)
                
                # TensorBoardに表示する
                self.writer.add_scalar('Train Loss', avg_train_loss, epoch)
                self.writer.add_scalar('Train Accuracy', avg_train_acc, epoch)

                # -------Validation----------
                # 検証モードに切り替え
                self.model.eval()
                running_val_loss = 0.0
                running_val_correct = 0
                running_val_samples = 0

                with torch.no_grad():
                    
                    for X_val, y_val in tqdm(self.val_loader,
                                            desc='Validation',
                                            total=len(self.val_loader),
                                            leave=False):
                        
                        # GPUにX_val, y_valを移動する
                        X_val, y_val = X_val.to(self.device), y_val.to(self.device)
                        # batch分で平均を取るのでy_valのbatchの箇所の値を取得
                        batch_size = y_val.size(0)

                        # モデル予測と損失計算まで
                        val_preds = self.model(X_val)
                        val_loss = self.criterion(val_preds, y_val)

                        # ここの * batch_sizeをなぜしているかわからない
                        running_val_loss += val_loss.item() * batch_size
                        running_val_correct += (val_preds.argmax(dim=1) == y_val).sum().item()
                        # 最後batch分で割って平均をだすので保存する
                        running_val_samples += batch_size

                #　現在のepochのbatch　sizeすうの累積で割って平均を出す
                avg_val_loss = running_val_loss / running_val_samples
                avg_val_acc = running_val_correct / running_val_samples

                # returnで返す,val_losses, val_accuracies
                self.val_losses.append(avg_val_loss)
                self.val_accuracies.append(avg_val_acc)
                # Tensorboard に表示するため、epoch毎のloss,accuracyを記録する
                self.writer.add_scalar('Val Loss', avg_val_loss, epoch)
                self.writer.add_scalar('Val Accuracy', avg_val_acc, epoch)

                print(
                    f"Epoch {epoch} | "
                    f"Train Loss: {avg_train_loss:.4f}, Train Acc: {avg_train_acc:.4f} | "
                    f"Val Loss: {avg_val_loss:.4f}, Val Acc: {avg_val_acc:.4f}"
                )

                # ------checkpoint & early stopping & scheduler----
                # 現在のlossが今までのbest_val_lossよりlossが小さければ更新。
                if avg_val_loss < best_val_loss:
                    best_val_loss = avg_val_loss
                    no_improve = 0
                    # best modelの情報をfileを作成して保存
                    base_dir = self.save_path or 'checkpoints'
                    os.makedirs(base_dir, exist_ok=True)
                    filename_best = os.path.join(base_dir, 'best_model.pth')
                    # 学習時間、modelの名前やepoch数 val lossの情報を.pthで保存
                    filename_full = os.path.join(
                        base_dir,
                        f'{self.timestamp}_{self.model_name}_epoch{epoch}_vallloss{avg_val_loss:.4f}.pth'
                    )
                    # best modelの重み、optimizer, loss　, epoch　数を登録
                    state = {
                        'model_dict': self.model.state_dict(),
                        'optimizer': self.optimizer.state_dict(),
                        'val_loss': avg_val_loss,
                        'epoch': epoch
                    }
                    # stateの情報を作成したfilepathに保存
                    torch.save(state, filename_best)
                    torch.save(state, filename_full)
                    print(f'[INFO] Saved best modelto: {filename_full}')
                else:
                    # もしもlossが前回より下がらなかったら +1する. early stoppingで5に設定していれば5になったら学習ストップ
                    no_improve += 1

                    # 設定したearly stopping数になったら学習終了
                if self.early_stopping and no_improve >= self.early_stopping:
                    print('Stopping Early')
                    break

                    # schedulerがあれば更新。optimizerのlrをcosine curveなどで変更できるので、プラトーから抜け出せるかも知れない
                if self.scheduler is not None:
                    self.scheduler.step()
        finally:
            # 例外が途中で起きても確実に閉じられる
            self.writer.close()
        # return finallyの外に書く
        return self.train_losses, self.val_losses, self.train_accuracies, self.val_accuracies


    # -----------Test Dataで予測をする-----------------------------
    def predict(self, dataloader, return_probs=False):
        """
        バッチ単位で推論を行う。
        Args:
            dataloader (DataLoader): 推論用DataLoader (バッチ可)
            return_probs (bool): Trueにすると各クラスの確率も返す

        Returns:
            preds (List[int]): 予測ラベル(整数ID)
            probs (list[Tensor], optional): クラス確率(softmax適用後)

        # 例： test_loaderに対して推論: preds= trainer.predict(test_loader)
        # 例: 確率も一緒に返すpreds, probs = trainer.predict(test_loader, return_probs=True)
        """
        self.model.eval()
        preds = []
        probs = []

        with torch.no_grad():
            for X in tqdm(dataloader, desc='Prediction', leave=False):
                if isinstance(X, (list, tuple)):
                    # X = (X, y) -> X[0]、Xだけ取る
                    X = X[0]
                X = X.to(self.device)
                outputs = self.model(X)
                pred_labels = torch.argmax(outputs, dim=1)
                # GPUにあるのでCPUに移動。modelが予測した一番高いデータのindexをリスト化
                # extendの理解がない
                preds.extend(pred_labels.cpu().tolist())
                if return_probs:
                    probs.extend(outputs.softmax(dim=1).cpu().tolist())
        return (preds, probs) if return_probs else preds

        
    # ---------Train Loss VS Validation Loss Plot--------------------------
    def metrics_plot(self):
        fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(12, 5))
        axes[0].plot(self.train_losses, label='Train', lw=3)
        axes[0].plot(self.val_losses, label='Val', lw=3)
        axes[0].set_xlabel('Epoch')
        axes[0].set_ylabel('Loss')
        axes[0].legend()
        axes[0].set_title('Loss Curve')
        axes[1].plot(self.train_accuracies, label='Train Acc', lw=3)
        axes[1].plot(self.val_accuracies, label='Val Acc', lw=3)
        axes[1].set_xlabel('Epoch')
        axes[1].set_ylabel('Accuracy')
        axes[1].legend()
        axes[1].set_title('Accuracy Curve')
        plt.tight_layout()
        plt.show()

    

    # -----------------Confusion Matrix Plot----------------------------                                 
    def plot_confusion_matrix_display(self, model, dataloader, class_names, device=None, normalize=True, cm_save_path=None, epoch=None):
        # deviceを渡してなかったらインスタンス変数のdeviceを使う
        device = device if device else self.device
        model.eval()
        y_true = []
        y_pred = []

        with torch.no_grad():
            for X, y in dataloader:
                X, y = X.to(device), y.to(device)
                # modelが出した値 logits
                outputs = model(X)
                # softmaxなどで確率ではなくてlogitsの各データの最大のindexを取得
                preds = torch.argmax(outputs, dim=1)
                y_true.extend(y.cpu().tolist())
                y_pred.extend(preds.cpu().tolist())

        
        # 混同行列を生成
        cm = confusion_matrix(y_true, y_pred, normalize='true' if normalize else None)
        # ConfusionMatrixDisplayでプロット
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)
        fig, ax = plt.subplots(figsize=(12, 8))
        disp.plot(ax=ax, cmap=plt.cm.Blues, colorbar=True)
        plt.title("Confution Matrix (Normalized)" if normalize else "Confusion Matrix")
        plt.xticks(rotation=45)
        plt.tight_layout()

        # --- TensorBoardに画像として追加 ---
        buf = io.BytesIO()
        plt.savefig(buf, format='png')
        buf.seek(0)
        image = Image.open(buf).convert("RGB")
        image_np = np.array(image)
        image_tensor = torch.tensor(image_np).permute(2, 0, 1).float() / 255.0
        tag = 'Confusion_Matrix_Display'
        step = epoch if epoch is not None else 0
        self.writer.add_image(tag, image_tensor, global_step=step)
    
        # 保存（任意）
        if cm_save_path:
            os.makedirs('results', exist_ok=True)
            image.save(cm_save_path)
            print(f"[INFO] Saved confusion matrix image to {cm_save_path}")
        else:
            plt.show()
    
        plt.close()
        
    # ---------------検証データで間違えたものだけPlotする=------------------------
    def plot_misclassified_images(self, class_names=None, max_images=25):
        
        self.model.eval()
        misclassified_images, misclassified_preds, correct_labels = [], [], []
    
        with torch.no_grad():
            for X_val, y_val in self.val_loader:
                X_val, y_val = X_val.to(self.device), y_val.to(self.device)
                preds = self.model(X_val).argmax(dim=1)
                wrong = preds != y_val
                if wrong.any():
                    misclassified_images.extend(X_val[wrong].cpu())
                    misclassified_preds.extend(preds[wrong].cpu())
                    correct_labels.extend(y_val[wrong].cpu())
                    if len(misclassified_images) >= max_images:
                        break
    
        n_images = min(len(misclassified_images), max_images)
        plt.figure(figsize=(12, 12))
        n_rowcol = int(n_images ** 0.5) + 1
        for i in range(n_images):
            image = misclassified_images[i]
            pred_label = misclassified_preds[i].item()
            true_label = correct_labels[i].item()
            plt.subplot(n_rowcol, n_rowcol, i + 1)
            img = image.permute(1, 2, 0)
            plt.imshow(img.squeeze(), cmap='gray' if img.shape[2] == 1 else None)
            title = f"Pred: {class_names[pred_label] if class_names else pred_label}\nTrue: {class_names[true_label] if class_names else true_label}"
            plt.title(title)
            plt.axis('off')
        plt.tight_layout()
    plt.show() 
                                    
                
        
