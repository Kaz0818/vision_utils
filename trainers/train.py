import os
from datetime import datetime
from pathlib import Path
from typing import List, Tuple, Optional, Dict, Any

import torch
import torch.nn as nn
from torch.optim import Optimizer
from torch.optim.lr_scheduler import _LRScheduler
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from zoneinfo import ZoneInfo


class Trainer:
    """
    モデルの訓練、評価、推論を管理するクラス。
    """

    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        criterion: nn.Module,
        optimizer: Optimizer,
        num_epochs: int,
        device: Optional[torch.device] = None,
        scheduler: Optional[_LRScheduler] = None,
        early_stopping_patience: Optional[int] = None,
        log_dir: str = "logs",
        checkpoint_dir: str = "checkpoints",
    ):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.criterion = criterion
        self.optimizer = optimizer
        self.num_epochs = num_epochs
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.scheduler = scheduler
        self.early_stopping_patience = early_stopping_patience

        self.model.to(self.device)

        self.timestamp = datetime.now(ZoneInfo("Asia/Tokyo")).strftime('%Y%m%d_%H%M%S')
        self.model_name = self.model.__class__.__name__
        
        # pathlibを使用してパスを管理
        self.log_dir = Path(log_dir) / f"{self.timestamp}_{self.model_name}"
        self.checkpoint_dir = Path(checkpoint_dir)
        
        self.writer = SummaryWriter(log_dir=str(self.log_dir))
        
        # 結果を格納するリスト
        self.history = {
            'train_loss': [], 'val_loss': [],
            'train_acc': [], 'val_acc': []
        }

    def _train_one_epoch(self) -> Tuple[float, float]:
        """1エポック分の訓練を実行"""
        self.model.train()
        total_loss = 0.0
        total_correct = 0
        total_samples = 0

        for inputs, labels in tqdm(self.train_loader, desc=f"Training Epoch {self.current_epoch+1}/{self.num_epochs}", leave=False):
            inputs, labels = inputs.to(self.device), labels.to(self.device)
            batch_size = inputs.size(0)

            # Forward pass
            outputs = self.model(inputs)
            loss = self.criterion(outputs, labels)

            # Backward and optimize
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            
            # 統計情報の更新
            total_loss += loss.item() * batch_size
            _, predicted = torch.max(outputs.data, 1)
            total_correct += (predicted == labels).sum().item()
            total_samples += batch_size

        avg_loss = total_loss / total_samples
        avg_acc = total_correct / total_samples
        return avg_loss, avg_acc

    def _validate_one_epoch(self) -> Tuple[float, float]:
        """1エポック分の検証を実行"""
        self.model.eval()
        total_loss = 0.0
        total_correct = 0
        total_samples = 0
        
        with torch.no_grad():
            for inputs, labels in tqdm(self.val_loader, desc="Validating", leave=False):
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                batch_size = inputs.size(0)

                outputs = self.model(inputs)
                loss = self.criterion(outputs, labels)

                total_loss += loss.item() * batch_size
                _, predicted = torch.max(outputs.data, 1)
                total_correct += (predicted == labels).sum().item()
                total_samples += batch_size

        avg_loss = total_loss / total_samples
        avg_acc = total_correct / total_samples
        return avg_loss, avg_acc
    
    def _save_checkpoint(self, val_loss: float):
        """モデルのチェックポイントを保存"""
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        state = {
            'epoch': self.current_epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'val_loss': val_loss,
        }
        
        filename = f"{self.timestamp}_{self.model_name}_epoch{self.current_epoch+1}_valloss{val_loss:.4f}.pth"
        filepath = self.checkpoint_dir / filename
        best_filepath = self.checkpoint_dir / "best_model.pth"
        
        torch.save(state, filepath)
        torch.save(state, best_filepath)
        print(f"\n[INFO] Checkpoint saved to: {filepath}")

    def train(self) -> Dict[str, List[float]]:
        """訓練ループ全体を実行"""
        best_val_loss = float('inf')
        epochs_no_improve = 0

        try:
            for epoch in range(self.num_epochs):
                self.current_epoch = epoch
                
                # 訓練と検証
                train_loss, train_acc = self._train_one_epoch()
                val_loss, val_acc = self._validate_one_epoch()

                # 結果の記録
                self.history['train_loss'].append(train_loss)
                self.history['train_acc'].append(train_acc)
                self.history['val_loss'].append(val_loss)
                self.history['val_acc'].append(val_acc)
                
                # TensorBoardへの記録
                self.writer.add_scalar('Loss/train', train_loss, epoch)
                self.writer.add_scalar('Loss/val', val_loss, epoch)
                self.writer.add_scalar('Accuracy/train', train_acc, epoch)
                self.writer.add_scalar('Accuracy/val', val_acc, epoch)

                print(
                    f"Epoch {epoch+1}/{self.num_epochs} | "
                    f"Train Loss: {train_loss:.4f}, Acc: {train_acc:.4f} | "
                    f"Val Loss: {val_loss:.4f}, Acc: {val_acc:.4f}"
                )
                
                # スケジューラの更新
                if self.scheduler:
                    self.scheduler.step()

                # Early Stopping と チェックポイント保存
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    epochs_no_improve = 0
                    self._save_checkpoint(val_loss)
                else:
                    epochs_no_improve += 1

                if self.early_stopping_patience and epochs_no_improve >= self.early_stopping_patience:
                    print(f"\n[INFO] Early stopping triggered after {self.early_stopping_patience} epochs with no improvement.")
                    break
        
        except KeyboardInterrupt:
            print("\n[INFO] Training interrupted by user.")
        finally:
            self.writer.close()
            print("[INFO] TensorBoard writer closed.")

        return self.history

    def predict(
        self,
        dataloader: DataLoader,
        return_probs: bool = False
    ) -> Union[List[int], Tuple[List[int], List[List[float]]]]:
        """データローダーからのデータに対して推論を実行"""
        self.model.eval()
        predictions = []
        probabilities = []
        
        with torch.no_grad():
            for inputs, _ in tqdm(dataloader, desc="Predicting", leave=False):
                inputs = inputs.to(self.device)
                outputs = self.model(inputs)
                
                _, preds = torch.max(outputs, 1)
                predictions.extend(preds.cpu().tolist())
                
                if return_probs:
                    probs = torch.softmax(outputs, dim=1)
                    probabilities.extend(probs.cpu().tolist())

        if return_probs:
            return predictions, probabilities
        return predictions