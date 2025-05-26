import logging
import os
from datetime import datetime
from typing import Optional, Tuple, List, Dict, Any, Union
from tqdm import tqdm
import torch
import torch.nn as nn
from torch.optim import Optimizer
from torch.optim.lr_scheduler import _LRScheduler
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter


class TrainerConfig:
    """Trainer設定用の定数クラス"""
    DEFAULT_LOG_DIR = "logs"
    DEFAULT_CHECKPOINT_DIR = "checkpoints"
    BEST_MODEL_FILENAME = "best_model.pth"
    TIMESTAMP_FORMAT = '%Y-%m-%d_%H-%M-%S'


class Trainer:
    """
    深層学習モデルの訓練を管理するクラス
    
    Features:
        - 訓練・検証ループの自動化
        - Early stopping
        - チェックポイント保存
        - TensorBoard logging
        - 学習率スケジューリング
    """
    
    def __init__(self, 
                 train_loader: DataLoader,
                 val_loader: DataLoader,
                 model: nn.Module,
                 criterion: nn.Module,
                 optimizer: Optimizer,
                 num_epochs: int,
                 device: Optional[torch.device] = None,
                 early_stopping: Optional[int] = None,
                 scheduler: Optional[_LRScheduler] = None,
                 save_path: Optional[str] = None,
                 log_dir: Optional[str] = None,
                 writer: Optional[SummaryWriter] = None,
                 verbose: bool = True):
        """
        Trainerクラスの初期化
        
        Args:
            train_loader: 訓練用DataLoader
            val_loader: 検証用DataLoader
            model: 訓練するモデル
            criterion: 損失関数
            optimizer: 最適化アルゴリズム
            num_epochs: エポック数
            device: 使用デバイス（None時は自動選択）
            early_stopping: Early stopping用の待機エポック数
            scheduler: 学習率スケジューラー
            save_path: モデル保存パス
            log_dir: ログディレクトリ
            writer: TensorBoardWriter（None時は自動作成）
            verbose: 詳細な出力を行うかどうか
        """
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
        self.verbose = verbose
        
        # メタデータの設定
        self.model_name = self.model.__class__.__name__
        self.timestamp = datetime.now().strftime(TrainerConfig.TIMESTAMP_FORMAT)
        
        # モデルをデバイスに移動
        self.model.to(self.device)
        
        # TensorBoard writer の設定
        self.log_dir = log_dir or f"{TrainerConfig.DEFAULT_LOG_DIR}/{self.timestamp}_{self.model_name}"
        self.writer = writer or SummaryWriter(self.log_dir)
        
        # ロギング設定
        self._setup_logging()
        
        # 訓練状態の初期化
        self._initialize_training_state()
    
    def _setup_logging(self) -> None:
        """ロギングの設定"""
        logging.basicConfig(
            level=logging.INFO if self.verbose else logging.WARNING,
            format='%(asctime)s - %(levelname)s - %(message)s'
        )
        self.logger = logging.getLogger(__name__)
    
    def _initialize_training_state(self) -> None:
        """訓練状態の初期化"""
        self.train_losses: List[float] = []
        self.train_accuracies: List[float] = []
        self.val_losses: List[float] = []
        self.val_accuracies: List[float] = []
        self.best_val_loss = float('inf')
        self.no_improve = 0
        self.current_epoch = 0
    
    def train(self) -> Tuple[List[float], List[float], List[float], List[float]]:
        """
        モデルの訓練を実行
        
        Returns:
            訓練損失、検証損失、訓練精度、検証精度のタプル
        """
        try:
            self.logger.info(f"Starting training on device: {self.device}")
            self.logger.info(f"Model: {self.model_name}, Epochs: {self.num_epochs}")
            
            return self._training_loop()
            
        except KeyboardInterrupt:
            self.logger.info("Training interrupted by user")
            return self._get_current_metrics()
            
        except Exception as e:
            self.logger.error(f"Training failed: {e}")
            raise
            
        finally:
            self.writer.close()
            self.logger.info("Training completed")
    
    def _training_loop(self) -> Tuple[List[float], List[float], List[float], List[float]]:
        """メインの訓練ループ"""
        for epoch in range(self.num_epochs):
            self.current_epoch = epoch
            
            # 1エポックの訓練と検証
            train_metrics = self._train_epoch(epoch)
            val_metrics = self._validate_epoch(epoch)
            
            # メトリクスの保存
            self._save_epoch_metrics(train_metrics, val_metrics)
            
            # ログ出力
            self._log_epoch_results(epoch, train_metrics, val_metrics)
            
            # TensorBoardログ
            self._log_to_tensorboard(epoch, train_metrics, val_metrics)
            
            # チェックポイント保存の判定
            if self._should_save_checkpoint(val_metrics['loss']):
                self._save_checkpoint(epoch, val_metrics['loss'])
            
            # Early stopping の判定
            if self._should_stop_early():
                self.logger.info(f"Early stopping at epoch {epoch}")
                break
            
            # スケジューラーの更新
            self._update_scheduler()
        
        return self._get_current_metrics()
    
    def _train_epoch(self, epoch: int) -> Dict[str, float]:
        """1エポックの訓練を実行"""
        return self._compute_metrics(
            self.train_loader, 
            is_training=True, 
            epoch=epoch,
            desc="Training"
        )
    
    def _validate_epoch(self, epoch: int) -> Dict[str, float]:
        """1エポックの検証を実行"""
        return self._compute_metrics(
            self.val_loader, 
            is_training=False, 
            epoch=epoch,
            desc="Validation"
        )
    
    def _compute_metrics(self, 
                        dataloader: DataLoader, 
                        is_training: bool, 
                        epoch: int,
                        desc: str) -> Dict[str, float]:
        """
        データローダーに対してメトリクスを計算
        
        Args:
            dataloader: 処理するDataLoader
            is_training: 訓練モードかどうか
            epoch: 現在のエポック
            desc: プログレスバーの説明
            
        Returns:
            損失と精度を含む辞書
        """
        # モデルのモード設定
        if is_training:
            self.model.train()
        else:
            self.model.eval()
        
        running_loss = 0.0
        running_correct = 0
        running_samples = 0
        
        # 勾配計算の制御
        context_manager = torch.enable_grad() if is_training else torch.no_grad()
        
        with context_manager:
            for batch_idx, (X, y) in enumerate(tqdm(dataloader, desc=desc, leave=False)):
                X, y = X.to(self.device), y.to(self.device)
                batch_size = y.size(0)
                
                # TensorBoardグラフの追加（最初のバッチのみ）
                if is_training and epoch == 0 and batch_idx == 0:
                    self.writer.add_graph(self.model, X)
                
                # 勾配のリセット（訓練時のみ）
                if is_training:
                    self.optimizer.zero_grad()
                
                # 順伝播
                preds = self.model(X)
                loss = self.criterion(preds, y)
                
                # 逆伝播（訓練時のみ）
                if is_training:
                    loss.backward()
                    self.optimizer.step()
                
                # メトリクスの累積
                running_loss += loss.item() * batch_size
                running_correct += (preds.argmax(dim=1) == y).sum().item()
                running_samples += batch_size
        
        # 平均メトリクスの計算
        avg_loss = running_loss / running_samples
        avg_accuracy = running_correct / running_samples
        
        return {
            'loss': avg_loss,
            'accuracy': avg_accuracy
        }
    
    def _save_epoch_metrics(self, train_metrics: Dict[str, float], val_metrics: Dict[str, float]) -> None:
        """エポックのメトリクスを保存"""
        self.train_losses.append(train_metrics['loss'])
        self.train_accuracies.append(train_metrics['accuracy'])
        self.val_losses.append(val_metrics['loss'])
        self.val_accuracies.append(val_metrics['accuracy'])
    
    def _log_epoch_results(self, epoch: int, train_metrics: Dict[str, float], val_metrics: Dict[str, float]) -> None:
        """エポック結果をログ出力"""
        if self.verbose:
            log_message = (
                f"Epoch {epoch:3d}/{self.num_epochs-1} | "
                f"Train Loss: {train_metrics['loss']:.4f}, Train Acc: {train_metrics['accuracy']:.4f} | "
                f"Val Loss: {val_metrics['loss']:.4f}, Val Acc: {val_metrics['accuracy']:.4f}"
            )
            
            if self.scheduler:
                current_lr = self.scheduler.get_last_lr()[0]
                log_message += f" | LR: {current_lr:.6f}"
                
            self.logger.info(log_message)
    
    def _log_to_tensorboard(self, epoch: int, train_metrics: Dict[str, float], val_metrics: Dict[str, float]) -> None:
        """TensorBoardにメトリクスをログ"""
        self.writer.add_scalars('Loss', {
            'Train': train_metrics['loss'],
            'Validation': val_metrics['loss']
        }, epoch)
        
        self.writer.add_scalars('Accuracy', {
            'Train': train_metrics['accuracy'],
            'Validation': val_metrics['accuracy']
        }, epoch)
        
        if self.scheduler:
            self.writer.add_scalar('Learning Rate', self.scheduler.get_last_lr()[0], epoch)
    
    def _should_save_checkpoint(self, val_loss: float) -> bool:
        """チェックポイント保存の判定"""
        if val_loss < self.best_val_loss:
            self.best_val_loss = val_loss
            self.no_improve = 0
            return True
        else:
            self.no_improve += 1
            return False
    
    def _should_stop_early(self) -> bool:
        """Early stoppingの判定"""
        return (self.early_stopping is not None and 
                self.no_improve >= self.early_stopping)
    
    def _save_checkpoint(self, epoch: int, val_loss: float) -> None:
        """モデルチェックポイントを保存"""
        base_dir = self.save_path or TrainerConfig.DEFAULT_CHECKPOINT_DIR
        os.makedirs(base_dir, exist_ok=True)
        
        # 保存する状態
        state = {
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict() if self.scheduler else None,
            'epoch': epoch,
            'val_loss': val_loss,
            'best_val_loss': self.best_val_loss,
            'train_losses': self.train_losses,
            'val_losses': self.val_losses,
            'train_accuracies': self.train_accuracies,
            'val_accuracies': self.val_accuracies,
            'model_name': self.model_name,
            'timestamp': self.timestamp
        }
        
        # Best modelの保存
        best_path = os.path.join(base_dir, TrainerConfig.BEST_MODEL_FILENAME)
        torch.save(state, best_path)
        
        # 詳細情報付きファイルの保存
        detailed_filename = (f'{self.timestamp}_{self.model_name}_'
                           f'epoch{epoch:03d}_valloss{val_loss:.4f}.pth')
        detailed_path = os.path.join(base_dir, detailed_filename)
        torch.save(state, detailed_path)
        
        self.logger.info(f'Saved best model to: {detailed_path}')
    
    def _update_scheduler(self) -> None:
        """学習率スケジューラーの更新"""
        if self.scheduler is not None:
            if hasattr(self.scheduler, 'step'):
                # 一般的なスケジューラー
                if 'ReduceLROnPlateau' in str(type(self.scheduler)):
                    # ReduceLROnPlateauは検証損失を必要とする
                    self.scheduler.step(self.val_losses[-1])
                else:
                    self.scheduler.step()
    
    def _get_current_metrics(self) -> Tuple[List[float], List[float], List[float], List[float]]:
        """現在のメトリクスを取得"""
        return (self.train_losses, self.val_losses, 
                self.train_accuracies, self.val_accuracies)
    
    def predict(self, 
                dataloader: DataLoader, 
                return_probs: bool = False) -> Union[List[int], Tuple[List[int], List[List[float]]]]:
        """
        バッチ単位で推論を実行
        
        Args:
            dataloader: 推論用DataLoader
            return_probs: Trueの場合、各クラスの確率も返す
            
        Returns:
            予測ラベルのリスト、またはラベルと確率のタプル
            
        Examples:
            # 予測ラベルのみ取得
            >>> preds = trainer.predict(test_loader)
            
            # 予測ラベルと確率を取得
            >>> preds, probs = trainer.predict(test_loader, return_probs=True)
        """
        self.model.eval()
        all_preds = []
        all_probs = []

        with torch.no_grad():
            for batch in tqdm(dataloader, desc='Prediction', leave=False):
                inputs = self._extract_inputs(batch)
                inputs = inputs.to(self.device)
                
                outputs = self.model(inputs)
                pred_labels = torch.argmax(outputs, dim=1)
                
                all_preds.extend(pred_labels.cpu().tolist())
                
                if return_probs:
                    probs = torch.softmax(outputs, dim=1)
                    all_probs.extend(probs.cpu().tolist())

        return (all_preds, all_probs) if return_probs else all_preds
    
    def _extract_inputs(self, batch: Any) -> torch.Tensor:
        """
        バッチから入力データを抽出
        
        Args:
            batch: DataLoaderからの1バッチ
            
        Returns:
            入力テンソル
        """
        if isinstance(batch, (list, tuple)):
            return batch[0]  # (X, y)の場合はXを返す
        return batch  # Xのみの場合
    
    def load_checkpoint(self, checkpoint_path: str, load_optimizer: bool = True, load_scheduler: bool = True) -> Dict[str, Any]:
        """
        チェックポイントからモデルを読み込み
        
        Args:
            checkpoint_path: チェックポイントファイルのパス
            load_optimizer: optimizerの状態も読み込むかどうか
            load_scheduler: schedulerの状態も読み込むかどうか
            
        Returns:
            チェックポイントの情報辞書
        """
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        # モデルの状態を読み込み
        self.model.load_state_dict(checkpoint['model_state_dict'])
        
        # オプティマイザーの状態を読み込み
        if load_optimizer and 'optimizer_state_dict' in checkpoint:
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        # スケジューラーの状態を読み込み
        if (load_scheduler and self.scheduler is not None and 
            'scheduler_state_dict' in checkpoint and 
            checkpoint['scheduler_state_dict'] is not None):
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        
        # 訓練状態の復元
        if 'train_losses' in checkpoint:
            self.train_losses = checkpoint['train_losses']
            self.val_losses = checkpoint['val_losses']
            self.train_accuracies = checkpoint['train_accuracies']
            self.val_accuracies = checkpoint['val_accuracies']
            self.best_val_loss = checkpoint.get('best_val_loss', float('inf'))
        
        self.logger.info(f"Checkpoint loaded from: {checkpoint_path}")
        self.logger.info(f"Epoch: {checkpoint.get('epoch', 'unknown')}, "
                        f"Val Loss: {checkpoint.get('val_loss', 'unknown')}")
        
        return checkpoint
    
    def get_training_summary(self) -> Dict[str, Any]:
        """
        訓練の要約情報を取得
        
        Returns:
            訓練の要約情報を含む辞書
        """
        if not self.train_losses:
            return {"message": "No training data available"}
        
        return {
            'model_name': self.model_name,
            'total_epochs': len(self.train_losses),
            'best_val_loss': self.best_val_loss,
            'best_val_accuracy': max(self.val_accuracies) if self.val_accuracies else 0,
            'final_train_loss': self.train_losses[-1],
            'final_train_accuracy': self.train_accuracies[-1],
            'final_val_loss': self.val_losses[-1],
            'final_val_accuracy': self.val_accuracies[-1],
            'device': str(self.device),
            'timestamp': self.timestamp
        }


# 使用例とヘルパー関数
def create_trainer(train_loader: DataLoader,
                  val_loader: DataLoader,
                  model: nn.Module,
                  criterion: nn.Module,
                  optimizer: Optimizer,
                  num_epochs: int,
                  **kwargs) -> Trainer:
    """
    Trainerインスタンスを作成するヘルパー関数
    
    Args:
        train_loader: 訓練用DataLoader
        val_loader: 検証用DataLoader
        model: 訓練するモデル
        criterion: 損失関数
        optimizer: 最適化アルゴリズム
        num_epochs: エポック数
        **kwargs: Trainerの追加引数
        
    Returns:
        設定済みのTrainerインスタンス
    """
    return Trainer(
        train_loader=train_loader,
        val_loader=val_loader,
        model=model,
        criterion=criterion,
        optimizer=optimizer,
        num_epochs=num_epochs,
        **kwargs
    )