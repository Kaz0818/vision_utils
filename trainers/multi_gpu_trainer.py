import os
from datetime import datetime
from typing import Optional, Tuple, List, Dict, Union
from tqdm import tqdm
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from timm.data.mixup import Mixup
from timm.loss import SoftTargetCrossEntropy


class TrainerConfig:
    """Trainer設定用の定数クラス"""
    DEFAULT_LOG_DIR = "logs"
    DEFAULT_CHECKPOINT_DIR = "checkpoints"
    BEST_MODEL_FILENAME = "best_model.pth"
    TIMESTAMP_FORMAT = '%Y-%m-%d_%H-%M-%S'


class MixupTrainer:
    def __init__(self, 
                 train_loader: DataLoader,
                 val_loader: DataLoader,
                 model: nn.Module,
                 train_criterion: Optional[nn.Module] = None,  # 訓練用損失関数
                 val_criterion: Optional[nn.Module] = None,    # 検証用損失関数
                 optimizer = None,
                 num_epochs: int = 100,
                 device: Optional[torch.device] = None,
                 early_stopping: Optional[int] = None,
                 scheduler = None,
                 save_path: Optional[str] = None,
                 log_dir: Optional[str] = None,
                 writer: Optional[SummaryWriter] = None,
                 verbose: bool = True,
                 # mixup関連パラメータ
                 mixup_alpha: float = 0.0,
                 cutmix_alpha: float = 0.0,
                 cutmix_minmax: Optional[Tuple[float, float]] = None,
                 prob: float = 1.0,
                 switch_prob: float = 0.5,
                 mode: str = 'batch',
                 label_smoothing: float = 0.1,
                 num_classes: int = 1000):
        """
        mixup対応Trainerクラス（訓練・検証で異なる損失関数対応）
        
        Args:
            train_loader: 訓練用DataLoader
            val_loader: 検証用DataLoader
            model: 学習対象モデル
            train_criterion: 訓練用損失関数（Noneの場合SoftTargetCrossEntropyを使用）
            val_criterion: 検証用損失関数（Noneの場合CrossEntropyLossを使用）
            optimizer: オプティマイザー
            num_epochs: エポック数
            device: デバイス
            early_stopping: Early stopping回数
            scheduler: 学習率スケジューラー
            save_path: モデル保存パス
            log_dir: ログディレクトリ
            writer: TensorBoardライター
            verbose: 詳細出力フラグ
            mixup_alpha: mixupのアルファ値（0.0で無効）
            cutmix_alpha: cutmixのアルファ値（0.0で無効）
            cutmix_minmax: cutmixの最小・最大比率
            prob: mixup/cutmixを適用する確率
            switch_prob: mixupとcutmixを切り替える確率
            mode: 'batch' or 'pair' or 'elem'
            label_smoothing: ラベルスムージング値
            num_classes: クラス数
        """
        # 基本設定
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.model = model
        self.optimizer = optimizer
        self.num_epochs = num_epochs
        self.device = device if device else torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.early_stopping = early_stopping
        self.scheduler = scheduler
        self.save_path = save_path
        self.verbose = verbose
        
        self.model_name = self.model.__class__.__name__
        self.timestamp = datetime.now().strftime(TrainerConfig.TIMESTAMP_FORMAT)
        self.model.to(self.device)
        
        # ログ設定
        self.log_dir = log_dir or f"{TrainerConfig.DEFAULT_LOG_DIR}/{self.timestamp}_{self.model_name}"
        self.writer = writer or SummaryWriter(self.log_dir)
        
        # 損失関数の設定
        self._setup_loss_functions(train_criterion, val_criterion, label_smoothing)
        
        # mixup設定
        self.use_mixup = mixup_alpha > 0 or cutmix_alpha > 0
        if self.use_mixup:
            self.mixup_fn = Mixup(
                mixup_alpha=mixup_alpha,
                cutmix_alpha=cutmix_alpha,
                cutmix_minmax=cutmix_minmax,
                prob=prob,
                switch_prob=switch_prob,
                mode=mode,
                label_smoothing=label_smoothing,
                num_classes=num_classes
            )
            if self.verbose:
                print(f"[INFO] Mixup enabled - mixup_alpha: {mixup_alpha}, cutmix_alpha: {cutmix_alpha}")
        else:
            self.mixup_fn = None
            if self.verbose:
                print("[INFO] Mixup disabled")
        
        self._initialize_training_state()
        
        if self.verbose:
            self._print_configuration()
    
    def _setup_loss_functions(self, train_criterion: Optional[nn.Module], 
                            val_criterion: Optional[nn.Module], 
                            label_smoothing: float):
        """損失関数の設定"""
        # 訓練用損失関数（mixup対応）
        if train_criterion is not None:
            self.train_criterion = train_criterion
        else:
            self.train_criterion = SoftTargetCrossEntropy()
            if self.verbose:
                print("[INFO] Using SoftTargetCrossEntropy for training")
        
        # 検証用損失関数（通常のCrossEntropy）
        if val_criterion is not None:
            self.val_criterion = val_criterion
        else:
            # label_smoothingを適用したCrossEntropyLoss
            if label_smoothing > 0:
                self.val_criterion = nn.CrossEntropyLoss(label_smoothing=label_smoothing)
                if self.verbose:
                    print(f"[INFO] Using CrossEntropyLoss with label_smoothing={label_smoothing} for validation")
            else:
                self.val_criterion = nn.CrossEntropyLoss()
                if self.verbose:
                    print("[INFO] Using standard CrossEntropyLoss for validation")
    
    def _print_configuration(self):
        """設定情報の出力"""
        print(f"[INFO] Configuration:")
        print(f"  - Model: {self.model_name}")
        print(f"  - Device: {self.device}")
        print(f"  - Epochs: {self.num_epochs}")
        print(f"  - Train criterion: {type(self.train_criterion).__name__}")
        print(f"  - Val criterion: {type(self.val_criterion).__name__}")
        print(f"  - Mixup: {'Enabled' if self.use_mixup else 'Disabled'}")
        if self.early_stopping:
            print(f"  - Early stopping: {self.early_stopping} epochs")
    
    def _initialize_training_state(self):
        """訓練状態の初期化"""
        self.train_losses: List[float] = []
        self.train_accuracies: List[float] = []
        self.val_losses: List[float] = []
        self.val_accuracies: List[float] = []
        self.best_val_loss = float('inf')
        self.no_improve = 0
        self.current_epoch = 0

    def train(self) -> Tuple[List[float], List[float], List[float], List[float]]:
        """モデルの訓練を実行"""
        try:
            if self.verbose:
                print(f"\n{'='*50}")
                print("Starting training...")
                print(f"{'='*50}")
            
            return self._training_loop()
            
        except KeyboardInterrupt:
            print("\n[INFO] Training interrupted by user")
            return self._get_current_metrics()
            
        except Exception as e:
            print(f"[ERROR] Training failed: {e}")
            raise
            
        finally:
            self.writer.close()
            if self.verbose:
                print(f"\n{'='*50}")
                print("Training completed")
                print(f"{'='*50}")

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
                if self.verbose:
                    print(f"[INFO] Early stopping at epoch {epoch}")
                break
            
            # スケジューラーの更新
            self._update_scheduler()
        
        return self._get_current_metrics()

    def _train_epoch(self, epoch: int) -> Dict[str, float]:
        """1エポックの訓練を実行（mixup対応）"""
        return self._compute_metrics(
            self.train_loader, 
            is_training=True, 
            epoch=epoch,
            desc="Training"
        )

    def _validate_epoch(self, epoch: int) -> Dict[str, float]:
        """1エポックの検証を実行（mixupなし）"""
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
        """メトリクス計算（mixup対応、訓練・検証で異なる損失関数）"""
        if is_training:
            self.model.train()
            criterion = self.train_criterion
        else:
            self.model.eval()
            criterion = self.val_criterion
        
        running_loss = 0.0
        running_correct = 0
        running_samples = 0
        
        context_manager = torch.enable_grad() if is_training else torch.no_grad()
        
        with context_manager:
            for batch_idx, (X, y) in enumerate(tqdm(dataloader, desc=desc, leave=False)):
                X, y = X.to(self.device), y.to(self.device)
                batch_size = y.size(0)
                original_y = y.clone()  # accuracy計算用に元のラベルを保存
                
                # TensorBoardグラフの追加（最初のバッチのみ）
                if is_training and epoch == 0 and batch_idx == 0:
                    self.writer.add_graph(self.model.module, X)
                
                # mixupの適用（訓練時のみ）
                if is_training and self.use_mixup:
                    X, y = self.mixup_fn(X, y)
                
                # 勾配のリセット（訓練時のみ）
                if is_training:
                    self.optimizer.zero_grad()
                
                # 順伝播
                preds = self.model(X)
                loss = criterion(preds, y)
                
                # 逆伝播（訓練時のみ）
                if is_training:
                    loss.backward()
                    self.optimizer.step()
                
                # メトリクスの累積
                running_loss += loss.item() * batch_size
                running_samples += batch_size
                
                # accuracy計算
                if is_training and self.use_mixup:
                    # mixup使用時は元のラベルで近似的に計算
                    running_correct += (preds.argmax(dim=1) == original_y).sum().item()
                else:
                    # 通常の場合
                    running_correct += (preds.argmax(dim=1) == y).sum().item()
        
        # 平均メトリクスの計算
        avg_loss = running_loss / running_samples
        avg_accuracy = running_correct / running_samples
        
        return {
            'loss': avg_loss,
            'accuracy': avg_accuracy
        }

    def _save_epoch_metrics(self, train_metrics: Dict[str, float], val_metrics: Dict[str, float]):
        """エポックのメトリクスを保存"""
        self.train_losses.append(train_metrics['loss'])
        self.train_accuracies.append(train_metrics['accuracy'])
        self.val_losses.append(val_metrics['loss'])
        self.val_accuracies.append(val_metrics['accuracy'])

    def _log_epoch_results(self, epoch: int, train_metrics: Dict[str, float], val_metrics: Dict[str, float]):
        """エポック結果をログ出力"""
        if self.verbose:
            log_message = (
                f"Epoch {epoch:3d}/{self.num_epochs-1} | "
                f"Train Loss: {train_metrics['loss']:.4f}, Train Acc: {train_metrics['accuracy']:.4f} | "
                f"Val Loss: {val_metrics['loss']:.4f}, Val Acc: {val_metrics['accuracy']:.4f}"
            )
            
            if self.scheduler:
                try:
                    current_lr = self.scheduler.get_last_lr()[0]
                    log_message += f" | LR: {current_lr:.6f}"
                except:
                    # 一部のスケジューラーでget_last_lr()が使用できない場合
                    pass
                
            print(log_message)

    def _log_to_tensorboard(self, epoch: int, train_metrics: Dict[str, float], val_metrics: Dict[str, float]):
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
            try:
                self.writer.add_scalar('Learning Rate', self.scheduler.get_last_lr()[0], epoch)
            except:
                pass

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

    def _save_checkpoint(self, epoch: int, val_loss: float):
        """モデルチェックポイントを保存"""
        base_dir = self.save_path or TrainerConfig.DEFAULT_CHECKPOINT_DIR
        os.makedirs(base_dir, exist_ok=True)
        model_to_save = self.model.module if isinstance(self.model, nn.DataParallel) else self.model
        
        state = {
            'model_state_dict': model_to_save.state_dict(),
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
            'timestamp': self.timestamp,
            'loss_config': {
                'train_criterion': type(self.train_criterion).__name__,
                'val_criterion': type(self.val_criterion).__name__,
            },
            'mixup_config': {
                'use_mixup': self.use_mixup,
                'mixup_fn': str(self.mixup_fn) if self.mixup_fn else None
            }
        }
        
        best_path = os.path.join(base_dir, TrainerConfig.BEST_MODEL_FILENAME)
        torch.save(state, best_path)
        
        detailed_filename = (f'{self.timestamp}_{self.model_name}_'
                           f'epoch{epoch:03d}_valloss{val_loss:.4f}.pth')
        detailed_path = os.path.join(base_dir, detailed_filename)
        torch.save(state, detailed_path)
        
        if self.verbose:
            print(f'[INFO] Saved best model to: {detailed_path}')

    def _update_scheduler(self):
        """学習率スケジューラーの更新"""
        if self.scheduler is not None:
            if hasattr(self.scheduler, 'step'):
                scheduler_name = type(self.scheduler).__name__
                if 'ReduceLROnPlateau' in scheduler_name:
                    self.scheduler.step(self.val_losses[-1])
                else:
                    self.scheduler.step()

    def _get_current_metrics(self) -> Tuple[List[float], List[float], List[float], List[float]]:
        """現在のメトリクスを取得"""
        return (self.train_losses, self.val_losses, 
                self.train_accuracies, self.val_accuracies)

    def predict(self, dataloader: DataLoader, return_probs: bool = False):
        """推論実行（mixupなし）"""
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

    def _extract_inputs(self, batch):
        """バッチから入力データを抽出"""
        if isinstance(batch, (list, tuple)):
            return batch[0]
        return batch

    def get_best_metrics(self) -> Dict[str, float]:
        """最良のメトリクスを取得"""
        if not self.val_losses:
            return {}
        
        best_epoch = self.val_losses.index(min(self.val_losses))
        return {
            'best_epoch': best_epoch,
            'best_val_loss': self.val_losses[best_epoch],
            'best_val_accuracy': self.val_accuracies[best_epoch],
            'train_loss_at_best': self.train_losses[best_epoch],
            'train_accuracy_at_best': self.train_accuracies[best_epoch]
        }

    def load_checkpoint(self, checkpoint_path: str, load_optimizer: bool = True, load_scheduler: bool = True):
        """チェックポイントからモデルを読み込み"""
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        # モデルの状態を読み込み
        self.model.load_state_dict(checkpoint['model_state_dict'])
        
        # オプティマイザーの状態を読み込み
        if load_optimizer and self.optimizer and 'optimizer_state_dict' in checkpoint:
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        # スケジューラーの状態を読み込み
        if load_scheduler and self.scheduler and 'scheduler_state_dict' in checkpoint:
            if checkpoint['scheduler_state_dict'] is not None:
                self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        
        # メトリクス履歴を復元
        if 'train_losses' in checkpoint:
            self.train_losses = checkpoint['train_losses']
            self.val_losses = checkpoint['val_losses']
            self.train_accuracies = checkpoint['train_accuracies']
            self.val_accuracies = checkpoint['val_accuracies']
            self.best_val_loss = checkpoint.get('best_val_loss', float('inf'))
        
        if self.verbose:
            print(f"[INFO] Loaded checkpoint from: {checkpoint_path}")
            print(f"[INFO] Checkpoint epoch: {checkpoint.get('epoch', 'Unknown')}")
            print(f"[INFO] Checkpoint val_loss: {checkpoint.get('val_loss', 'Unknown'):.4f}")


# 使用例とヘルパー関数
def create_mixup_trainer(train_loader, val_loader, model, optimizer, 
                        mixup_alpha=0.2, cutmix_alpha=1.0, num_classes=1000,
                        label_smoothing=0.1, **kwargs):
    """
    Mixup対応Trainerの作成ヘルパー関数
    
    Args:
        train_loader: 訓練用DataLoader
        val_loader: 検証用DataLoader  
        model: モデル
        optimizer: オプティマイザー
        mixup_alpha: mixupのアルファ値
        cutmix_alpha: cutmixのアルファ値
        num_classes: クラス数
        label_smoothing: ラベルスムージング値
        **kwargs: その他のTrainerパラメータ
    
    Returns:
        MixupTrainer: 設定済みのTrainerインスタンス
    """
    return MixupTrainer(
        train_loader=train_loader,
        val_loader=val_loader,
        model=model,
        optimizer=optimizer,
        mixup_alpha=mixup_alpha,
        cutmix_alpha=cutmix_alpha,
        num_classes=num_classes,
        label_smoothing=label_smoothing,
        **kwargs
    )