"""
trainer.py - Training Engine with MLflow Experiment Tracking

Covers:
- Full training loop (train/val/test)
- MLflow experiment tracking & model registry
- Early stopping, gradient clipping
- Mixed precision training (AMP)
- Hyperparameter optimization with Optuna
- Model checkpointing
- Comprehensive metrics (accuracy, F1, precision, recall, ROC-AUC)
"""

import os
import time
import json
import logging
import numpy as np
import torch
import torch.nn as nn
from torch.optim import AdamW, Adam
from torch.cuda.amp import GradScaler, autocast
from torch.utils.data import DataLoader
from typing import Dict, Optional, Callable, Any
from pathlib import Path
from dataclasses import dataclass, field

import mlflow
import mlflow.pytorch
from sklearn.metrics import (
    accuracy_score, f1_score, precision_score,
    recall_score, roc_auc_score, classification_report
)
import optuna
from tqdm import tqdm

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class TrainingConfig:
    """Hyperparameters and training settings."""
    # Model
    model_type: str = "bert"  # bert, cnn, lstm, ensemble
    model_name: str = "distilbert-base-uncased"
    num_classes: int = 3

    # Training
    epochs: int = 10
    learning_rate: float = 2e-5
    weight_decay: float = 0.01
    warmup_ratio: float = 0.1
    max_grad_norm: float = 1.0
    label_smoothing: float = 0.1

    # Regularization
    dropout_rate: float = 0.3
    use_mixed_precision: bool = True

    # Early stopping
    patience: int = 3
    min_delta: float = 1e-4

    # Checkpointing
    checkpoint_dir: str = "checkpoints"
    save_best_only: bool = True

    # MLflow
    mlflow_uri: str = "http://localhost:5000"
    experiment_name: str = "sentiment-classification"

    # Optimization
    scheduler: str = "cosine"  # linear, cosine, plateau
    batch_size: int = 32
    accumulation_steps: int = 1  # gradient accumulation

    # Metrics
    monitor_metric: str = "val_f1"


class EarlyStopping:
    """Early stopping with configurable patience and delta."""

    def __init__(self, patience: int = 5, min_delta: float = 1e-4, mode: str = "max"):
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.counter = 0
        self.best_score = None
        self.should_stop = False

    def __call__(self, score: float) -> bool:
        if self.best_score is None:
            self.best_score = score
        else:
            if self.mode == "max":
                improved = score > self.best_score + self.min_delta
            else:
                improved = score < self.best_score - self.min_delta

            if improved:
                self.best_score = score
                self.counter = 0
            else:
                self.counter += 1
                if self.counter >= self.patience:
                    self.should_stop = True
                    logger.info(f"Early stopping triggered after {self.counter} epochs without improvement")

        return self.should_stop


class MetricsCalculator:
    """Calculate classification metrics."""

    @staticmethod
    def compute(
        y_true: np.ndarray,
        y_pred: np.ndarray,
        y_proba: np.ndarray,
        num_classes: int
    ) -> Dict[str, float]:
        metrics = {
            'accuracy': accuracy_score(y_true, y_pred),
            'f1_macro': f1_score(y_true, y_pred, average='macro', zero_division=0),
            'f1_weighted': f1_score(y_true, y_pred, average='weighted', zero_division=0),
            'precision_macro': precision_score(y_true, y_pred, average='macro', zero_division=0),
            'recall_macro': recall_score(y_true, y_pred, average='macro', zero_division=0),
        }

        # ROC-AUC (multiclass)
        if num_classes == 2:
            metrics['roc_auc'] = roc_auc_score(y_true, y_proba[:, 1])
        else:
            try:
                metrics['roc_auc'] = roc_auc_score(
                    y_true, y_proba, multi_class='ovr', average='macro'
                )
            except Exception:
                metrics['roc_auc'] = 0.0

        return metrics


class Trainer:
    """
    Full-featured trainer with MLflow tracking, early stopping, and mixed precision.
    """

    def __init__(
        self,
        model: nn.Module,
        config: TrainingConfig,
        device: Optional[torch.device] = None
    ):
        self.model = model
        self.config = config
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        logger.info(f"Using device: {self.device}")
        self.model.to(self.device)

        # Loss function with label smoothing
        self.criterion = nn.CrossEntropyLoss(label_smoothing=config.label_smoothing)

        # Mixed precision scaler
        self.scaler = GradScaler() if config.use_mixed_precision and self.device.type == 'cuda' else None

        # Setup MLflow
        self._setup_mlflow()

        # Metrics tracker
        self.history = {'train': [], 'val': []}
        self.best_model_state = None

        # Create checkpoint dir
        Path(config.checkpoint_dir).mkdir(parents=True, exist_ok=True)

    def _setup_mlflow(self):
        """Initialize MLflow tracking."""
        try:
            mlflow.set_tracking_uri(self.config.mlflow_uri)
        except Exception:
            logger.warning("Could not connect to MLflow server, using local tracking")
            mlflow.set_tracking_uri("./mlruns")

        mlflow.set_experiment(self.config.experiment_name)

    def _get_optimizer(self) -> torch.optim.Optimizer:
        """Get optimizer with layer-wise learning rates if supported."""
        if hasattr(self.model, 'get_optimizer_groups'):
            # BERT-style with custom optimizer groups
            param_groups = self.model.get_optimizer_groups(
                learning_rate=self.config.learning_rate,
                weight_decay=self.config.weight_decay
            )
            return AdamW(param_groups)
        else:
            return AdamW(
                self.model.parameters(),
                lr=self.config.learning_rate,
                weight_decay=self.config.weight_decay
            )

    def _get_scheduler(self, optimizer, num_training_steps: int):
        """Get learning rate scheduler."""
        num_warmup_steps = int(num_training_steps * self.config.warmup_ratio)

        if self.config.scheduler == "linear":
            from transformers import get_linear_schedule_with_warmup
            return get_linear_schedule_with_warmup(
                optimizer, num_warmup_steps, num_training_steps
            )
        elif self.config.scheduler == "cosine":
            from transformers import get_cosine_schedule_with_warmup
            return get_cosine_schedule_with_warmup(
                optimizer, num_warmup_steps, num_training_steps
            )
        elif self.config.scheduler == "plateau":
            return torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer, mode='max', patience=2, factor=0.5
            )
        else:
            return None

    def train_epoch(
        self,
        train_loader: DataLoader,
        optimizer: torch.optim.Optimizer,
        scheduler=None
    ) -> Dict[str, float]:
        """Run one training epoch."""
        self.model.train()
        total_loss = 0.0
        all_preds, all_labels, all_probs = [], [], []

        optimizer.zero_grad()

        pbar = tqdm(train_loader, desc="Training", leave=False)
        for step, batch in enumerate(pbar):
            # Move batch to device
            input_ids = batch['input_ids'].to(self.device)
            labels = batch['label'].to(self.device)
            attention_mask = batch.get('attention_mask', None)
            if attention_mask is not None:
                attention_mask = attention_mask.to(self.device)

            # Forward pass with optional mixed precision
            if self.scaler:
                with autocast():
                    outputs = self.model(input_ids, attention_mask=attention_mask)
                    loss = self.criterion(outputs['logits'], labels)
                    loss = loss / self.config.accumulation_steps
                self.scaler.scale(loss).backward()
            else:
                outputs = self.model(input_ids, attention_mask=attention_mask)
                loss = self.criterion(outputs['logits'], labels)
                loss = loss / self.config.accumulation_steps
                loss.backward()

            # Gradient accumulation
            if (step + 1) % self.config.accumulation_steps == 0:
                if self.scaler:
                    self.scaler.unscale_(optimizer)
                # Gradient clipping
                nn.utils.clip_grad_norm_(self.model.parameters(), self.config.max_grad_norm)

                if self.scaler:
                    self.scaler.step(optimizer)
                    self.scaler.update()
                else:
                    optimizer.step()

                if scheduler and not isinstance(scheduler,
                    torch.optim.lr_scheduler.ReduceLROnPlateau):
                    scheduler.step()

                optimizer.zero_grad()

            total_loss += loss.item() * self.config.accumulation_steps
            preds = outputs['probabilities'].argmax(dim=-1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_probs.extend(outputs['probabilities'].detach().cpu().numpy())

            pbar.set_postfix({'loss': f'{total_loss/(step+1):.4f}'})

        metrics = MetricsCalculator.compute(
            np.array(all_labels),
            np.array(all_preds),
            np.array(all_probs),
            self.config.num_classes
        )
        metrics['loss'] = total_loss / len(train_loader)
        return metrics

    @torch.no_grad()
    def evaluate(self, loader: DataLoader, phase: str = "val") -> Dict[str, float]:
        """Evaluate model on a data loader."""
        self.model.eval()
        total_loss = 0.0
        all_preds, all_labels, all_probs = [], [], []

        for batch in tqdm(loader, desc=f"Evaluating ({phase})", leave=False):
            input_ids = batch['input_ids'].to(self.device)
            labels = batch['label'].to(self.device)
            attention_mask = batch.get('attention_mask', None)
            if attention_mask is not None:
                attention_mask = attention_mask.to(self.device)

            outputs = self.model(input_ids, attention_mask=attention_mask)
            loss = self.criterion(outputs['logits'], labels)

            total_loss += loss.item()
            preds = outputs['probabilities'].argmax(dim=-1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_probs.extend(outputs['probabilities'].cpu().numpy())

        metrics = MetricsCalculator.compute(
            np.array(all_labels),
            np.array(all_preds),
            np.array(all_probs),
            self.config.num_classes
        )
        metrics['loss'] = total_loss / len(loader)
        return metrics

    def train(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
        test_loader: Optional[DataLoader] = None
    ) -> Dict[str, Any]:
        """
        Full training loop with MLflow tracking.
        """
        optimizer = self._get_optimizer()
        num_training_steps = len(train_loader) * self.config.epochs
        scheduler = self._get_scheduler(optimizer, num_training_steps)
        early_stopping = EarlyStopping(
            patience=self.config.patience,
            min_delta=self.config.min_delta,
            mode="max"
        )

        best_val_score = 0.0
        best_epoch = 0

        with mlflow.start_run() as run:
            # Log hyperparameters
            mlflow.log_params({
                'model_type': self.config.model_type,
                'learning_rate': self.config.learning_rate,
                'epochs': self.config.epochs,
                'batch_size': self.config.batch_size,
                'weight_decay': self.config.weight_decay,
                'scheduler': self.config.scheduler,
                'dropout_rate': self.config.dropout_rate,
                'label_smoothing': self.config.label_smoothing,
                'num_classes': self.config.num_classes
            })

            # Log model architecture info
            if hasattr(self.model, 'get_model_info'):
                mlflow.log_params(self.model.get_model_info())

            logger.info(f"Starting training for {self.config.epochs} epochs...")
            logger.info(f"MLflow Run ID: {run.info.run_id}")

            for epoch in range(self.config.epochs):
                epoch_start = time.time()

                # Train
                train_metrics = self.train_epoch(train_loader, optimizer, scheduler)

                # Validate
                val_metrics = self.evaluate(val_loader, "val")

                # Update scheduler if ReduceLROnPlateau
                if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                    scheduler.step(val_metrics['f1_macro'])

                epoch_time = time.time() - epoch_start

                # Log metrics to MLflow
                for key, value in train_metrics.items():
                    mlflow.log_metric(f"train_{key}", value, step=epoch)
                for key, value in val_metrics.items():
                    mlflow.log_metric(f"val_{key}", value, step=epoch)
                mlflow.log_metric("epoch_time", epoch_time, step=epoch)
                mlflow.log_metric("learning_rate",
                    optimizer.param_groups[0]['lr'], step=epoch)

                # Store history
                self.history['train'].append(train_metrics)
                self.history['val'].append(val_metrics)

                val_score = val_metrics[self.config.monitor_metric.replace('val_', '')]
                
                logger.info(
                    f"Epoch {epoch+1}/{self.config.epochs} | "
                    f"Train Loss: {train_metrics['loss']:.4f} | "
                    f"Val Loss: {val_metrics['loss']:.4f} | "
                    f"Val F1: {val_metrics['f1_macro']:.4f} | "
                    f"Val Acc: {val_metrics['accuracy']:.4f} | "
                    f"Time: {epoch_time:.1f}s"
                )

                # Save best model
                if val_score > best_val_score:
                    best_val_score = val_score
                    best_epoch = epoch + 1
                    self.best_model_state = {
                        k: v.cpu().clone() for k, v in self.model.state_dict().items()
                    }
                    # Save checkpoint
                    ckpt_path = os.path.join(
                        self.config.checkpoint_dir,
                        f"best_{self.config.model_type}_model.pt"
                    )
                    torch.save({
                        'epoch': epoch,
                        'model_state_dict': self.best_model_state,
                        'optimizer_state_dict': optimizer.state_dict(),
                        'val_score': best_val_score,
                        'config': self.config
                    }, ckpt_path)
                    logger.info(f"  ✅ New best model saved (epoch {best_epoch}, score: {best_val_score:.4f})")

                # Early stopping check
                if early_stopping(val_score):
                    logger.info(f"Early stopping at epoch {epoch+1}")
                    break

            # Load best model for final evaluation
            if self.best_model_state:
                self.model.load_state_dict(self.best_model_state)
                self.model.to(self.device)

            # Final test evaluation
            final_metrics = {}
            if test_loader:
                test_metrics = self.evaluate(test_loader, "test")
                for key, value in test_metrics.items():
                    mlflow.log_metric(f"test_{key}", value)
                final_metrics['test'] = test_metrics
                logger.info(f"\n{'='*50}")
                logger.info(f"FINAL TEST RESULTS:")
                logger.info(f"  Accuracy:  {test_metrics['accuracy']:.4f}")
                logger.info(f"  F1 Macro:  {test_metrics['f1_macro']:.4f}")
                logger.info(f"  ROC-AUC:   {test_metrics['roc_auc']:.4f}")
                logger.info(f"{'='*50}")

            # Log best metrics
            mlflow.log_metrics({
                'best_val_score': best_val_score,
                'best_epoch': best_epoch,
            })

            # Register model in MLflow Model Registry
            model_uri = f"runs:/{run.info.run_id}/model"
            try:
                mlflow.pytorch.log_model(
                    self.model,
                    "model",
                    registered_model_name=f"{self.config.model_type}-sentiment"
                )
                logger.info(f"✅ Model registered in MLflow Registry")
            except Exception as e:
                logger.warning(f"Could not register model: {e}")

            final_metrics['best_val_score'] = best_val_score
            final_metrics['best_epoch'] = best_epoch
            final_metrics['run_id'] = run.info.run_id
            final_metrics['history'] = self.history

        return final_metrics


# ─── Hyperparameter Optimization with Optuna ────────────────────────────────

def create_optuna_objective(
    train_loader: DataLoader,
    val_loader: DataLoader,
    model_class,
    model_kwargs: dict,
    n_epochs: int = 3
) -> Callable:
    """Create Optuna objective function for hyperparameter search."""

    def objective(trial: optuna.Trial) -> float:
        # Suggest hyperparameters
        lr = trial.suggest_float("learning_rate", 1e-5, 1e-3, log=True)
        dropout = trial.suggest_float("dropout_rate", 0.1, 0.5)
        weight_decay = trial.suggest_float("weight_decay", 1e-4, 0.1, log=True)
        scheduler = trial.suggest_categorical("scheduler", ["linear", "cosine"])
        label_smoothing = trial.suggest_float("label_smoothing", 0.0, 0.2)

        # Update model kwargs
        model_kwargs_trial = {**model_kwargs, 'dropout_rate': dropout}

        # Create model
        model = model_class(**model_kwargs_trial)

        config = TrainingConfig(
            epochs=n_epochs,
            learning_rate=lr,
            weight_decay=weight_decay,
            dropout_rate=dropout,
            scheduler=scheduler,
            label_smoothing=label_smoothing,
            patience=2,
            experiment_name="optuna-hpo"
        )

        trainer = Trainer(model, config)
        results = trainer.train(train_loader, val_loader)

        return results['best_val_score']

    return objective


def run_hyperparameter_search(
    train_loader: DataLoader,
    val_loader: DataLoader,
    model_class,
    model_kwargs: dict,
    n_trials: int = 20,
    n_epochs: int = 3
) -> optuna.Study:
    """Run Optuna hyperparameter search."""
    logger.info(f"Starting Optuna HPO: {n_trials} trials, {n_epochs} epochs each")

    study = optuna.create_study(
        direction="maximize",
        study_name="sentiment-hpo",
        pruner=optuna.pruners.MedianPruner(n_startup_trials=5)
    )

    objective = create_optuna_objective(
        train_loader, val_loader, model_class, model_kwargs, n_epochs
    )

    study.optimize(objective, n_trials=n_trials, show_progress_bar=True)

    logger.info(f"Best trial: {study.best_trial.number}")
    logger.info(f"Best params: {study.best_params}")
    logger.info(f"Best val score: {study.best_value:.4f}")

    return study


if __name__ == "__main__":
    import sys
    sys.path.append('../../')
    from src.data.data_loader import DataPipeline, DataConfig
    from src.models.bert_model import TransformerClassifier

    print("Testing Trainer with BERT...")

    # Quick data
    config = DataConfig(batch_size=8)
    pipeline = DataPipeline(config)
    loaders, n_classes = pipeline.prepare_loaders(mode="bert")

    # Model
    model = TransformerClassifier(num_classes=n_classes)

    # Train
    train_config = TrainingConfig(
        model_type="bert",
        num_classes=n_classes,
        epochs=2,
        learning_rate=2e-5,
        mlflow_uri="./mlruns"
    )

    trainer = Trainer(model, train_config)
    results = trainer.train(loaders['train'], loaders['val'], loaders['test'])
    print(f"✅ Training complete! Best val score: {results['best_val_score']:.4f}")
