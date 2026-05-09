"""
Training Loop: train_with_cv(), train_single(), com StratifiedKFold e callbacks
Responsible for executing the training of a specific model.
"""

from typing import Tuple, Optional, List, Dict, Any
from dataclasses import dataclass
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import StratifiedKFold
import warnings

warnings.filterwarnings("ignore")


@dataclass
class TrainingConfig:
    """Configuration for training"""
    epochs: int = 50
    batch_size: int = 32
    learning_rate: float = 0.001
    weight_decay: float = 0.0
    early_stopping_patience: int = 10
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    verbose: bool = True


class EarlyStopping:
    """Early stopping based on validation loss"""

    def __init__(self, patience: int = 10, verbose: bool = False):
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_loss = None
        self.early_stop = False

    def __call__(self, val_loss: float) -> bool:
        if self.best_loss is None:
            self.best_loss = val_loss
        elif val_loss < self.best_loss:
            self.best_loss = val_loss
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
                if self.verbose:
                    print(f"EarlyStopping triggered at patience {self.patience}")

        return self.early_stop


def train_single(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    config: TrainingConfig,
) -> float:
    """
    Trains the model for one epoch and returns val_accuracy.

    Args:
        model: nn.Module
        train_loader: DataLoader with training data
        val_loader: DataLoader with validation data
        config: TrainingConfig

    Returns:
        float: validation accuracy
    """
    device = torch.device(config.device)
    model = model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )

    early_stopping = EarlyStopping(
        patience=config.early_stopping_patience, verbose=config.verbose
    )

    best_val_acc = 0.0
    train_losses = []

    for epoch in range(config.epochs):
        # TRAIN PHASE
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0

        for inputs, targets in train_loader:
            inputs, targets = inputs.to(device), targets.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            _, predicted = outputs.max(1)
            train_correct += predicted.eq(targets).sum().item()
            train_total += targets.size(0)

        train_loss = train_loss / len(train_loader)
        train_acc = train_correct / train_total
        train_losses.append(train_loss)

        # VAL PHASE
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0

        with torch.no_grad():
            for inputs, targets in val_loader:
                inputs, targets = inputs.to(device), targets.to(device)

                outputs = model(inputs)
                loss = criterion(outputs, targets)

                val_loss += loss.item()
                _, predicted = outputs.max(1)
                val_correct += predicted.eq(targets).sum().item()
                val_total += targets.size(0)

        val_loss = val_loss / len(val_loader)
        val_acc = val_correct / val_total

        if config.verbose and epoch % 10 == 0:
            print(
                f"  Epoch {epoch+1}/{config.epochs} | "
                f"Train loss: {train_loss:.4f}, acc: {train_acc:.4f} | "
                f"Val loss: {val_loss:.4f}, acc: {val_acc:.4f}"
            )

        # Best val accuracy
        if val_acc > best_val_acc:
            best_val_acc = val_acc

        # Early stopping
        if early_stopping(val_loss):
            break

    return best_val_acc


def train_with_cv(
    model: nn.Module,
    X: np.ndarray,
    y: np.ndarray,
    n_splits: int = 5,
    config: Optional[TrainingConfig] = None,
) -> float:
    """
    Trains the model using Stratified K-Fold Cross-Validation.

    Args:
        model: nn.Module (will be cloned for each fold)
        X: feature array (N, features)
        y: target array (N,)
        n_splits: number of folds (default 5)
        config: TrainingConfig or None (uses default)

    Returns:
        float: average validation accuracy across all folds
    """
    if config is None:
        config = TrainingConfig()

    # Validate that X is 2D or 3D.
    if X.ndim == 1:
        X = X.reshape(-1, 1)
    elif X.ndim > 3:
        X = X.reshape(X.shape[0], -1)  # Flatten for models that expect 2D input

    if config.verbose:
        print(f"Starting {n_splits}-Fold CV | Shape: X={X.shape}, y={y.shape}")

    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    fold_accuracies = []

    for fold_idx, (train_idx, val_idx) in enumerate(skf.split(X, y)):
        if config.verbose:
            print(f"\nFold {fold_idx + 1}/{n_splits}")

        X_train, X_val = X[train_idx], X[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]

        # Convert to Tensor
        X_train_tensor = torch.from_numpy(X_train).float()
        y_train_tensor = torch.from_numpy(y_train).long()
        X_val_tensor = torch.from_numpy(X_val).float()
        y_val_tensor = torch.from_numpy(y_val).long()

        # DataLoaders
        train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
        val_dataset = TensorDataset(X_val_tensor, y_val_tensor)

        train_loader = DataLoader(
            train_dataset,
            batch_size=config.batch_size,
            shuffle=True,
            num_workers=0,
        )
        val_loader = DataLoader(
            val_dataset,
            batch_size=config.batch_size,
            shuffle=False,
            num_workers=0,
        )

        # Clone model for this fold.
        import copy
        fold_model = copy.deepcopy(model)

        # Train
        fold_acc = train_single(fold_model, train_loader, val_loader, config)
        fold_accuracies.append(fold_acc)

        if config.verbose:
            print(f"  Fold {fold_idx + 1} val_acc: {fold_acc:.4f}")

    mean_accuracy = np.mean(fold_accuracies)
    std_accuracy = np.std(fold_accuracies)

    if config.verbose:
        print(f"\n{'='*60}")
        print(f"CV Results: {mean_accuracy:.4f} (+/- {std_accuracy:.4f})")
        print(f"Fold accuracies: {[f'{acc:.4f}' for acc in fold_accuracies]}")
        print(f"{'='*60}")

    return mean_accuracy
