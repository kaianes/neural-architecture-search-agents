"""
Evaluator: Calcula métricas de avaliação (accuraria, F1, ROC-AUC, etc)
"""

from typing import Dict, Tuple, Optional
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    roc_auc_score,
    precision_score,
    recall_score,
    confusion_matrix,
)


def evaluate(
    model: nn.Module,
    X_test: np.ndarray,
    y_test: np.ndarray,
    batch_size: int = 32,
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
    task_type: str = "binary_classification",
) -> Dict[str, float]:
    """
    Evalua modelo em test set e retorna métricas.

    Args:
        model: nn.Module já treinado
        X_test: feature array (N, features)
        y_test: target array (N,)
        batch_size: batch size para avaliação
        device: cuda ou cpu
        task_type: 'binary_classification' ou 'multiclass'

    Returns:
        dict com métricas: {'accuracy': ..., 'f1': ..., 'roc_auc': ..., ...}
    """
    device_obj = torch.device(device)
    model = model.to(device_obj)
    model.eval()

    # Converter para tensor
    if X_test.ndim == 1:
        X_test = X_test.reshape(-1, 1)
    elif X_test.ndim > 3:
        X_test = X_test.reshape(X_test.shape[0], -1)

    X_test_tensor = torch.from_numpy(X_test).float()
    y_test_tensor = torch.from_numpy(y_test).long()

    # DataLoader
    test_dataset = TensorDataset(X_test_tensor, y_test_tensor)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    # Prediction
    all_preds = []
    all_probs = []
    all_targets = []

    with torch.no_grad():
        for inputs, targets in test_loader:
            inputs, targets = inputs.to(device_obj), targets.to(device_obj)

            outputs = model(inputs)

            # Predictions
            _, predicted = outputs.max(1)
            all_preds.extend(predicted.cpu().numpy())

            # Probabilities (softmax)
            probs = torch.softmax(outputs, dim=1)
            all_probs.extend(probs.cpu().numpy())

            all_targets.extend(targets.cpu().numpy())

    all_preds = np.array(all_preds)
    all_probs = np.array(all_probs)
    all_targets = np.array(all_targets)

    # Calculate metrics
    metrics = {}

    # Accuracy
    metrics["accuracy"] = accuracy_score(all_targets, all_preds)

    # Precision, Recall, F1
    if task_type == "binary_classification":
        metrics["precision"] = precision_score(all_targets, all_preds)
        metrics["recall"] = recall_score(all_targets, all_preds)
        metrics["f1"] = f1_score(all_targets, all_preds)

        # ROC-AUC (binary)
        try:
            metrics["roc_auc"] = roc_auc_score(all_targets, all_probs[:, 1])
        except Exception as e:
            metrics["roc_auc"] = 0.0
            print(f"Warning: could not compute ROC-AUC: {e}")

    elif task_type == "multiclass":
        metrics["precision"] = precision_score(all_targets, all_preds, average="weighted")
        metrics["recall"] = recall_score(all_targets, all_preds, average="weighted")
        metrics["f1"] = f1_score(all_targets, all_preds, average="weighted")

    # Confusion matrix
    cm = confusion_matrix(all_targets, all_preds)
    metrics["confusion_matrix"] = cm.tolist()

    return metrics


def evaluate_from_scores(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_proba: Optional[np.ndarray] = None,
    task_type: str = "binary_classification",
) -> Dict[str, float]:
    """
    Calcula métricas a partir de predictions já computadas.

    Args:
        y_true: ground truth labels
        y_pred: predicted labels
        y_proba: predicted probabilities (opcional, para ROC-AUC)
        task_type: tipo de tarefa

    Returns:
        dict com métricas
    """
    metrics = {}

    metrics["accuracy"] = accuracy_score(y_true, y_pred)

    if task_type == "binary_classification":
        metrics["precision"] = precision_score(y_true, y_pred, zero_division=0)
        metrics["recall"] = recall_score(y_true, y_pred, zero_division=0)
        metrics["f1"] = f1_score(y_true, y_pred, zero_division=0)

        if y_proba is not None:
            try:
                metrics["roc_auc"] = roc_auc_score(y_true, y_proba[:, 1])
            except Exception:
                metrics["roc_auc"] = 0.0

    elif task_type == "multiclass":
        metrics["precision"] = precision_score(
            y_true, y_pred, average="weighted", zero_division=0
        )
        metrics["recall"] = recall_score(
            y_true, y_pred, average="weighted", zero_division=0
        )
        metrics["f1"] = f1_score(y_true, y_pred, average="weighted", zero_division=0)

    cm = confusion_matrix(y_true, y_pred)
    metrics["confusion_matrix"] = cm.tolist()

    return metrics
