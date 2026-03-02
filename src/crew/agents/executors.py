"""
BuilderAgent: Constructs PyTorch models from specifications.
TrainerAgent: Trains models and collects metrics.
EvaluatorAgent: Calculates final scores (accuracy + efficiency).
CriticAgent: Identifies errors, generates feedback, and suggests improvements.
"""
from __future__ import annotations

from typing import Any, Dict, Optional
import torch
import torch.nn as nn
from crew.agents.base import BaseNASAgent
from crew.shared_state.context import SearchContext, TrialState
from crew.reasoning.patterns import CriticalReflection
from models.simple_cnn import SimpleCNN
from utils.metrics import accuracy, count_params, try_flops
from utils.logger import get_logger


class BuilderAgent(BaseNASAgent):
    """Constructs and validates PyTorch models."""
    
    def __init__(self):
        super().__init__(
            name="BuilderAgent",
            role="Model Constructor",
            description="Transforms architecture specifications into trainable PyTorch models"
        )
    
    async def execute(self, context: SearchContext, task: Dict[str, Any]) -> Dict[str, Any]:
        """
        Constructs a model from an architecture specification.
        
        Expected task:
            {
                "type": "build_model",
                "trial_id": 1,
                "architecture": {
                    "conv_channels": 32,
                    "kernel_size": 3,
                    "dropout": 0.1
                },
                "dataset": "MNIST"
            }
        """
        trial_id = task.get("trial_id")
        architecture = task.get("architecture", {})
        dataset = task.get("dataset", "MNIST")
        device = task.get("device", context.device)
        
        observation = f"Building model for trial {trial_id}: {architecture}"
        self.add_react_step(
            observation=observation,
            reasoning="Validate architecture spec and instantiate model",
            action="Construct SimpleCNN",
            confidence=0.9
        )
        
        try:
            # Build model
            model = self._build_model(architecture, dataset)
            model = model.to(device)
            
            # Validate that it is trainable
            self._validate_model(model)
            
            # Collect model information
            param_count = count_params(model)
            model_summary = f"Model built: {param_count:,} parameters"
            
            self.log_reasoning(f"✅ Model valid and trainable: {param_count:,} params")
            
            return {
                "status": "success",
                "trial_id": trial_id,
                "model": model,
                "param_count": param_count,
                "message": model_summary,
                "react_trace": self.react_trace.to_string()
            }
        except Exception as e:
            error_msg = f"Failed to build model: {str(e)}"
            self.log_reasoning(f"❌ {error_msg}")
            return {
                "status": "failed",
                "trial_id": trial_id,
                "error": error_msg,
                "react_trace": self.react_trace.to_string()
            }
    
    def _build_model(self, architecture: Dict[str, Any], dataset: str) -> nn.Module:
        """Builds the model."""
        in_ch = 1 if dataset in ["MNIST", "FASHIONMNIST"] else 3
        num_classes = 10
        
        return SimpleCNN(
            in_ch=in_ch,
            num_classes=num_classes,
            conv_channels=int(architecture.get("conv_channels", 32)),
            kernel_size=int(architecture.get("kernel_size", 3)),
            dropout=float(architecture.get("dropout", 0.1))
        )
    
    def _validate_model(self, model: nn.Module) -> None:
        """Validates that the model is trainable."""
        # Check that it has parameters
        params = list(model.parameters())
        if not params:
            raise ValueError("Model has no trainable parameters")
        
        # Check that it has gradients
        if any(p.grad_fn is None for p in params if p.requires_grad):
            pass  # OK, gradients will be created in backward


class TrainerAgent(BaseNASAgent):
    """Trains models and collects metrics."""
    
    def __init__(self):
        super().__init__(
            name="TrainerAgent",
            role="Model Trainer",
            description="Trains models and collects training/validation metrics"
        )
    
    async def execute(self, context: SearchContext, task: Dict[str, Any]) -> Dict[str, Any]:
        """
        Trains a model.
        
        Expected task:
            {
                "type": "train_model",
                "trial_id": 1,
                "model": <nn.Module>,
                "train_loader": <DataLoader>,
                "val_loader": <DataLoader>,
                "epochs": 2,
                "device": "cuda"
            }
        """
        trial_id = task.get("trial_id")
        model = task.get("model")
        train_loader = task.get("train_loader")
        val_loader = task.get("val_loader")
        epochs = task.get("epochs", 2)
        device = task.get("device", context.device)
        
        observation = f"Training model for trial {trial_id} for {epochs} epochs"
        self.add_react_step(
            observation=observation,
            reasoning="Execute standard training loop with validation",
            action="Train and validate",
            confidence=0.9
        )
        
        try:
            optimizer = torch.optim.AdamW(model.parameters(), lr=0.01)
            criterion = nn.CrossEntropyLoss()
            
            metrics = {
                "train_losses": [],
                "train_accs": [],
                "val_losses": [],
                "val_accs": []
            }
            
            for epoch in range(epochs):
                # Train
                train_loss, train_acc = self._train_epoch(model, train_loader, device, optimizer, criterion)
                metrics["train_losses"].append(train_loss)
                metrics["train_accs"].append(train_acc)
                
                # Validate
                val_loss, val_acc = self._validate_epoch(model, val_loader, device, criterion)
                metrics["val_losses"].append(val_loss)
                metrics["val_accs"].append(val_acc)
                
                self.logger.info(f"[Trial {trial_id}] Epoch {epoch+1}/{epochs}: "
                                 f"train_acc={train_acc:.4f}, val_acc={val_acc:.4f}")
            
            final_train_acc = metrics["train_accs"][-1]
            final_val_acc = metrics["val_accs"][-1]
            
            self.log_reasoning(f"✅ Training complete. Final val_acc={final_val_acc:.4f}")
            
            return {
                "status": "success",
                "trial_id": trial_id,
                "model": model,
                "metrics": metrics,
                "final_train_acc": final_train_acc,
                "final_val_acc": final_val_acc,
                "react_trace": self.react_trace.to_string()
            }
        except Exception as e:
            error_msg = f"Training failed: {str(e)}"
            self.log_reasoning(f"❌ {error_msg}")
            return {
                "status": "failed",
                "trial_id": trial_id,
                "error": error_msg,
                "react_trace": self.react_trace.to_string()
            }
    
    def _train_epoch(self, model, loader, device, optimizer, criterion) -> tuple:
        """Trains one epoch."""
        model.train()
        total_loss, total_acc, n = 0.0, 0.0, 0
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            out = model(x)
            loss = criterion(out, y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * x.size(0)
            total_acc += accuracy(out.detach(), y) * x.size(0)
            n += x.size(0)
        return total_loss / n, total_acc / n
    
    def _validate_epoch(self, model, loader, device, criterion) -> tuple:
        """Validates one epoch."""
        model.eval()
        total_loss, total_acc, n = 0.0, 0.0, 0
        with torch.no_grad():
            for x, y in loader:
                x, y = x.to(device), y.to(device)
                out = model(x)
                loss = criterion(out, y)
                total_loss += loss.item() * x.size(0)
                total_acc += accuracy(out, y) * x.size(0)
                n += x.size(0)
        return total_loss / n, total_acc / n


class EvaluatorAgent(BaseNASAgent):
    """Evaluates models and calculates final scores."""
    
    def __init__(self):
        super().__init__(
            name="EvaluatorAgent",
            role="Performance Evaluator",
            description="Calculates final scores combining accuracy and efficiency metrics"
        )
    
    async def execute(self, context: SearchContext, task: Dict[str, Any]) -> Dict[str, Any]:
        """
        Evaluates a trained model.
        
        Expected task:
            {
                "type": "evaluate",
                "trial_id": 1,
                "model": <nn.Module>,
                "val_acc": 0.95,
                "metrics": {...},
                "dataset": "MNIST"
            }
        """
        trial_id = task.get("trial_id")
        model = task.get("model")
        val_acc = task.get("val_acc", 0.0)
        metrics = task.get("metrics", {})
        dataset = task.get("dataset", "MNIST")
        
        observation = f"Evaluating trial {trial_id}: val_acc={val_acc:.4f}"
        
        try:
            # Collect efficiency metrics
            param_count = count_params(model)
            flops = try_flops(model, input_size=(1, 1 if dataset in ["MNIST", "FASHIONMNIST"] else 3, 28 if dataset == "MNIST" else 32, 28 if dataset == "MNIST" else 32))
            
            # Combined score: 70% accuracy, 30% efficiency (parameters)
            efficiency_score = 1.0 / (1.0 + param_count / 100000.0)  # Penalize large models
            combined_score = 0.7 * val_acc + 0.3 * efficiency_score
            
            self.add_react_step(
                observation=observation,
                reasoning=f"Combine accuracy ({val_acc:.4f}) and efficiency ({efficiency_score:.4f}) for final score",
                action=f"Assign score={combined_score:.4f}",
                confidence=0.85
            )
            
            self.log_reasoning(f"✅ Score calculated: {combined_score:.4f} (acc={val_acc:.4f}, eff={efficiency_score:.4f})")
            
            return {
                "status": "success",
                "trial_id": trial_id,
                "accuracy": val_acc,
                "param_count": param_count,
                "flops": flops,
                "efficiency_score": efficiency_score,
                "combined_score": combined_score,
                "react_trace": self.react_trace.to_string()
            }
        except Exception as e:
            return {
                "status": "failed",
                "trial_id": trial_id,
                "error": str(e),
                "react_trace": self.react_trace.to_string()
            }


class CriticAgent(BaseNASAgent):
    """Critical agent: identifies problems and suggests improvements."""
    
    def __init__(self):
        super().__init__(
            name="CriticAgent",
            role="Quality Critic",
            description="Identifies overfitting, instability, and poor configurations; suggests improvements"
        )
    
    async def execute(self, context: SearchContext, task: Dict[str, Any]) -> Dict[str, Any]:
        """
        Critiques results of a trial.
        
        Expected task:
            {
                "type": "critique",
                "trial_id": 1,
                "metrics": {
                    "train_accs": [0.8, 0.85],
                    "val_accs": [0.79, 0.82]
                },
                "score": 0.82,
                "param_count": 50000
            }
        """
        trial_id = task.get("trial_id")
        metrics = task.get("metrics", {})
        score = task.get("score", 0.0)
        param_count = task.get("param_count", 0)
        
        observation = f"Analyzing trial {trial_id} for quality issues"
        
        self.add_react_step(
            observation=observation,
            reasoning="Check for overfitting, instability, poor performance patterns",
            action="Generate critique and suggestions",
            confidence=0.8
        )
        
        # Analysis
        reflection = CriticalReflection(
            decision_description=f"Trial {trial_id} configuration and training",
            confidence_before=0.7,
            confidence_after=0.8
        )
        
        # Detect problems
        train_accs = metrics.get("train_accs", [])
        val_accs = metrics.get("val_accs", [])
        
        if len(train_accs) > 0 and len(val_accs) > 0:
            gap = train_accs[-1] - val_accs[-1]
            
            if gap > 0.1:
                reflection.concerns.append(f"⚠️  Overfitting: train_acc={train_accs[-1]:.4f} > val_acc={val_accs[-1]:.4f} (gap={gap:.4f})")
                reflection.improvements.append("💡 Increase dropout or regularization")
            else:
                reflection.strengths.append(f"✅ Good generalization: gap={gap:.4f}")
        
        if score < 0.7:
            reflection.weaknesses.append(f"⚠️  Low performance: score={score:.4f}")
            reflection.improvements.append("💡 Try different learning rate or architecture")
        else:
            reflection.strengths.append(f"✅ Good performance: score={score:.4f}")
        
        if param_count > 200000:
            reflection.concerns.append(f"⚠️  Large model: {param_count:,} parameters")
            reflection.improvements.append("💡 Reduce conv_channels or add pruning")
        
        self.log_reasoning(f"Critique generated: {len(reflection.concerns)} concerns, {len(reflection.improvements)} suggestions")
        
        return {
            "status": "success",
            "trial_id": trial_id,
            "reflection": reflection.to_string(),
            "concerns": reflection.concerns,
            "improvements": reflection.improvements,
            "react_trace": self.react_trace.to_string()
        }
