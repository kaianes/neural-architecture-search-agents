from __future__ import annotations

from typing import Any, Dict, Optional

from models.simple_cnn import SimpleCNN
from search_spaces.base import SearchSpaceSpec


def _pick(guidance: Optional[Dict[str, Any]], key: str, default: Any) -> Any:
    if not guidance:
        return default
    value = guidance.get(key)
    return default if value is None else value


class SimpleCNNSearchSpace(SearchSpaceSpec):
    def __init__(self) -> None:
        super().__init__(name="simple_cnn_default")

    def sample(self, trial: Any, guidance: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        conv_choices = _pick(guidance, "conv_channels", [16, 32, 48, 64])
        kernel_choices = _pick(guidance, "kernel_size", [3, 5])

        return {
            "conv_channels": trial.suggest_categorical("conv_channels", conv_choices),
            "kernel_size": trial.suggest_categorical("kernel_size", kernel_choices),
            "dropout": trial.suggest_float("dropout", 0.0, 0.5),
            "lr": trial.suggest_float("lr", 1e-3, 1e-1, log=True),
        }

    def build_model(self, params: Dict[str, Any], dataset_meta: Dict[str, Any]) -> Any:
        return SimpleCNN(
            in_ch=int(dataset_meta["in_ch"]),
            num_classes=int(dataset_meta.get("num_classes", 10)),
            conv_channels=int(params["conv_channels"]),
            kernel_size=int(params["kernel_size"]),
            dropout=float(params["dropout"]),
        )
