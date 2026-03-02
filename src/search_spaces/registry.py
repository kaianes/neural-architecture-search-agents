from __future__ import annotations

from typing import Dict

from search_spaces.simple_cnn_default import SimpleCNNSearchSpace


def get_search_space(name: str):
    registry: Dict[str, object] = {
        "simple_cnn_default": SimpleCNNSearchSpace(),
        "simple_cnn": SimpleCNNSearchSpace(),
    }
    key = (name or "simple_cnn_default").strip().lower()
    if key not in registry:
        raise ValueError(f"Unsupported search space: {name}. Available: {sorted(registry)}")
    return registry[key]
