from typing import Any, Callable, Dict, Tuple, Union

class SearchAgent:
    """Wraps a NAS strategy (e.g., Optuna) and normalizes its output."""

    def __init__(self, strategy_fn: Callable, name: str = "search"):
        self.strategy_fn = strategy_fn
        self.name = name

    def run(self, cfg: Dict[str, Any], device) -> Dict[str, Any]:
        raw = self.strategy_fn(cfg, device)
        return self._normalize(raw)

    def _normalize(self, raw: Union[Dict[str, Any], Tuple[Any, Any]]) -> Dict[str, Any]:
        if isinstance(raw, dict):
            best_value = raw.get("best_value")
            best_params = raw.get("best_params")
            return {
                "strategy": self.name,
                "best_value": best_value,
                "best_params": best_params,
                "trials": raw.get("trials"),
                "attrs": raw.get("attrs"),
            }
        if not isinstance(raw, tuple) or len(raw) < 2:
            raise ValueError(
                "strategy_fn must return a dict with 'best_value'/'best_params' "
                "or a tuple(best_value, best_params)."
            )
        best_value, best_params = raw[0], raw[1]
        return {
            "strategy": self.name,
            "best_value": best_value,
            "best_params": best_params,
            "trials": None,
            "attrs": None,
        }
