from __future__ import annotations

from typing import Any, Dict, List

from contracts import TrialRecord


class ProposalAgent:
    """Rule-based proposal agent that can be replaced by an LLM strategy later."""

    def __init__(self, strategy: str = "rule_based", max_initial_suggestions: int = 5) -> None:
        self.strategy = strategy
        self.max_initial_suggestions = max(1, max_initial_suggestions)

    def propose_trials(self, cfg: Dict[str, Any], memory_hits: List[TrialRecord]) -> List[Dict[str, Any]]:
        if self.strategy != "rule_based":
            return []

        suggestions: List[Dict[str, Any]] = []
        for hit in memory_hits:
            if not hit.params:
                continue
            suggestions.append(dict(hit.params))
            if len(suggestions) >= self.max_initial_suggestions:
                break

        # If memory is empty, add one stable baseline suggestion.
        if not suggestions:
            suggestions.append(
                {
                    "conv_channels": 32,
                    "kernel_size": 3,
                    "dropout": 0.1,
                    "lr": float(cfg.get("lr", 1e-2)),
                }
            )

        return suggestions[: self.max_initial_suggestions]

    def suggest_guidance(self, memory_hits: List[TrialRecord]) -> Dict[str, Any]:
        if not memory_hits:
            return {}

        conv_channels = sorted({int(r.params.get("conv_channels", 32)) for r in memory_hits if r.params})
        kernel_size = sorted({int(r.params.get("kernel_size", 3)) for r in memory_hits if r.params})
        return {
            "conv_channels": conv_channels[:4] if conv_channels else [16, 32, 48, 64],
            "kernel_size": kernel_size[:2] if kernel_size else [3, 5],
        }
