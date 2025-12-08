from typing import Any, Dict, List, Optional


class EvaluationAgent:
    """
    Summarizes search history into a concise report.
    """

    def summarize(self, history: Dict[str, Any]) -> Dict[str, Any]:
        best_value = history.get("best_value")
        best_params = history.get("best_params")
        trials = history.get("trials") or []
        attrs = history.get("attrs") or {}

        leaderboard = self._top_trials(trials, limit=5)
        return {
            "best_value": best_value,
            "best_params": best_params,
            "trials": leaderboard,
            "attrs": attrs,
        }

    def _top_trials(self, trials: List[Dict[str, Any]], limit: int = 5) -> List[Dict[str, Any]]:
        if not trials:
            return []
        sorted_trials = sorted(trials, key=lambda t: t.get("value", float("-inf")), reverse=True)
        return sorted_trials[:limit]
