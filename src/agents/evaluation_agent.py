from typing import Dict

class EvaluationAgent:
    def summarize(self, history: Dict):
        # Extend with richer explainability later
        return {
            "best_value": history.get("best_value"),
            "best_params": history.get("best_params"),
        }

