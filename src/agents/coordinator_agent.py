from typing import Any, Dict, Tuple
from utils.logger import get_logger


class CoordinatorAgent:
    """
    Orchestrates the NAS flow by delegating to SearchAgent, then summarizing with EvaluationAgent.
    """

    def __init__(self, search_agent, evaluation_agent, logger=None):
        self.search_agent = search_agent
        self.evaluation_agent = evaluation_agent
        self.logger = logger or get_logger(__name__)

    def execute(self, cfg: Dict[str, Any], device) -> Dict[str, Any]:
        # Run search; support both tuple return (best_value, best_params)
        # and dict-based richer results for future extensions.
        search_result = self.search_agent.run(cfg, device)

        if isinstance(search_result, dict):
            history = search_result
        else:
            best_value, best_params = self._unpack_tuple(search_result)
            history = {"best_value": best_value, "best_params": best_params}

        summary = self.evaluation_agent.summarize(history)
        self._log_summary(summary)
        return summary

    def _unpack_tuple(self, result: Tuple[Any, Any]) -> Tuple[float, Dict[str, Any]]:
        if not isinstance(result, tuple) or len(result) < 2:
            raise ValueError(
                "SearchAgent.run must return (best_value, best_params) or a dict with keys."
            )
        return result[0], result[1]

    def _log_summary(self, summary: Dict[str, Any]) -> None:
        best_value = summary.get("best_value")
        best_params = summary.get("best_params")
        self.logger.info(f"Coordinator summary -> best_value: {best_value}, best_params: {best_params}")
