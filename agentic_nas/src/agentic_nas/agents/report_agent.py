"""
Report Agent: builds a final markdown report from evaluator outputs.
"""

from __future__ import annotations

import json
from typing import Any, Dict, List


class ReportAgent:
    """Generates a project-style markdown report with required sections."""

    def _normalize_evaluator_output(self, evaluator_output: Any) -> Dict[str, Any]:
        if isinstance(evaluator_output, str):
            parsed = json.loads(evaluator_output)
            return self._normalize_evaluator_output(parsed)

        if not isinstance(evaluator_output, dict):
            raise ValueError("evaluator_output must be a JSON string or dictionary.")

        if "output_json" in evaluator_output and isinstance(evaluator_output["output_json"], dict):
            return evaluator_output["output_json"]

        if "raw" in evaluator_output and isinstance(evaluator_output["raw"], str):
            try:
                return self._normalize_evaluator_output(evaluator_output["raw"])
            except Exception:
                pass

        return evaluator_output

    @staticmethod
    def _to_bullets(items: List[str]) -> str:
        if not items:
            return "- Not available."
        return "\n".join(f"- {item}" for item in items)

    @staticmethod
    def _format_params(params: Dict[str, Any]) -> str:
        if not params:
            return "- Not available."
        lines = []
        for key in sorted(params.keys()):
            lines.append(f"- `{key}`: `{params[key]}`")
        return "\n".join(lines)

    @staticmethod
    def _format_families(ranking: List[Dict[str, Any]]) -> str:
        if not ranking:
            return "- Not available."
        lines = []
        for entry in ranking:
            family = entry.get("family", "unknown")
            best_value = entry.get("best_value", "n/a")
            quality = entry.get("quality_tier", "n/a")
            efficiency = entry.get("efficiency_tier", "n/a")
            lines.append(
                f"- `{family}`: best_value={best_value}, quality={quality}, efficiency={efficiency}"
            )
        return "\n".join(lines)

    def generate_report(self, evaluator_output: Any) -> str:
        data = self._normalize_evaluator_output(evaluator_output)

        profile_summary = data.get("profile_summary", {}) or {}
        selected_model = data.get("selected_model", {}) or {}
        performance_summary = data.get("performance_summary", {}) or {}
        tradeoff = data.get("cost_quality_tradeoff", {}) or {}
        overfit = data.get("overfitting_assessment", {}) or {}
        handoff_notes = data.get("handoff_notes", []) or []
        failed_families = data.get("failed_families", []) or []

        identified_problem = (
            profile_summary.get("likely_task_type")
            or performance_summary.get("task_type")
            or "Not available."
        )
        data_modality = profile_summary.get("data_modality", "Not available.")

        chosen_strategy = (
            "Agentic NAS with profiler -> builder -> trainer (Optuna) -> evaluator, "
            "using bounded search spaces and ranking models by objective score plus cost/quality trade-off."
        )

        architecture_families = self._format_families(performance_summary.get("ranking", []) or [])
        best_family = selected_model.get("family", "Not available.")
        best_value = selected_model.get("best_value", "Not available.")
        best_params = self._format_params(selected_model.get("best_params", {}) or {})
        selection_reasons = self._to_bullets(selected_model.get("selection_reasons", []) or [])

        final_performance_lines = [
            f"- Best family: `{best_family}`",
            f"- Best objective value: `{best_value}`",
        ]
        margin = tradeoff.get("best_value_margin_vs_next")
        if margin is not None:
            final_performance_lines.append(f"- Margin vs next candidate: `{margin}`")
        final_performance = "\n".join(final_performance_lines)

        limitations_notes = list(overfit.get("notes", []) or [])
        if failed_families:
            limitations_notes.append(f"Failed families during optimization: {failed_families}.")
        if not limitations_notes:
            limitations_notes = ["No explicit limitations reported."]

        next_steps_notes = handoff_notes if handoff_notes else ["Define next iteration based on evaluation insights."]

        markdown = (
            "# NAS Optimization Report\n\n"
            "## Identified problem\n"
            f"{identified_problem}\n\n"
            "## Data modality\n"
            f"{data_modality}\n\n"
            "## Chosen strategy\n"
            f"{chosen_strategy}\n\n"
            "## Architecture families considered\n"
            f"{architecture_families}\n\n"
            "## Best configuration found\n"
            f"- Selected family: `{best_family}`\n"
            f"{best_params}\n\n"
            "## Final performance\n"
            f"{final_performance}\n\n"
            "## Justification for the choice\n"
            f"{selection_reasons}\n"
            f"- Trade-off summary: {tradeoff.get('summary', 'Not available.')}\n\n"
            "## Limitations\n"
            f"{self._to_bullets(limitations_notes)}\n\n"
            "## Next steps\n"
            f"{self._to_bullets(next_steps_notes)}\n"
        )

        return markdown

