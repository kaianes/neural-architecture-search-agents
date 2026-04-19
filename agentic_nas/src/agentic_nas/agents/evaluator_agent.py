"""
Evaluator Agent: interprets NAS results and prepares downstream handoff.
"""

from __future__ import annotations

import json
import statistics
from typing import Any, Dict, List, Tuple


class EvaluatorAgent:
    """Interprets trainer output with cost/quality and overfitting diagnostics."""

    def _normalize_trainer_output(self, trainer_output: Any) -> Dict[str, Any]:
        if isinstance(trainer_output, str):
            parsed = json.loads(trainer_output)
            return self._normalize_trainer_output(parsed)

        if not isinstance(trainer_output, dict):
            raise ValueError("trainer_output must be a JSON string or dictionary.")

        # Handles persisted task payload shape:
        # {"task_alias": "...", "output_json": {...}, "output_raw": "..."}
        if "output_json" in trainer_output and isinstance(trainer_output["output_json"], dict):
            return trainer_output["output_json"]

        # Handles crew payload shape with raw nested string
        if "raw" in trainer_output and isinstance(trainer_output["raw"], str):
            try:
                return self._normalize_trainer_output(trainer_output["raw"])
            except Exception:
                pass

        return trainer_output

    @staticmethod
    def _safe_float(value: Any, default: float = 0.0) -> float:
        try:
            return float(value)
        except (TypeError, ValueError):
            return default

    @staticmethod
    def _quality_tier(score: float) -> str:
        if score >= 0.9:
            return "excellent"
        if score >= 0.8:
            return "strong"
        if score >= 0.7:
            return "moderate"
        if score >= 0.6:
            return "weak_to_moderate"
        return "weak"

    @staticmethod
    def _cost_score_for_family(family: str, params: Dict[str, Any]) -> float:
        family_lower = str(family).lower()
        score = 1.0

        if "cnn" in family_lower:
            score += 2.0
        elif "rnn" in family_lower:
            score += 2.5
        elif "mlp" in family_lower:
            score += 1.0
        else:
            score += 1.5

        filters = params.get("filters", 0)
        hidden_units = params.get("hidden_units", 0)
        num_blocks = params.get("num_blocks", params.get("num_conv_layers", 1))
        num_layers = params.get("num_layers", 1)

        score += EvaluatorAgent._safe_float(filters, 0.0) / 64.0
        score += EvaluatorAgent._safe_float(hidden_units, 0.0) / 256.0
        score += max(EvaluatorAgent._safe_float(num_blocks, 1.0) - 1.0, 0.0) * 0.6
        score += max(EvaluatorAgent._safe_float(num_layers, 1.0) - 1.0, 0.0) * 0.4
        return score

    @staticmethod
    def _cost_tier(cost_score: float) -> str:
        if cost_score >= 5.0:
            return "high"
        if cost_score >= 3.0:
            return "medium"
        return "low"

    @staticmethod
    def _extract_values(trial_history: List[Dict[str, Any]]) -> List[float]:
        values: List[float] = []
        for trial in trial_history:
            values.append(EvaluatorAgent._safe_float(trial.get("value"), default=0.0))
        return values

    def evaluate(self, trainer_output: Any) -> Dict[str, Any]:
        trainer = self._normalize_trainer_output(trainer_output)
        families_results = trainer.get("families_results", {})

        if not families_results:
            raise ValueError("No families_results found in trainer output.")

        ranking: List[Dict[str, Any]] = []
        failed_families: List[str] = []

        for family, result in families_results.items():
            if "error" in result:
                failed_families.append(family)
                continue

            best_value = self._safe_float(result.get("best_value"), 0.0)
            best_params = result.get("best_params", {}) or {}
            trial_values = self._extract_values(result.get("trial_history", []) or [])
            value_std = statistics.pstdev(trial_values) if len(trial_values) > 1 else 0.0
            value_mean = statistics.fmean(trial_values) if trial_values else best_value

            cost_score = self._cost_score_for_family(family, best_params)
            ranking.append(
                {
                    "family": family,
                    "best_value": best_value,
                    "mean_trial_value": value_mean,
                    "trial_value_std": value_std,
                    "n_trials_completed": int(result.get("n_trials_completed", len(trial_values) or 0)),
                    "cost_score": round(cost_score, 3),
                    "efficiency_tier": self._cost_tier(cost_score),
                    "quality_tier": self._quality_tier(best_value),
                    "best_params": best_params,
                }
            )

        if not ranking:
            raise ValueError("All families failed; evaluator cannot rank models.")

        ranking.sort(key=lambda x: x["best_value"], reverse=True)

        selected_family = trainer.get("best_overall_family")
        selected_entry = next((r for r in ranking if r["family"] == selected_family), ranking[0])
        top_entry = ranking[0]
        second_entry = ranking[1] if len(ranking) > 1 else None

        profile_text = " ".join(
            str(v) for v in (trainer.get("profile_summary", {}) or {}).values()
        ).lower()
        small_data_signal = any(
            token in profile_text
            for token in ["small sample", "overfitting", "high dimensionality", "low data volume"]
        )

        overfit_risk_points = 0
        if small_data_signal:
            overfit_risk_points += 2
        if selected_entry["efficiency_tier"] == "high":
            overfit_risk_points += 2
        elif selected_entry["efficiency_tier"] == "medium":
            overfit_risk_points += 1
        if selected_entry["trial_value_std"] > 0.06:
            overfit_risk_points += 1

        if overfit_risk_points >= 4:
            overfit_risk = "high"
        elif overfit_risk_points >= 2:
            overfit_risk = "medium"
        else:
            overfit_risk = "low"

        value_gap_to_second = (
            selected_entry["best_value"] - second_entry["best_value"] if second_entry else 0.0
        )
        selection_reasons = [
            f"Highest observed objective value ({selected_entry['best_value']:.4f}).",
            f"Efficiency tier is {selected_entry['efficiency_tier']} with cost score {selected_entry['cost_score']}.",
        ]
        if second_entry:
            selection_reasons.append(
                f"Margin vs second option ({second_entry['family']}): {value_gap_to_second:.4f}."
            )

        if overfit_risk == "high":
            overfit_notes = [
                "Model complexity and/or data regime indicate elevated overfitting risk.",
                "Use stricter regularization and additional validation before final acceptance.",
            ]
        elif overfit_risk == "medium":
            overfit_notes = [
                "Some instability/complexity signals exist; monitor variance in additional runs.",
            ]
        else:
            overfit_notes = [
                "No strong overfitting signal from available objective history.",
            ]

        handoff_notes = [
            f"Adopt '{selected_entry['family']}' as current best candidate with params: {selected_entry['best_params']}.",
            "Run a confirmatory experiment with more trials and fixed seed for reproducibility.",
            "Track accuracy versus compute cost in the next iteration to validate deployment fit.",
        ]

        if failed_families:
            handoff_notes.append(
                f"Investigate failed families before final architecture freeze: {failed_families}."
            )

        return {
            "profile_summary": trainer.get("profile_summary", {}),
            "selected_model": {
                "family": selected_entry["family"],
                "best_value": selected_entry["best_value"],
                "best_params": selected_entry["best_params"],
                "selection_reasons": selection_reasons,
            },
            "performance_summary": {
                "task_type": trainer.get("task_type"),
                "num_classes": trainer.get("num_classes"),
                "input_shape": trainer.get("input_shape"),
                "ranking": ranking,
            },
            "cost_quality_tradeoff": {
                "chosen_family_efficiency_tier": selected_entry["efficiency_tier"],
                "chosen_family_quality_tier": selected_entry["quality_tier"],
                "best_value_margin_vs_next": round(value_gap_to_second, 6) if second_entry else None,
                "summary": (
                    f"Chosen family balances score {selected_entry['best_value']:.4f} "
                    f"with {selected_entry['efficiency_tier']} estimated cost."
                ),
            },
            "overfitting_assessment": {
                "risk_level": overfit_risk,
                "evidence": {
                    "small_data_signal": small_data_signal,
                    "chosen_family_trial_std": round(selected_entry["trial_value_std"], 6),
                    "chosen_family_efficiency_tier": selected_entry["efficiency_tier"],
                },
                "notes": overfit_notes,
            },
            "failed_families": failed_families,
            "handoff_notes": handoff_notes,
        }
