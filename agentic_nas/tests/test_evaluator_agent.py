from agentic_nas.agents.evaluator_agent import EvaluatorAgent


def test_evaluator_ranks_by_best_value_descending(trainer_output):
    result = EvaluatorAgent().evaluate(trainer_output)
    ranking = result["performance_summary"]["ranking"]

    assert [entry["family"] for entry in ranking] == ["mlp", "cnn1d"]
    assert ranking[0]["best_value"] >= ranking[1]["best_value"]
    assert result["selected_model"]["family"] == "mlp"
    assert "performance_summary" in result
    assert "cost_quality_tradeoff" in result
    assert "overfitting_assessment" in result
    assert result["profile_summary"]["data_modality"] == "1D Time-Series/Signal (ECG)"


def test_evaluator_tie_uses_best_overall_family_when_available(trainer_output):
    trainer_output["families_results"]["cnn1d"]["best_value"] = 0.91
    trainer_output["families_results"]["cnn1d"]["trial_history"][-1]["value"] = 0.91
    trainer_output["best_overall_family"] = "cnn1d"

    result = EvaluatorAgent().evaluate(trainer_output)

    assert result["selected_model"]["family"] == "cnn1d"
    assert result["cost_quality_tradeoff"]["best_value_margin_vs_next"] == 0.0
