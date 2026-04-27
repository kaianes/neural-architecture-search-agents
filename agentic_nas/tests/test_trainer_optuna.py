import numpy as np
import pytest

pytest.importorskip("torch")
pytest.importorskip("optuna")

from agentic_nas.agents import trainer_agent as trainer_agent_module
from agentic_nas.agents.trainer_agent import TrainerAgent
from agentic_nas.training import optuna_controller


def test_optuna_controller_runs_one_fast_tpe_trial(monkeypatch):
    monkeypatch.setattr(optuna_controller, "build_model", lambda **kwargs: object())
    monkeypatch.setattr(optuna_controller, "train_with_cv", lambda **kwargs: 0.8)
    monkeypatch.setattr(
        optuna_controller,
        "evaluate",
        lambda **kwargs: {"accuracy": 0.9},
    )

    X_train = np.zeros((8, 4), dtype=np.float32)
    y_train = np.array([0, 1, 0, 1, 0, 1, 0, 1], dtype=np.int64)
    X_test = np.zeros((4, 4), dtype=np.float32)
    y_test = np.array([0, 1, 0, 1], dtype=np.int64)

    result = optuna_controller.run_optimization_for_family(
        family_name="mlp",
        search_space_spec={
            "hidden_units": {"type": "categorical", "values": [8]},
            "num_layers": {"type": "int", "min": 1, "max": 2, "step": 1},
        },
        X_train=X_train,
        y_train=y_train,
        X_test=X_test,
        y_test=y_test,
        input_shape=(4,),
        num_classes=2,
        n_trials=1,
        task_type="binary_classification",
    )

    assert result["sampler"] == "TPESampler"
    assert result["n_trials_completed"] == 1
    assert result["best_value"] == pytest.approx(0.85)
    assert result["best_params"]
    assert len(result["trial_history"]) == 1
    assert result["trial_history"][0]["state"] == "COMPLETE"


def test_trainer_returns_expected_nas_result_shape(monkeypatch, builder_output, tiny_training_csv):
    def fake_run_optimization_for_family(
        family_name,
        search_space_spec,
        X_train,
        y_train,
        X_test,
        y_test,
        input_shape,
        num_classes,
        n_trials,
        task_type,
    ):
        best_value = 0.91 if family_name == "mlp" else 0.82
        return {
            "study_name": f"agentic_nas_{family_name}",
            "n_trials_completed": n_trials,
            "best_value": best_value,
            "best_params": {"fixture_param": family_name},
            "best_trial_number": 0,
            "trial_history": [
                {
                    "trial_id": 0,
                    "params": {"fixture_param": family_name},
                    "value": best_value,
                    "state": "COMPLETE",
                }
            ],
            "sampler": "TPESampler",
            "family_name": family_name,
        }

    monkeypatch.setattr(
        trainer_agent_module,
        "run_optimization_for_family",
        fake_run_optimization_for_family,
    )

    result = TrainerAgent().run_from_builder_and_csv(
        builder_json=builder_output,
        dataset_path=str(tiny_training_csv),
        n_trials_per_family=1,
        test_size=0.2,
    )

    assert result["selected_families"] == ["cnn1d", "mlp"]
    assert result["best_overall_family"] == "mlp"
    assert set(result["families_results"]) == {"cnn1d", "mlp"}

    for family_result in result["families_results"].values():
        assert family_result["n_trials_completed"] == 1
        assert family_result["best_value"] > 0
        assert family_result["best_params"]
        assert len(family_result["trial_history"]) == 1
        assert family_result["sampler"] == "TPESampler"
