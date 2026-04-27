import json
import shutil
import sys
import uuid
from pathlib import Path

import pandas as pd
import pytest


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))


@pytest.fixture
def tmp_path():
    base_dir = PROJECT_ROOT / "data" / "outputs" / "pytest_tmp"
    case_dir = base_dir / f"case_{uuid.uuid4().hex}"
    case_dir.mkdir(parents=True, exist_ok=False)
    try:
        yield case_dir
    finally:
        resolved_base = base_dir.resolve()
        resolved_case = case_dir.resolve()
        if resolved_base in resolved_case.parents and resolved_case.exists():
            shutil.rmtree(resolved_case, ignore_errors=True)


@pytest.fixture
def ecg_2500_csv(tmp_path):
    feature_columns = [str(idx) for idx in range(2500)]
    rows = []
    rows.append({**{col: 0.0 for col in feature_columns}, "Target": "Normal"})
    rows.append({**{col: 1.0 for col in feature_columns}, "Target": "Sleep Apnea"})
    rows.append({**{col: 0.0 for col in feature_columns}, "Target": "Normal"})
    rows.append({**{col: 2.0 for col in feature_columns}, "Target": "Sleep Apnea"})
    rows[-1]["7"] = None

    csv_path = tmp_path / "ecg_2500.csv"
    pd.DataFrame(rows).to_csv(csv_path, index=False)
    return csv_path


@pytest.fixture
def tiny_training_csv(tmp_path):
    rows = []
    for idx in range(20):
        label = "Normal" if idx % 2 == 0 else "Sleep Apnea"
        rows.append(
            {
                **{str(feature_idx): float(idx + feature_idx) for feature_idx in range(8)},
                "Target": label,
            }
        )

    csv_path = tmp_path / "tiny_train.csv"
    pd.DataFrame(rows).to_csv(csv_path, index=False)
    return csv_path


@pytest.fixture
def builder_output():
    fixture_path = Path(__file__).parent / "fixtures" / "builder_output.json"
    return json.loads(fixture_path.read_text(encoding="utf-8"))


@pytest.fixture
def trainer_output():
    return {
        "profile_summary": {
            "data_modality": "1D Time-Series/Signal (ECG)",
            "likely_task_type": "Binary Classification",
            "key_constraints": ["Small sample size"],
        },
        "selected_families": ["cnn1d", "mlp"],
        "families_results": {
            "cnn1d": {
                "study_name": "agentic_nas_cnn1d",
                "n_trials_completed": 2,
                "best_value": 0.82,
                "best_params": {"filters": 16, "num_blocks": 1},
                "best_trial_number": 1,
                "trial_history": [
                    {"trial_id": 0, "params": {"filters": 8}, "value": 0.72, "state": "COMPLETE"},
                    {"trial_id": 1, "params": {"filters": 16}, "value": 0.82, "state": "COMPLETE"},
                ],
                "sampler": "TPESampler",
            },
            "mlp": {
                "study_name": "agentic_nas_mlp",
                "n_trials_completed": 2,
                "best_value": 0.91,
                "best_params": {"hidden_units": 16, "num_layers": 1},
                "best_trial_number": 0,
                "trial_history": [
                    {"trial_id": 0, "params": {"hidden_units": 16}, "value": 0.91, "state": "COMPLETE"},
                    {"trial_id": 1, "params": {"hidden_units": 8}, "value": 0.85, "state": "COMPLETE"},
                ],
                "sampler": "TPESampler",
            },
        },
        "best_overall_family": "mlp",
        "task_type": "binary_classification",
        "num_classes": 2,
        "input_shape": [8],
    }


@pytest.fixture
def evaluator_output(trainer_output):
    return {
        "profile_summary": trainer_output["profile_summary"],
        "selected_model": {
            "family": "mlp",
            "best_value": 0.91,
            "best_params": {"hidden_units": 16, "num_layers": 1},
            "selection_reasons": ["Highest observed objective value (0.9100)."],
        },
        "performance_summary": {
            "task_type": "binary_classification",
            "num_classes": 2,
            "input_shape": [8],
            "ranking": [
                {
                    "family": "mlp",
                    "best_value": 0.91,
                    "mean_trial_value": 0.88,
                    "trial_value_std": 0.03,
                    "n_trials_completed": 2,
                    "cost_score": 2.1,
                    "efficiency_tier": "low",
                    "quality_tier": "excellent",
                    "best_params": {"hidden_units": 16, "num_layers": 1},
                },
                {
                    "family": "cnn1d",
                    "best_value": 0.82,
                    "mean_trial_value": 0.77,
                    "trial_value_std": 0.05,
                    "n_trials_completed": 2,
                    "cost_score": 3.1,
                    "efficiency_tier": "medium",
                    "quality_tier": "strong",
                    "best_params": {"filters": 16, "num_blocks": 1},
                },
            ],
        },
        "cost_quality_tradeoff": {
            "chosen_family_efficiency_tier": "low",
            "chosen_family_quality_tier": "excellent",
            "best_value_margin_vs_next": 0.09,
            "summary": "MLP is the highest-scoring fixture model.",
        },
        "overfitting_assessment": {
            "risk_level": "low",
            "evidence": {},
            "notes": ["No strong overfitting signal from available objective history."],
        },
        "failed_families": [],
        "handoff_notes": ["Run a confirmatory experiment."],
    }
