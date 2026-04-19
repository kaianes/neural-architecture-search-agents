"""Tools to execute the trainer pipeline from CrewAI tasks."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from crewai.tools import tool

from agentic_nas.agents.trainer_agent import TrainerAgent


@tool("execute_builder_plan_with_optuna")
def execute_builder_plan_with_optuna(
    builder_output_json: Any,
    dataset_path: str,
    n_trials_per_family: int = 50,
    test_size: float = 0.2,
) -> str:
    """
    Execute the NAS training pipeline from the Builder JSON plan using Optuna.

    Args:
        builder_output_json: Builder output as strict JSON string or dict.
        dataset_path: ECG CSV path containing features and target column.
        n_trials_per_family: Number of Optuna trials per model family.
        test_size: Test split fraction.

    Returns:
        Strict JSON string with optimization results.
    """
    if isinstance(builder_output_json, str):
        try:
            builder_payload = json.loads(builder_output_json)
        except json.JSONDecodeError as exc:
            raise ValueError(f"builder_output_json is not valid JSON: {exc}") from exc
    elif isinstance(builder_output_json, dict):
        builder_payload = builder_output_json
    else:
        raise ValueError("builder_output_json must be a JSON string or a dictionary.")

    if n_trials_per_family < 1:
        raise ValueError("n_trials_per_family must be >= 1.")

    dataset_file = Path(dataset_path)
    if not dataset_file.is_absolute():
        project_root = Path(__file__).resolve().parents[3]
        dataset_file = project_root / dataset_file

    trainer = TrainerAgent()
    results = trainer.run_from_builder_and_csv(
        builder_json=builder_payload,
        dataset_path=str(dataset_file),
        n_trials_per_family=n_trials_per_family,
        test_size=test_size,
    )

    return json.dumps(results, indent=2, ensure_ascii=False)
