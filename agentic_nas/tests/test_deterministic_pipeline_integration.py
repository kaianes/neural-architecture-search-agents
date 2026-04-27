import json
from dataclasses import dataclass
from typing import Any

import pytest

pytest.importorskip("torch")

from agentic_nas.agents import trainer_agent as trainer_agent_module
from agentic_nas.agents.evaluator_agent import EvaluatorAgent
from agentic_nas.agents.report_agent import ReportAgent
from agentic_nas.agents.trainer_agent import TrainerAgent
from agentic_nas.artifacts import persist_agent_outputs


@dataclass
class FakeTaskOutput:
    name: str
    agent: str
    raw: str
    json_dict: Any = None
    description: str = "Fixture task"
    expected_output: str = "Fixture output"
    output_format: str = "raw"


@dataclass
class FakeCrewResult:
    tasks_output: list[FakeTaskOutput]
    raw: str = "fixture crew result"
    json_dict: Any = None
    token_usage: Any = None


def test_deterministic_builder_trainer_evaluator_reporter_pipeline(
    monkeypatch,
    tmp_path,
    builder_output,
    tiny_training_csv,
):
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

    trainer_output = TrainerAgent().run_from_builder_and_csv(
        builder_json=builder_output,
        dataset_path=str(tiny_training_csv),
        n_trials_per_family=1,
        test_size=0.2,
    )
    evaluator_output = EvaluatorAgent().evaluate(trainer_output)
    report = ReportAgent().generate_report(evaluator_output)

    run_dir = tmp_path / "data" / "outputs" / "runs" / "integration_ecg"
    run_dir.mkdir(parents=True)
    (run_dir / "report.md").write_text(report, encoding="utf-8")

    crew_result = FakeCrewResult(
        tasks_output=[
            FakeTaskOutput(
                name="builder_agent",
                agent="NAS Search Space Builder Specialist",
                raw=json.dumps(builder_output),
                json_dict=builder_output,
            ),
            FakeTaskOutput(
                name="trainer_agent",
                agent="Neural Architecture Search Trainer using Optuna",
                raw=json.dumps(trainer_output),
                json_dict=trainer_output,
            ),
            FakeTaskOutput(
                name="evaluator_agent",
                agent="NAS Performance Evaluator and Decision Analyst",
                raw=json.dumps(evaluator_output),
                json_dict=evaluator_output,
            ),
            FakeTaskOutput(
                name="report_agent",
                agent="NAS Reporting Specialist",
                raw=report,
            ),
        ]
    )
    persist_agent_outputs(
        project_root=tmp_path,
        run_dir=run_dir,
        mode="ecg",
        include_trainer=True,
        include_evaluator=True,
        include_reporter=True,
        crew_result=crew_result,
    )

    assert (run_dir / "agents" / "builder_agent.json").exists()
    assert (run_dir / "agents" / "trainer_agent.json").exists()
    assert (run_dir / "agents" / "evaluator_agent.json").exists()
    assert (run_dir / "agents" / "report_agent.json").exists()
    assert (run_dir / "manifest.json").exists()
    assert (run_dir / "crew_result.json").exists()
    assert (run_dir / "report.md").exists()
    assert "## Data modality\n1D Time-Series/Signal (ECG)" in report
    assert evaluator_output["selected_model"]["family"] == "mlp"
