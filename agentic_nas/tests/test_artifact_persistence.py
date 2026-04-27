import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from agentic_nas.artifacts import persist_agent_outputs


@dataclass
class FakeTaskOutput:
    name: str
    agent: str
    raw: str
    json_dict: Any = None
    description: str = "Task description"
    expected_output: str = "Expected output"
    output_format: str = "raw"


@dataclass
class FakeCrewResult:
    tasks_output: list[FakeTaskOutput]
    raw: str = "crew raw"
    json_dict: Any = None
    token_usage: Any = None


def test_persisted_agent_wrappers_and_manifest_have_required_keys(tmp_path):
    run_dir = tmp_path / "data" / "outputs" / "runs" / "test_run_ecg"
    crew_result = FakeCrewResult(
        tasks_output=[
            FakeTaskOutput(
                name="builder_agent",
                agent="NAS Search Space Builder Specialist",
                raw='{"search_space": {"mlp": {"hyperparameters": {}}}}',
            ),
            FakeTaskOutput(
                name="report_agent",
                agent="NAS Reporting Specialist",
                raw="# NAS Optimization Report",
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

    wrapper_keys = {
        "task_alias",
        "task_name",
        "agent",
        "description",
        "expected_output",
        "output_format",
        "output_json",
        "output_raw",
    }
    builder_wrapper = json.loads((run_dir / "agents" / "builder_agent.json").read_text())
    report_wrapper = json.loads((run_dir / "agents" / "report_agent.json").read_text())

    assert wrapper_keys.issubset(builder_wrapper)
    assert wrapper_keys.issubset(report_wrapper)
    assert builder_wrapper["output_json"]["search_space"]
    assert report_wrapper["output_raw"] == "# NAS Optimization Report"

    manifest = json.loads((run_dir / "manifest.json").read_text())
    manifest_keys = {
        "run_id",
        "mode",
        "include_trainer",
        "include_evaluator",
        "include_reporter",
        "agent_outputs",
        "crew_result_file",
    }

    assert manifest_keys.issubset(manifest)
    assert manifest["run_id"] == "test_run_ecg"
    assert manifest["mode"] == "ecg"
    assert manifest["include_trainer"] is True
    assert manifest["include_evaluator"] is True
    assert manifest["include_reporter"] is True
    assert Path(tmp_path / manifest["crew_result_file"]).exists()
