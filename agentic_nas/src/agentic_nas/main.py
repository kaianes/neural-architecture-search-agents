import argparse
import json
import os
from datetime import datetime
from pathlib import Path
from typing import Any

from dotenv import load_dotenv

from agentic_nas.data.hf_image_loader import DEFAULT_DENTEX_DATASET_ID
from agentic_nas.crew import build_crew


def _env_to_bool(value: str | None, default: bool = False) -> bool:
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "y", "on"}


def _resolve_task_alias(task_name: str | None, agent_name: str, index: int) -> str:
    normalized = " ".join(part for part in (task_name, agent_name) if part).lower()

    if "profil" in normalized:
        return "profiling_agent"
    if "build" in normalized:
        return "builder_agent"
    if "train" in normalized:
        return "trainer_agent"
    if "evaluat" in normalized:
        return "evaluator_agent"
    if "report" in normalized:
        return "report_agent"
    return f"task_{index:02d}"


def _json_default(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, tuple):
        return list(value)
    enum_value = getattr(value, "value", None)
    if enum_value is not None:
        return enum_value
    return str(value)


def _try_parse_json(raw_output: str) -> Any | None:
    if not raw_output:
        return None
    try:
        return json.loads(raw_output)
    except (TypeError, json.JSONDecodeError):
        return None


def _persist_agent_outputs(
    project_root: Path,
    run_dir: Path,
    mode: str,
    include_trainer: bool,
    include_evaluator: bool,
    include_reporter: bool,
    crew_result: Any,
) -> None:
    agents_dir = run_dir / "agents"
    agents_dir.mkdir(parents=True, exist_ok=True)

    tasks_output = list(getattr(crew_result, "tasks_output", []) or [])
    saved_files: dict[str, str] = {}

    for index, task_output in enumerate(tasks_output, start=1):
        task_name = getattr(task_output, "name", None)
        agent_name = getattr(task_output, "agent", "")
        alias = _resolve_task_alias(task_name=task_name, agent_name=agent_name, index=index)
        if alias in saved_files:
            alias = f"{alias}_{index:02d}"

        raw_output = getattr(task_output, "raw", "")
        json_output = getattr(task_output, "json_dict", None)
        if json_output is None:
            json_output = _try_parse_json(raw_output)

        payload = {
            "task_alias": alias,
            "task_name": task_name,
            "agent": agent_name,
            "description": getattr(task_output, "description", ""),
            "expected_output": getattr(task_output, "expected_output", None),
            "output_format": _json_default(getattr(task_output, "output_format", "raw")),
            "output_json": json_output,
            "output_raw": raw_output,
        }

        task_file = agents_dir / f"{alias}.json"
        task_file.write_text(
            json.dumps(payload, indent=2, ensure_ascii=False, default=_json_default),
            encoding="utf-8",
        )
        saved_files[alias] = str(task_file.relative_to(project_root))

    token_usage = getattr(crew_result, "token_usage", None)
    if token_usage is not None and hasattr(token_usage, "model_dump"):
        token_usage = token_usage.model_dump(mode="json")

    crew_payload = {
        "raw": getattr(crew_result, "raw", str(crew_result)),
        "json_dict": getattr(crew_result, "json_dict", None),
        "token_usage": token_usage,
    }
    crew_result_file = run_dir / "crew_result.json"
    crew_result_file.write_text(
        json.dumps(crew_payload, indent=2, ensure_ascii=False, default=_json_default),
        encoding="utf-8",
    )

    manifest = {
        "run_id": run_dir.name,
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "mode": mode,
        "include_trainer": include_trainer,
        "include_evaluator": include_evaluator,
        "include_reporter": include_reporter,
        "agent_outputs": saved_files,
        "crew_result_file": str(crew_result_file.relative_to(project_root)),
    }
    manifest_file = run_dir / "manifest.json"
    manifest_file.write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False, default=_json_default),
        encoding="utf-8",
    )


def run() -> None:
    project_root = Path(__file__).resolve().parents[2]
    load_dotenv(project_root / ".env")

    default_ecg_path = project_root / "data" / "raw" / "sleep_apnea_ecg.csv"

    parser = argparse.ArgumentParser(description="Run Agentic NAS dataset profiling.")
    parser.add_argument("--mode", choices=["ecg", "xray"], default=os.getenv("PROFILE_MODE", "ecg"))
    parser.add_argument("--dataset-path", default=os.getenv("ECG_DATASET_PATH", str(default_ecg_path)))
    parser.add_argument("--hf-dataset-id", default=os.getenv("HF_DATASET_ID", DEFAULT_DENTEX_DATASET_ID))
    parser.add_argument("--hf-config-name", default=os.getenv("HF_CONFIG_NAME"))
    parser.add_argument("--hf-split", default=os.getenv("HF_SPLIT", "train"))
    parser.add_argument("--output-file", default=os.getenv("PROFILE_OUTPUT_FILE"))
    parser.add_argument(
        "--trainer-trials",
        type=int,
        default=int(os.getenv("TRAINER_TRIALS", "5")),
        help="Number of Optuna trials per family for trainer agent.",
    )
    parser.add_argument(
        "--include-trainer",
        action=argparse.BooleanOptionalAction,
        default=_env_to_bool(os.getenv("INCLUDE_TRAINER"), default=True),
        help="Include trainer agent task in the crew run.",
    )
    parser.add_argument(
        "--include-evaluator",
        action=argparse.BooleanOptionalAction,
        default=_env_to_bool(os.getenv("INCLUDE_EVALUATOR"), default=True),
        help="Include evaluator agent task after trainer in the crew run.",
    )
    parser.add_argument(
        "--include-reporter",
        action=argparse.BooleanOptionalAction,
        default=_env_to_bool(os.getenv("INCLUDE_REPORTER"), default=True),
        help="Include report agent task after evaluator in the crew run.",
    )
    parser.add_argument(
        "--report-output-file",
        default=os.getenv("REPORT_OUTPUT_FILE"),
        help="Path to save the markdown report generated by report agent.",
    )
    args = parser.parse_args()

    output_root = project_root / "data" / "outputs"
    run_id = f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_{args.mode}"
    run_dir = output_root / "runs" / run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    report_output_file = args.report_output_file
    if args.include_reporter and not report_output_file:
        report_output_file = str(run_dir / "report.md")

    crew = build_crew(
        profile_mode=args.mode,
        dataset_path=args.dataset_path,
        hf_dataset_id=args.hf_dataset_id,
        hf_split=args.hf_split,
        hf_config_name=args.hf_config_name,
        include_trainer=args.include_trainer,
        include_evaluator=args.include_evaluator,
        include_reporter=args.include_reporter,
        trainer_trials=args.trainer_trials,
        report_output_path=report_output_file,
    )
    result = crew.kickoff()

    _persist_agent_outputs(
        project_root=project_root,
        run_dir=run_dir,
        mode=args.mode,
        include_trainer=args.include_trainer,
        include_evaluator=args.include_evaluator,
        include_reporter=args.include_reporter,
        crew_result=result,
    )

    latest_run_file = output_root / "latest_run.txt"
    latest_run_file.parent.mkdir(parents=True, exist_ok=True)
    latest_run_file.write_text(str(run_dir.relative_to(project_root)), encoding="utf-8")

    if args.output_file:
        output_file = Path(args.output_file)
        if not output_file.is_absolute():
            output_file = project_root / output_file

        output_file.parent.mkdir(parents=True, exist_ok=True)
        output_file.write_text(str(result), encoding="utf-8")


if __name__ == "__main__":
    run()
