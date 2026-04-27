import json
from datetime import datetime
from pathlib import Path
from typing import Any


def resolve_task_alias(task_name: str | None, agent_name: str, index: int) -> str:
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


def json_default(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, tuple):
        return list(value)
    enum_value = getattr(value, "value", None)
    if enum_value is not None:
        return enum_value
    return str(value)


def try_parse_json(raw_output: str) -> Any | None:
    if not raw_output:
        return None
    try:
        return json.loads(raw_output)
    except (TypeError, json.JSONDecodeError):
        return None


def persist_agent_outputs(
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
        alias = resolve_task_alias(task_name=task_name, agent_name=agent_name, index=index)
        if alias in saved_files:
            alias = f"{alias}_{index:02d}"

        raw_output = getattr(task_output, "raw", "")
        json_output = getattr(task_output, "json_dict", None)
        if json_output is None:
            json_output = try_parse_json(raw_output)

        payload = {
            "task_alias": alias,
            "task_name": task_name,
            "agent": agent_name,
            "description": getattr(task_output, "description", ""),
            "expected_output": getattr(task_output, "expected_output", None),
            "output_format": json_default(getattr(task_output, "output_format", "raw")),
            "output_json": json_output,
            "output_raw": raw_output,
        }

        task_file = agents_dir / f"{alias}.json"
        task_file.write_text(
            json.dumps(payload, indent=2, ensure_ascii=False, default=json_default),
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
        json.dumps(crew_payload, indent=2, ensure_ascii=False, default=json_default),
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
        json.dumps(manifest, indent=2, ensure_ascii=False, default=json_default),
        encoding="utf-8",
    )
