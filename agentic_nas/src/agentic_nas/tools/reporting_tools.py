"""Tools for generating final markdown reports from evaluator outputs."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from crewai.tools import tool

from agentic_nas.agents.report_agent import ReportAgent


@tool("generate_nas_markdown_report")
def generate_nas_markdown_report(
    evaluator_output_json: Any,
    output_path: str = "",
) -> str:
    """
    Generate a markdown report from evaluator output and optionally persist to file.

    Args:
        evaluator_output_json: Evaluator output as JSON string or dict.
        output_path: Optional destination for .md report.

    Returns:
        Markdown report string.
    """
    payload = evaluator_output_json
    if isinstance(evaluator_output_json, str):
        try:
            payload = json.loads(evaluator_output_json)
        except json.JSONDecodeError as exc:
            raise ValueError(f"evaluator_output_json is not valid JSON: {exc}") from exc

    report_agent = ReportAgent()
    report_markdown = report_agent.generate_report(payload)

    if output_path:
        destination = Path(output_path)
        if not destination.is_absolute():
            project_root = Path(__file__).resolve().parents[3]
            destination = project_root / destination
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(report_markdown, encoding="utf-8")

    return report_markdown
