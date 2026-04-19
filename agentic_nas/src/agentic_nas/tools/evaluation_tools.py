"""Tools to execute post-training evaluation and handoff synthesis."""

from __future__ import annotations

import json
from typing import Any

from crewai.tools import tool

from agentic_nas.agents.evaluator_agent import EvaluatorAgent


@tool("evaluate_nas_results")
def evaluate_nas_results(trainer_output_json: Any) -> str:
    """
    Interpret trainer output and produce a structured evaluation handoff JSON.

    Args:
        trainer_output_json: Trainer output as strict JSON string or dict.

    Returns:
        Strict JSON string with interpreted performance and handoff notes.
    """
    payload = trainer_output_json
    if isinstance(trainer_output_json, str):
        try:
            payload = json.loads(trainer_output_json)
        except json.JSONDecodeError as exc:
            raise ValueError(f"trainer_output_json is not valid JSON: {exc}") from exc

    evaluator = EvaluatorAgent()
    results = evaluator.evaluate(payload)
    return json.dumps(results, indent=2, ensure_ascii=False)

