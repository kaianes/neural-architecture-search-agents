from __future__ import annotations

from typing import Any, Dict

from crewai import LLM


def make_llm(cfg: Dict[str, Any]) -> LLM:
    return LLM(
        model=str(cfg.get("model", "ollama/gpt-oss:120b-cloud")),
        temperature=float(cfg.get("temperature", 0.0)),
    )
