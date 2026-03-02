from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional


@dataclass
class CrewRuntime:
    cfg: Dict[str, Any]
    device: str
    results_dir: Path
    logger: Optional[Any] = None


_RUNTIME: Optional[CrewRuntime] = None


def set_runtime(cfg: Dict[str, Any], device: str, results_dir: Path, logger: Optional[Any] = None) -> None:
    global _RUNTIME
    _RUNTIME = CrewRuntime(cfg=cfg, device=str(device), results_dir=results_dir, logger=logger)


def get_runtime() -> CrewRuntime:
    if _RUNTIME is None:
        raise RuntimeError("Crew runtime is not configured. Call set_runtime first.")
    return _RUNTIME
