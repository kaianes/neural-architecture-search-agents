from __future__ import annotations

from contextlib import contextmanager
from typing import Dict, Iterator

try:
    from codecarbon import EmissionsTracker
except Exception:
    EmissionsTracker = None


@contextmanager
def carbon_tracker(run_name: str = "run") -> Iterator[Dict[str, float | None]]:
    state: Dict[str, float | None] = {"emissions_kg": None}
    tracker = EmissionsTracker(project_name=run_name) if EmissionsTracker else None
    if tracker:
        tracker.start()
    try:
        yield state
    finally:
        if tracker:
            emissions = tracker.stop()
            state["emissions_kg"] = float(emissions) if emissions is not None else None
