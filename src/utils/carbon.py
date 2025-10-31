from contextlib import contextmanager

try:
    from codecarbon import EmissionsTracker
except Exception:
    EmissionsTracker = None


@contextmanager
def carbon_tracker(run_name: str = "run"):
    tracker = EmissionsTracker(project_name=run_name) if EmissionsTracker else None
    if tracker:
        tracker.start()
    try:
        yield
    finally:
        if tracker:
            tracker.stop()