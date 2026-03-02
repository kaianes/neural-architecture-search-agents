from __future__ import annotations

import hashlib
import json
import csv
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional


def compute_config_hash(cfg: Dict[str, Any]) -> str:
    payload = json.dumps(cfg, sort_keys=True, default=str).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()

# Creates the run_id and run_dir based on the current timestamp if not provided
def init_run_context(cfg: Dict[str, Any]) -> Dict[str, Any]:
    tracking = cfg.setdefault("tracking", {})
    paths = cfg.setdefault("paths", {})

    base_results = Path(paths.get("results", "experiments/results"))
    run_id = tracking.get("run_id") or datetime.now().strftime("run_%Y%m%d_%H%M%S")
    run_dir = Path(tracking.get("run_dir") or (base_results / run_id))

    run_dir.mkdir(parents=True, exist_ok=True)
    base_results.mkdir(parents=True, exist_ok=True)

    tracking["run_id"] = run_id
    tracking["run_dir"] = str(run_dir)
    tracking["config_hash"] = compute_config_hash(cfg)
    return cfg


def append_jsonl(path: str | Path, record: Dict[str, Any]) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=True, default=str) + "\n")


def write_json(path: str | Path, payload: Dict[str, Any]) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, default=str)


def append_emissions_record(
    run_dir: str | Path,
    trial_id: str,
    emissions_kg: Optional[float] = None,
    accuracy: Optional[float] = None,
    params_M: Optional[float] = None,
    flops_B: Optional[float] = None,
    timestamp: Optional[str] = None
) -> Path:
    """
    Appends an emissions record to the run-specific CSV file.
    
    Args:
        run_dir: Path to the run directory
        trial_id: Trial identifier
        emissions_kg: Carbon emissions in kg
        accuracy: Model accuracy score
        params_M: Model parameters in millions
        flops_B: Floating point operations in billions
        timestamp: ISO format timestamp
    
    Returns:
        Path to the emissions CSV file
    """
    run_dir = Path(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    
    emissions_csv = run_dir / "emissions.csv"
    timestamp = timestamp or datetime.now().isoformat()
    
    # Check if file exists to determine if we need to write header
    file_exists = emissions_csv.exists()
    
    with emissions_csv.open("a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        
        # Write header if file is new
        if not file_exists:
            writer.writerow([
                "trial_id",
                "emissions_kg",
                "accuracy",
                "params_M",
                "flops_B",
                "timestamp"
            ])
        
        # Write data row
        writer.writerow([
            trial_id,
            emissions_kg if emissions_kg is not None else "",
            accuracy if accuracy is not None else "",
            params_M if params_M is not None else "",
            flops_B if flops_B is not None else "",
            timestamp
        ])
    
    return emissions_csv


def read_emissions_csv(run_dir: str | Path) -> List[Dict[str, Any]]:
    """
    Reads all emissions records from the run-specific CSV file.
    
    Args:
        run_dir: Path to the run directory
    
    Returns:
        List of dictionaries containing emissions data
    """
    run_dir = Path(run_dir)
    emissions_csv = run_dir / "emissions.csv"
    
    if not emissions_csv.exists():
        return []
    
    records = []
    with emissions_csv.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            # Convert numeric fields
            if row.get("emissions_kg"):
                row["emissions_kg"] = float(row["emissions_kg"])
            if row.get("accuracy"):
                row["accuracy"] = float(row["accuracy"])
            if row.get("params_M"):
                row["params_M"] = float(row["params_M"])
            if row.get("flops_B"):
                row["flops_B"] = float(row["flops_B"])
            records.append(row)
    
    return records


def summarize_emissions(run_dir: str | Path) -> Dict[str, Any]:
    """
    Calculates summary statistics for emissions in a run.
    
    Args:
        run_dir: Path to the run directory
    
    Returns:
        Dictionary with summary statistics
    """
    records = read_emissions_csv(run_dir)
    
    if not records:
        return {
            "total_trials": 0,
            "total_emissions_kg": 0.0,
            "avg_emissions_per_trial_kg": 0.0,
            "max_emissions_kg": 0.0,
            "min_emissions_kg": 0.0
        }
    
    emissions = [r.get("emissions_kg", 0.0) for r in records if r.get("emissions_kg")]
    
    return {
        "total_trials": len(records),
        "total_emissions_kg": sum(emissions) if emissions else 0.0,
        "avg_emissions_per_trial_kg": sum(emissions) / len(emissions) if emissions else 0.0,
        "max_emissions_kg": max(emissions) if emissions else 0.0,
        "min_emissions_kg": min(emissions) if emissions else 0.0,
        "records_count": len(records)
    }


def persist_summary(cfg: Dict[str, Any], payload: Dict[str, Any], name: str) -> Dict[str, str]:
    tracking = cfg.get("tracking") or {}
    paths = cfg.get("paths") or {}
    run_dir = Path(tracking.get("run_dir") or paths.get("results", "experiments/results"))
    legacy_dir = Path(paths.get("results", "experiments/results"))

    run_path = run_dir / name
    write_json(run_path, payload)

    out = {"run_path": run_path.as_posix()}
    if tracking.get("mirror_legacy_results", True):
        legacy_path = legacy_dir / name
        if legacy_path.resolve() != run_path.resolve():
            write_json(legacy_path, payload)
        out["legacy_path"] = legacy_path.as_posix()
    return out
