from __future__ import annotations

import json
from copy import deepcopy
from pathlib import Path
from typing import Any, Dict

from crewai.tools import tool

from agents.evaluation_agent import EvaluationAgent
from nas.optuna_search import run_optuna_search
from crew.runtime import get_runtime


@tool("Run Optuna Block")
def run_optuna_block(plan_json: str) -> str:
    """Run one NAS block from a JSON payload and persist block-level summary."""
    payload = json.loads(plan_json)
    return execute_optuna_block(payload)


def execute_optuna_block(payload: Dict[str, Any]) -> str:
    """Execute a block payload without requiring CrewAI tool-calling."""
    rt = get_runtime()

    block_trials = int(payload.get("block_trials", 0))
    block_id = int(payload.get("block_id", 0))
    if block_trials < 1 or block_id < 1:
        raise ValueError("Invalid block payload. Expected positive block_id and block_trials.")

    cfg_block = deepcopy(rt.cfg)
    cfg_block.setdefault("optuna", {})["n_trials"] = block_trials

    block_dir = rt.results_dir / "crewai_blocks"
    block_dir.mkdir(parents=True, exist_ok=True)
    cfg_block["tracking"] = cfg_block.get("tracking") or {}
    cfg_block["tracking"]["run_dir"] = str(block_dir / f"block_{block_id:03d}")

    summary = run_optuna_search(cfg_block, rt.device)

    out_path = block_dir / f"block_{block_id:03d}_summary.json"
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    return f"Block {block_id} done. Summary saved to: {out_path.as_posix()}"


@tool("Summarize Results")
def summarize_results(paths_json: str) -> str:
    """Count generated CrewAI block summaries in the results directory."""
    payload = json.loads(paths_json)
    return summarize_results_payload(payload)


def summarize_results_payload(payload: Dict[str, Any]) -> str:
    """Summarize result files without requiring CrewAI tool-calling."""
    rt = get_runtime()
    results_dir = Path(payload.get("results_dir") or rt.results_dir)
    blocks_dir = results_dir / "crewai_blocks"

    if not blocks_dir.exists():
        return f"No blocks found in {blocks_dir.as_posix()}."

    block_files = sorted(blocks_dir.glob("block_*_summary.json"))
    return f"Found {len(block_files)} block summaries in {blocks_dir.as_posix()}."


def load_aggregated_summary(cfg: Dict[str, Any], results_dir: Path, final_output: str, mode: str) -> Dict[str, Any]:
    block_files = sorted((results_dir / "crewai_blocks").glob("block_*_summary.json"))
    block_summaries = []
    for fp in block_files:
        try:
            with fp.open("r", encoding="utf-8") as f:
                block_summaries.append(json.load(f))
        except Exception:
            continue

    all_trials = []
    best_value = None
    best_params = None
    direction = str((cfg.get("optuna") or {}).get("direction", "maximize")).lower()

    for s in block_summaries:
        all_trials.extend(s.get("trials") or [])
        val = s.get("best_value")
        if val is None:
            continue
        if best_value is None:
            best_value = val
            best_params = s.get("best_params")
            continue
        if direction == "minimize" and val < best_value:
            best_value = val
            best_params = s.get("best_params")
        if direction != "minimize" and val > best_value:
            best_value = val
            best_params = s.get("best_params")

    summary = EvaluationAgent().summarize(
        {
            "best_value": best_value,
            "best_params": best_params,
            "trials": all_trials,
            "attrs": {},
        }
    )
    summary["run_id"] = (cfg.get("tracking") or {}).get("run_id")
    summary["run_dir"] = (cfg.get("tracking") or {}).get("run_dir")
    summary["attrs"] = {
        **(summary.get("attrs") or {}),
        "orchestrator": "crewai",
        "execution_mode": mode,
        "results_dir": results_dir.as_posix(),
        "executed_blocks": len(block_summaries),
        "final_output": final_output,
    }
    return summary
