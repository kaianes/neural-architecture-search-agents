from __future__ import annotations

import json

from crewai import Task


def build_plan_task(agent, total_trials: int, block_size: int, n_blocks: int) -> Task:
    description = (
        "Create a NAS execution plan in pure JSON.\n"
        f"Constraints: total_trials={total_trials}, block_size={block_size}, n_blocks={n_blocks}.\n"
        "Rules:\n"
        "1) Output ONLY valid JSON.\n"
        "2) JSON root must be an object with key 'blocks'.\n"
        "3) Each block must contain integer fields: block_id, block_trials.\n"
        "4) The sum(block_trials) must be exactly total_trials.\n"
        "5) block_id must start at 1 and be sequential.\n"
        "Example: {\"blocks\":[{\"block_id\":1,\"block_trials\":5}]}"
    )
    return Task(
        description=description,
        expected_output="JSON object with execution blocks.",
        agent=agent,
    )


def build_run_task(agent, block_id: int, block_trials: int) -> Task:
    payload = json.dumps({"block_id": int(block_id), "block_trials": int(block_trials)})
    description = (
        "Execute exactly one NAS block by calling tool 'Run Optuna Block' exactly once.\n"
        f"Use this exact payload JSON: {payload}\n"
        "Return the tool output string as the final answer."
    )
    return Task(
        description=description,
        expected_output="Execution confirmation for the block.",
        agent=agent,
    )


def build_report_task(agent, results_dir: str) -> Task:
    payload = json.dumps({"results_dir": results_dir})
    description = (
        "Call tool 'Summarize Results' once using this JSON payload:\n"
        f"{payload}\n"
        "Return a concise final report sentence."
    )
    return Task(
        description=description,
        expected_output="A concise summary referencing result artifacts.",
        agent=agent,
    )
