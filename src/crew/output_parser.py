from __future__ import annotations

import json
import re
from typing import Any, Dict, List


def _find_first_json_object(text: str) -> Dict[str, Any]:
    if not text:
        raise ValueError("Planner returned empty output.")

    stripped = text.strip()
    try:
        return json.loads(stripped)
    except Exception:
        pass

    match = re.search(r"\{[\s\S]*\}", text)
    if not match:
        raise ValueError("Planner output does not contain a JSON object.")

    return json.loads(match.group(0))


def parse_block_plan(text: str, total_trials: int, block_size: int) -> List[Dict[str, int]]:
    payload = _find_first_json_object(text)
    blocks = payload.get("blocks")
    if not isinstance(blocks, list) or not blocks:
        raise ValueError("Planner output must include a non-empty 'blocks' list.")

    normalized: List[Dict[str, int]] = []
    for i, block in enumerate(blocks, start=1):
        if not isinstance(block, dict):
            raise ValueError("Each block must be an object.")
        bid = int(block.get("block_id", i))
        bt = int(block.get("block_trials", 0))
        if bid != i:
            raise ValueError("block_id must be sequential starting at 1.")
        if bt < 1:
            raise ValueError("block_trials must be >= 1.")
        if bt > block_size and i != len(blocks):
            raise ValueError("block_trials exceeds block_size for non-final block.")
        normalized.append({"block_id": bid, "block_trials": bt})

    total = sum(b["block_trials"] for b in normalized)
    if total != int(total_trials):
        raise ValueError(f"Invalid trial budget from planner: expected {total_trials}, got {total}.")

    return normalized
