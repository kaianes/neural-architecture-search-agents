from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

from contracts import TrialRecord

try:
    import faiss
except Exception:
    faiss = None


_DATASET_TO_ID = {"MNIST": 1.0, "FASHIONMNIST": 2.0, "CIFAR10": 3.0}


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except Exception:
        return default


def _ctx_to_vector(context: Dict[str, Any]) -> np.ndarray:
    dataset = str(context.get("dataset", "MNIST")).upper()
    params = context.get("params") or {}
    vector = np.array(
        [
            _DATASET_TO_ID.get(dataset, 0.0),
            _safe_float(context.get("in_ch"), 0.0),
            _safe_float(context.get("size"), 0.0),
            _safe_float(context.get("batch_size"), 0.0),
            _safe_float(context.get("epochs"), 0.0),
            _safe_float(params.get("conv_channels"), 0.0),
            _safe_float(params.get("kernel_size"), 0.0),
            _safe_float(params.get("dropout"), 0.0),
            _safe_float(np.log10(max(_safe_float(params.get("lr"), 1e-3), 1e-8))),
            _safe_float(context.get("value"), 0.0),
            _safe_float(context.get("model_params"), 0.0) / 1_000_000.0,
            _safe_float(context.get("flops"), 0.0) / 1_000_000_000.0,
            _safe_float(context.get("emissions_kg"), 0.0),
        ],
        dtype=np.float32,
    )
    return vector


class FaissMemoryStore:
    def __init__(self, index_path: str, records_path: str) -> None:
        self.index_path = Path(index_path)
        self.records_path = Path(records_path)
        self.index_path.parent.mkdir(parents=True, exist_ok=True)
        self.records_path.parent.mkdir(parents=True, exist_ok=True)

        self.records: List[TrialRecord] = []
        self.matrix = np.empty((0, 13), dtype=np.float32)
        self.index = None

        self._load_records()
        self._rebuild_index()

    def _load_records(self) -> None:
        if not self.records_path.exists():
            return
        with self.records_path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    self.records.append(TrialRecord.model_validate_json(line))
                except Exception:
                    continue

    def _rebuild_index(self) -> None:
        if not self.records:
            self.matrix = np.empty((0, 13), dtype=np.float32)
            self.index = None
            return

        self.matrix = np.vstack([_ctx_to_vector(self._record_context(r)) for r in self.records]).astype(np.float32)

        if faiss:
            self.index = faiss.IndexFlatL2(self.matrix.shape[1])
            self.index.add(self.matrix)
            faiss.write_index(self.index, str(self.index_path))
        else:
            self.index = None

    def _record_context(self, record: TrialRecord) -> Dict[str, Any]:
        ctx = dict(record.context)
        ctx.update(
            {
                "dataset": record.dataset,
                "params": record.params,
                "value": record.value,
                "model_params": record.model_params,
                "flops": record.flops,
                "emissions_kg": record.emissions_kg,
            }
        )
        return ctx

    def add(self, record: TrialRecord) -> None:
        self.records.append(record)
        with self.records_path.open("a", encoding="utf-8") as f:
            f.write(record.model_dump_json() + "\n")

        vec = _ctx_to_vector(self._record_context(record)).reshape(1, -1)
        if self.matrix.size == 0:
            self.matrix = vec.astype(np.float32)
        else:
            self.matrix = np.vstack([self.matrix, vec]).astype(np.float32)

        if faiss:
            if self.index is None:
                self.index = faiss.IndexFlatL2(self.matrix.shape[1])
            self.index.add(vec.astype(np.float32))
            faiss.write_index(self.index, str(self.index_path))

    def query(self, context: Dict[str, Any], k: int = 5) -> List[TrialRecord]:
        if not self.records:
            return []

        k = max(1, min(k, len(self.records)))
        q = _ctx_to_vector(context).reshape(1, -1).astype(np.float32)

        if faiss and self.index is not None:
            _, idx = self.index.search(q, k)
            return [self.records[int(i)] for i in idx[0] if int(i) >= 0]

        # Numpy fallback if faiss is unavailable.
        distances = np.linalg.norm(self.matrix - q, axis=1)
        order = np.argsort(distances)[:k]
        return [self.records[int(i)] for i in order.tolist()]
