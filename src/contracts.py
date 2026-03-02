from __future__ import annotations

from typing import Any, Dict, List, Optional, Protocol

from pydantic import BaseModel, Field


class SearchSpace(Protocol):
    """Search space contract used by the NAS engine."""

    name: str

    def sample(self, trial: Any, guidance: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        ...

    def build_model(self, params: Dict[str, Any], dataset_meta: Dict[str, Any]) -> Any:
        ...


class TrialRecord(BaseModel):
    """Persistent per-trial record used for observability and memory."""

    run_id: str
    trial_number: int
    dataset: str
    device: str
    search_space: str
    params: Dict[str, Any]
    value: float
    train_loss: Optional[float] = None
    train_acc: Optional[float] = None
    eval_loss: Optional[float] = None
    eval_acc: Optional[float] = None
    model_params: Optional[int] = None
    flops: Optional[int] = None
    emissions_kg: Optional[float] = None
    duration_s: Optional[float] = None
    config_hash: str
    context: Dict[str, Any] = Field(default_factory=dict)
    tags: List[str] = Field(default_factory=list)


class MemoryStore(Protocol):
    """Memory store contract for retrieval across runs."""

    def add(self, record: TrialRecord) -> None:
        ...

    def query(self, context: Dict[str, Any], k: int = 5) -> List[TrialRecord]:
        ...
