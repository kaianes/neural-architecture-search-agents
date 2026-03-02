from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional


@dataclass
class SearchSpaceSpec:
    name: str

    def sample(self, trial: Any, guidance: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        raise NotImplementedError

    def build_model(self, params: Dict[str, Any], dataset_meta: Dict[str, Any]) -> Any:
        raise NotImplementedError
