from typing import Any, Dict, Optional


def build_observation_schema(
    *,
    dataset_id: str,
    dataset_name: str,
    source_type: str,
    source_uri: str,
    local_path: str,
    raw_observations: Dict[str, Any],
    target_info: Dict[str, Any],
    quality_observations: Dict[str, Any],
    modality_specific_observations: Optional[Dict[str, Any]] = None,
    notes: Optional[list[str]] = None,
) -> Dict[str, Any]:
    return {
        "dataset_id": dataset_id,
        "dataset_name": dataset_name,
        "source": {
            "type": source_type,
            "uri": source_uri,
            "local_path": local_path,
        },
        "raw_observations": raw_observations,
        "target_info": target_info,
        "quality_observations": quality_observations,
        "modality_specific_observations": modality_specific_observations or {},
        "notes": notes or [],
    }