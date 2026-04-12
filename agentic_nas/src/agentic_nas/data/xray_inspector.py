from __future__ import annotations

from typing import Any, Dict, List, Optional

from datasets import DatasetDict

from agentic_nas.data.base_schema import build_observation_schema


def _pick_split(dataset: DatasetDict, requested_split: str) -> tuple[str, Any]:
    available_splits = list(dataset.keys())
    if not available_splits:
        raise ValueError("Dataset has no available splits.")

    if requested_split in dataset:
        return requested_split, dataset[requested_split]

    return available_splits[0], dataset[available_splits[0]]


def _extract_image_metadata(sample: Dict[str, Any]) -> Dict[str, Any]:
    image_field_name = None
    image_width = None
    image_height = None
    image_mode = None
    image_channels = None

    for key, value in sample.items():
        if hasattr(value, "size") and hasattr(value, "mode"):
            image_field_name = key
            width, height = value.size
            image_width = int(width)
            image_height = int(height)
            image_mode = str(value.mode)

            if hasattr(value, "getbands"):
                image_channels = len(value.getbands())
            break

    return {
        "image_field_name": image_field_name,
        "image_width": image_width,
        "image_height": image_height,
        "image_mode": image_mode,
        "image_channels": image_channels,
    }


def _guess_target_field(sample_keys: List[str], feature_types: Dict[str, str]) -> Optional[str]:
    preferred_names = ["label", "labels", "target", "class", "diagnosis"]

    for candidate in preferred_names:
        if candidate in sample_keys:
            return candidate

    for key, feature_type in feature_types.items():
        if feature_type == "ClassLabel":
            return key

    return None


def _to_label_text(value: Any, label_names: Optional[List[str]]) -> str:
    if label_names and isinstance(value, int) and 0 <= value < len(label_names):
        return str(label_names[value])
    return str(value)


def _collect_label_examples(
    split_data: Any,
    target_field: Optional[str],
    label_names: Optional[List[str]],
    max_rows: int = 50,
    max_examples: int = 10,
) -> List[str]:
    if not target_field:
        return []

    examples: List[str] = []
    observed = set()

    upper_bound = min(len(split_data), max_rows)
    for idx in range(upper_bound):
        value = split_data[idx].get(target_field)
        value_as_text = _to_label_text(value, label_names)
        if value_as_text not in observed:
            observed.add(value_as_text)
            examples.append(value_as_text)
        if len(examples) >= max_examples:
            break

    return examples


def inspect_xray_dataset_dict(
    dataset: DatasetDict,
    dataset_id: str,
    dataset_name: str,
    requested_split: str = "train",
    split_metadata: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    selected_split, split_data = _pick_split(dataset, requested_split)

    sample = split_data[0] if len(split_data) > 0 else {}
    sample_keys = list(sample.keys())

    feature_types: Dict[str, str] = {
        key: feature.__class__.__name__ for key, feature in split_data.features.items()
    }

    image_metadata = _extract_image_metadata(sample)

    target_field = None
    if split_metadata and split_metadata.get("target_name"):
        target_field = str(split_metadata["target_name"])
    else:
        target_field = _guess_target_field(sample_keys, feature_types)

    label_names = None
    if target_field and target_field in split_data.features:
        target_feature = split_data.features[target_field]
        if hasattr(target_feature, "names"):
            label_names = list(target_feature.names)

    if split_metadata and split_metadata.get("target_values"):
        label_examples = [str(v) for v in split_metadata.get("target_values", [])]
    else:
        label_examples = _collect_label_examples(split_data, target_field, label_names)

    if split_metadata and "num_samples_in_selected_split" in split_metadata:
        num_samples = int(split_metadata["num_samples_in_selected_split"])
    else:
        num_samples = int(len(split_data))

    available_splits = (
        split_metadata.get("available_splits")
        if split_metadata and split_metadata.get("available_splits")
        else list(dataset.keys())
    )

    raw_observations = {
        "available_splits": available_splits,
        "selected_split": selected_split,
        "num_samples_in_selected_split": num_samples,
        "sample_keys": sample_keys,
        "feature_types": feature_types,
    }

    target_info = {
        "target_name": target_field,
        "target_values": label_examples,
        "target_distribution": None,
    }

    quality_observations = {
        "selected_split_is_empty": num_samples == 0,
        "missing_target_field": target_field is None,
        "image_metadata_missing": image_metadata["image_field_name"] is None,
    }

    modality_specific_observations = {
        "storage_format": "huggingface_dataset",
        "image_field_name": image_metadata["image_field_name"],
        "image_width": image_metadata["image_width"],
        "image_height": image_metadata["image_height"],
        "image_mode": image_metadata["image_mode"],
        "image_channels": image_metadata["image_channels"],
    }

    return build_observation_schema(
        dataset_id=dataset_id,
        dataset_name=dataset_name,
        source_type="huggingface",
        source_uri=dataset_id,
        local_path="",
        raw_observations=raw_observations,
        target_info=target_info,
        quality_observations=quality_observations,
        modality_specific_observations=modality_specific_observations,
        notes=[
            "This schema contains objective observations only.",
            "Task/modality conclusions must be inferred by the agent layer.",
        ],
    )
