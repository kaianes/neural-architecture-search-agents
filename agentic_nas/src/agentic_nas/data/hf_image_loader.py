from __future__ import annotations

from io import BytesIO
from typing import Any, Dict, List, Optional, Tuple

import fsspec
from datasets import Dataset, DatasetDict, load_dataset
from huggingface_hub import list_repo_files
from PIL import Image


DEFAULT_DENTEX_DATASET_ID = "ibrahimhamamci/DENTEX"
IMAGE_EXTENSIONS = (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff")


def _normalize_split_name(split: str) -> str:
    split_lower = split.strip().lower()
    if split_lower in {"train", "training"}:
        return "train"
    if split_lower in {"validation", "valid", "val"}:
        return "validation"
    if split_lower in {"test", "testing"}:
        return "test"
    return split_lower


def _load_single_split_standard(
    dataset_id: str,
    split: str,
    config_name: Optional[str],
) -> Dataset:
    load_kwargs = {"split": split}
    if config_name:
        load_kwargs["name"] = config_name

    dataset = load_dataset(path=dataset_id, **load_kwargs)
    if not isinstance(dataset, Dataset):
        raise TypeError(
            "Expected a Dataset when loading a single split. "
            f"Got {type(dataset).__name__} for '{dataset_id}:{split}'."
        )

    return dataset


def _split_from_archive_path(archive_path: str) -> Optional[str]:
    name = archive_path.lower()
    if "train" in name:
        return "train"
    if "valid" in name or "val" in name:
        return "validation"
    if "test" in name:
        return "test"
    return None


def _build_split_archive_url(dataset_id: str, archive_path: str) -> str:
    return f"https://huggingface.co/datasets/{dataset_id}/resolve/main/{archive_path}"


def _infer_label_from_path(path: str, selected_split: str) -> Optional[str]:
    parts = path.replace("\\", "/").split("/")
    if len(parts) < 2:
        return None

    candidate = parts[-2].strip().lower()
    non_label_tokens = {
        "images",
        "image",
        "xrays",
        "xray",
        "data",
        "train",
        "test",
        "validation",
        "valid",
        selected_split,
        f"{selected_split}_data",
        "training_data",
        "test_data",
        "validation_data",
    }

    if not candidate or candidate in non_label_tokens:
        return None

    return candidate


def _extract_label_examples(image_paths: List[str], selected_split: str, limit: int = 10) -> List[str]:
    labels: List[str] = []
    seen = set()

    for path in image_paths:
        label = _infer_label_from_path(path, selected_split)
        if label and label not in seen:
            seen.add(label)
            labels.append(label)
        if len(labels) >= limit:
            break

    return labels


def _load_single_split_from_archives(
    dataset_id: str,
    split: str,
) -> Tuple[Dataset, Dict[str, Any]]:
    files = list_repo_files(dataset_id, repo_type="dataset")
    archive_paths = [file for file in files if file.lower().endswith(".zip")]

    split_to_archive: Dict[str, str] = {}
    for archive_path in archive_paths:
        inferred_split = _split_from_archive_path(archive_path)
        if inferred_split and inferred_split not in split_to_archive:
            split_to_archive[inferred_split] = archive_path

    available_splits = sorted(split_to_archive.keys())
    if not available_splits:
        raise ValueError(f"No split archives (.zip) found for dataset '{dataset_id}'.")

    selected_split = split if split in split_to_archive else available_splits[0]
    archive_path = split_to_archive[selected_split]
    archive_url = _build_split_archive_url(dataset_id, archive_path)

    fs = fsspec.filesystem("zip", fo=archive_url)
    image_paths = sorted(
        path for path in fs.find("/") if path.lower().endswith(IMAGE_EXTENSIONS)
    )

    if not image_paths:
        raise ValueError(f"No image files found in archive '{archive_path}'.")

    first_image_path = image_paths[0]
    with fs.open(first_image_path, "rb") as fp:
        image_bytes = fp.read()

    image = Image.open(BytesIO(image_bytes))
    image.load()

    label_examples = _extract_label_examples(image_paths, selected_split)
    target_name = "label" if label_examples else None

    sample_record: Dict[str, List[Any]] = {"image": [image]}
    if target_name:
        sample_record[target_name] = [label_examples[0]]

    dataset = Dataset.from_dict(sample_record)

    metadata = {
        "available_splits": available_splits,
        "selected_split": selected_split,
        "num_samples_in_selected_split": len(image_paths),
        "target_name": target_name,
        "target_values": label_examples,
    }

    return dataset, metadata


def load_dentex_dataset_split(
    dataset_id: str = DEFAULT_DENTEX_DATASET_ID,
    split: str = "train",
    config_name: Optional[str] = None,
) -> Tuple[DatasetDict, Dict[str, Any]]:
    """
    Load one split from a Hugging Face dataset.

    1) Try standard load_dataset(dataset_id, split=...).
    2) If it fails (mixed split formats/cache issues), use archive metadata fallback
       that reads the selected split zip directly from the Hub.
    """
    normalized_split = _normalize_split_name(split)

    try:
        dataset = _load_single_split_standard(
            dataset_id=dataset_id,
            split=normalized_split,
            config_name=config_name,
        )
        metadata = {
            "available_splits": [normalized_split],
            "selected_split": normalized_split,
            "num_samples_in_selected_split": int(len(dataset)),
            "target_name": None,
            "target_values": [],
        }
        return DatasetDict({normalized_split: dataset}), metadata

    except Exception as primary_exc:
        try:
            dataset, metadata = _load_single_split_from_archives(
                dataset_id=dataset_id,
                split=normalized_split,
            )
            selected_split = metadata["selected_split"]
            return DatasetDict({selected_split: dataset}), metadata

        except Exception as fallback_exc:
            raise RuntimeError(
                "Failed to load dataset split via standard and archive fallback paths. "
                f"standard_error={primary_exc}; fallback_error={fallback_exc}"
            ) from fallback_exc
