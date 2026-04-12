import json
from typing import Optional

from crewai.tools import tool

from agentic_nas.data.csv_loader import load_csv_dataset
from agentic_nas.data.ecg_inspector import inspect_ecg_sleep_apnea_dataframe
from agentic_nas.data.hf_image_loader import (
    DEFAULT_DENTEX_DATASET_ID,
    load_dentex_dataset_split,
)
from agentic_nas.data.xray_inspector import inspect_xray_dataset_dict


@tool("inspect_sleep_apnea_dataset")
def inspect_sleep_apnea_dataset(file_path: str) -> str:
    """
    Loads the ECG sleep apnea CSV dataset, extracts objective observations,
    and returns a JSON schema for agent interpretation.
    """
    df = load_csv_dataset(file_path)
    schema = inspect_ecg_sleep_apnea_dataframe(df, local_path=file_path)
    return json.dumps(schema, indent=2)


@tool("inspect_xray_dataset")
def inspect_xray_dataset(
    dataset_id: str = DEFAULT_DENTEX_DATASET_ID,
    split: str = "train",
    config_name: Optional[str] = None,
) -> str:
    """
    Loads a Hugging Face X-ray dataset and returns objective observations using the shared schema.
    """
    dataset, split_metadata = load_dentex_dataset_split(
        dataset_id=dataset_id,
        split=split,
        config_name=config_name,
    )

    schema = inspect_xray_dataset_dict(
        dataset=dataset,
        dataset_id=dataset_id,
        dataset_name=dataset_id,
        requested_split=split_metadata.get("selected_split", split),
        split_metadata=split_metadata,
    )
    return json.dumps(schema, indent=2)
