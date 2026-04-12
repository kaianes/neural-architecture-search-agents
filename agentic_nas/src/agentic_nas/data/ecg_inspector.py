from typing import Any, Dict
import pandas as pd

from agentic_nas.data.base_schema import build_observation_schema


def inspect_ecg_sleep_apnea_dataframe(df: pd.DataFrame, local_path: str) -> Dict[str, Any]:
    target_column = "Target" if "Target" in df.columns else None

    feature_columns = [col for col in df.columns if col != target_column]
    numeric_feature_columns = [
        col for col in feature_columns
        if pd.api.types.is_numeric_dtype(df[col])
    ]

    target_values = None
    target_distribution = None
    if target_column is not None:
        target_values = sorted(df[target_column].dropna().astype(str).unique().tolist())
        target_distribution = df[target_column].astype(str).value_counts(dropna=False).to_dict()

    missing_values_by_column = df.isna().sum().to_dict()
    missing_values_total = int(df.isna().sum().sum())
    duplicate_rows = int(df.duplicated().sum())

    # simple pattern evidence for columns like "0", "1", ..., "2499"
    sequential_numeric_name_count = sum(col.isdigit() for col in feature_columns)
    feature_column_name_pattern = "unknown"
    if len(feature_columns) > 0 and sequential_numeric_name_count == len(feature_columns):
        feature_column_name_pattern = "numeric_sequential"

    raw_observations = {
        "num_samples": int(df.shape[0]),
        "num_columns_total": int(df.shape[1]),
        "num_input_features": int(len(feature_columns)),
        "feature_column_name_pattern": feature_column_name_pattern,
        "feature_column_sample": feature_columns[:10],
        "feature_dtypes_summary": {
            col: str(df[col].dtype) for col in feature_columns[:20]
        },
        "all_feature_numeric": len(numeric_feature_columns) == len(feature_columns),
    }

    target_info = {
        "target_name": target_column,
        "target_values": target_values,
        "target_distribution": target_distribution,
    }

    quality_observations = {
        "missing_values_total": missing_values_total,
        "missing_values_by_column_sample": {
            k: missing_values_by_column[k]
            for k in list(missing_values_by_column.keys())[:20]
        },
        "duplicate_rows": duplicate_rows,
    }

    modality_specific_observations = {
        "storage_format": "csv",
        "sample_representation": "single_row_numeric_vector",
        "approx_vector_length_per_sample": int(len(feature_columns)),
    }

    return build_observation_schema(
        dataset_id="sleep_apnea_ecg",
        dataset_name="ECG Data for Sleep Apnea Detection",
        source_type="kaggle",
        source_uri="ucimachinelearning/ecg-data-for-sleep-apnea-detection",
        local_path=local_path,
        raw_observations=raw_observations,
        target_info=target_info,
        quality_observations=quality_observations,
        modality_specific_observations=modality_specific_observations,
        notes=[
            "This schema contains observations only, not final semantic conclusions.",
            "The agent must infer modality and likely task type from these observations.",
        ],
    )