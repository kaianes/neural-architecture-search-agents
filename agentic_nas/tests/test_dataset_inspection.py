import pytest

from agentic_nas.data.csv_loader import load_csv_dataset
from agentic_nas.data.ecg_inspector import inspect_ecg_sleep_apnea_dataframe


def test_ecg_inspector_reports_shape_target_quality(ecg_2500_csv):
    df = load_csv_dataset(str(ecg_2500_csv))

    schema = inspect_ecg_sleep_apnea_dataframe(df, local_path=str(ecg_2500_csv))

    raw = schema["raw_observations"]
    target = schema["target_info"]
    quality = schema["quality_observations"]
    modality = schema["modality_specific_observations"]

    assert raw["num_samples"] == 4
    assert raw["num_columns_total"] == 2501
    assert raw["num_input_features"] == 2500
    assert raw["all_feature_numeric"] is True
    assert raw["feature_column_name_pattern"] == "numeric_sequential"
    assert target["target_name"] == "Target"
    assert sorted(target["target_values"]) == ["Normal", "Sleep Apnea"]
    assert target["target_distribution"] == {"Normal": 2, "Sleep Apnea": 2}
    assert quality["missing_values_total"] == 1
    assert quality["duplicate_rows"] == 1
    assert modality["approx_vector_length_per_sample"] == 2500


def test_csv_loader_missing_file_raises_clear_error(tmp_path):
    missing_path = tmp_path / "missing.csv"

    with pytest.raises(FileNotFoundError, match="Dataset file not found"):
        load_csv_dataset(str(missing_path))
