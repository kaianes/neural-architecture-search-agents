import pytest

from agentic_nas.training.search_space_parser import (
    SearchSpaceParseError,
    SearchSpaceParser,
    parse_search_space_param,
)


def test_builder_search_space_with_cnn1d_and_mlp_is_valid(builder_output):
    parser = SearchSpaceParser(builder_output)

    assert parser.validate() is True
    assert [family["family"] for family in parser.get_selected_families()] == ["cnn1d", "mlp"]
    assert parser.get_profile_summary()["data_modality"] == "1D Time-Series/Signal (ECG)"


def test_supported_hyperparameter_definitions_parse_correctly():
    int_param = parse_search_space_param(
        "num_layers", {"type": "int", "min": 1, "max": 3, "step": 1}
    )
    float_param = parse_search_space_param(
        "learning_rate", {"type": "float", "min": 0.0001, "max": 0.01, "log": True}
    )
    stepped_float = parse_search_space_param(
        "dropout_rate", {"type": "float", "min": 0.1, "max": 0.5, "step": 0.1}
    )
    categorical_param = parse_search_space_param(
        "activation", {"type": "categorical", "values": ["relu", "selu"]}
    )
    bool_param = parse_search_space_param("use_batch_norm", {"type": "bool"})

    assert int_param.param_type == "int"
    assert int_param.kwargs == {"low": 1, "high": 3, "step": 1}
    assert float_param.param_type == "float"
    assert float_param.kwargs["log"] is True
    assert stepped_float.kwargs["step"] == 0.1
    assert categorical_param.param_type == "categorical"
    assert categorical_param.kwargs == {"choices": ["relu", "selu"]}
    assert bool_param.param_type == "categorical"
    assert bool_param.kwargs == {"choices": [True, False]}


@pytest.mark.parametrize(
    "param_spec,error_text",
    [
        ({"type": "int", "min": 3, "max": 1}, "min >= max"),
        ({"type": "float", "min": 0.1}, "must have 'min' and 'max'"),
        ({"type": "categorical", "values": []}, "non-empty list"),
        ({"type": "unknown"}, "invalid type"),
    ],
)
def test_invalid_hyperparameter_definitions_raise_clear_errors(builder_output, param_spec, error_text):
    builder_output["search_space"]["mlp"]["hyperparameters"]["bad_param"] = param_spec

    with pytest.raises(SearchSpaceParseError, match=error_text):
        SearchSpaceParser(builder_output).validate()


def test_missing_required_builder_keys_raise_clear_error(builder_output):
    del builder_output["search_space"]

    with pytest.raises(SearchSpaceParseError, match="Missing required key: search_space"):
        SearchSpaceParser(builder_output).validate()
