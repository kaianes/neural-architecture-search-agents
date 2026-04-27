from agentic_nas.agents.report_agent import ReportAgent


def test_report_contains_required_sections_and_modality(evaluator_output):
    markdown = ReportAgent().generate_report(evaluator_output)

    required_sections = [
        "## Identified problem",
        "## Data modality",
        "## Chosen strategy",
        "## Architecture families considered",
        "## Best configuration found",
        "## Final performance",
        "## Justification for the choice",
        "## Limitations",
        "## Next steps",
    ]
    for section in required_sections:
        assert section in markdown

    assert "## Data modality\n1D Time-Series/Signal (ECG)" in markdown
    assert "## Data modality\nNot available." not in markdown


def test_report_uses_nested_upstream_data_modality(evaluator_output):
    nested_payload = {
        **evaluator_output,
        "profile_summary": {},
        "trainer_output_json": {
            "profile_summary": {
                "data_modality": "1D Time-Series / Signal (ECG)",
            }
        },
    }

    markdown = ReportAgent().generate_report(nested_payload)

    assert "## Data modality\n1D Time-Series/Signal (ECG)" in markdown
