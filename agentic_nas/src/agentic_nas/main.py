import argparse
import os
from datetime import datetime
from pathlib import Path

from dotenv import load_dotenv

from agentic_nas.data.hf_image_loader import DEFAULT_DENTEX_DATASET_ID
from agentic_nas.crew import build_crew


def run() -> None:
    project_root = Path(__file__).resolve().parents[2]
    load_dotenv(project_root / ".env")

    default_ecg_path = project_root / "data" / "raw" / "sleep_apnea_ecg.csv"

    parser = argparse.ArgumentParser(description="Run Agentic NAS dataset profiling.")
    parser.add_argument("--mode", choices=["ecg", "xray"], default=os.getenv("PROFILE_MODE", "ecg"))
    parser.add_argument("--dataset-path", default=os.getenv("ECG_DATASET_PATH", str(default_ecg_path)))
    parser.add_argument("--hf-dataset-id", default=os.getenv("HF_DATASET_ID", DEFAULT_DENTEX_DATASET_ID))
    parser.add_argument("--hf-config-name", default=os.getenv("HF_CONFIG_NAME"))
    parser.add_argument("--hf-split", default=os.getenv("HF_SPLIT", "train"))
    parser.add_argument("--output-file", default=os.getenv("PROFILE_OUTPUT_FILE"))
    args = parser.parse_args()

    crew = build_crew(
        profile_mode=args.mode,
        dataset_path=args.dataset_path,
        hf_dataset_id=args.hf_dataset_id,
        hf_split=args.hf_split,
        hf_config_name=args.hf_config_name,
    )
    result = crew.kickoff()

    output_dir = project_root / "data" / "outputs"
    output_dir.mkdir(parents=True, exist_ok=True)

    run_date = datetime.now().strftime("%Y-%m-%d")
    default_output_name = f"{args.mode}_profile_{run_date}.md"
    output_file = Path(args.output_file) if args.output_file else output_dir / default_output_name

    if not output_file.is_absolute():
        output_file = project_root / output_file

    output_file.parent.mkdir(parents=True, exist_ok=True)
    output_file.write_text(str(result), encoding="utf-8")


if __name__ == "__main__":
    run()
