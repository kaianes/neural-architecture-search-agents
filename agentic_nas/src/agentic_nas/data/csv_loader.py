from pathlib import Path
import pandas as pd


def load_csv_dataset(file_path: str) -> pd.DataFrame:
    path = Path(file_path)

    if not path.exists():
        raise FileNotFoundError(f"Dataset file not found: {file_path}")

    df = pd.read_csv(path)
    df.columns = [str(col) for col in df.columns]
    return df