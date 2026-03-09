import pandas as pd
from pathlib import Path
import logging
import yaml

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


def load_config(config_path: str = "config/config.yaml"):
    with open(config_path) as f:
        return yaml.safe_load(f)


def ingest_data(config: dict):
    raw_path = Path(config['data']['raw_path'])
    interim_dir = Path(config['data']['interim_path'])

    interim_dir.mkdir(parents=True, exist_ok=True)

    logging.info(f"Reading raw CSV: {raw_path}")

    df = pd.read_csv(raw_path, dtype={
        "User_ID": "string",
        "Age": "int8",
        "Gender": "category",
        "Occupation": "category",
        "Device_Type": "category",
        "Work_Productivity_Score": "int8",
        "Stress_Level": "int8",
        "App_Usage_Count": "int16",
        "Caffeine_Intake_Cups": "int8"
    })

    logging.info(f"Loaded {len(df):,} rows, {df.shape[1]} columns")

    output_path = interim_dir / "raw_data.parquet"

    df.to_parquet(output_path, index=False, compression="snappy")

    logging.info(f"Saved parquet → {output_path}")

    return df


if __name__ == "__main__":
    config = load_config()
    ingest_data(config)