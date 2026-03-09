import pandas as pd
from pathlib import Path
import logging
import yaml

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def load_config(config_path: str = "config/config.yaml"):
    with open(config_path) as f:
        return yaml.safe_load(f)

def clean_data(config: dict):
    input_path = Path(config['data']['interim_path']) / "raw_data.parquet"
    output_path = Path(config['data']['processed_path'])
    output_path.mkdir(parents=True, exist_ok=True)
    
    logging.info(f"Reading interim data: {input_path}")
    df = pd.read_parquet(input_path)
    
    # Example cleaning steps
    df = df.drop_duplicates()
    df = df.fillna({
        "Work_Productivity_Score": df["Work_Productivity_Score"].median(),
        "Stress_Level": df["Stress_Level"].median()
    })
    
    logging.info(f"Saving cleaned data to processed folder")
    df.to_parquet(output_path / "cleaned_data.parquet", index=False)
    logging.info(f"Saved cleaned data → {output_path / 'cleaned_data.parquet'}")
    return df

if __name__ == "__main__":
    config = load_config()
    clean_data(config)