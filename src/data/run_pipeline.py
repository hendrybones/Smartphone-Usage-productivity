from pathlib import Path
import pandas as pd
import sys

sys.path.append("../src/models")
sys.path.append("../src/features")

project_root = Path(r"C:\Users\Administrator\OneDrive\Documents\Data Science\Smartphone-Usage-productivity")

from ingestation import ingest_data, load_config
from cleaning import clean_data
from validation import run_data_validation
from preprocessors import preprocess_data
from feature_engineering import engineer_features     


def main():
    config = load_config()
    
    print("Starting...")
    
    ingest_data(config)
    clean_data(config)
    
    cleaned_file = Path(config['data']['processed_path']) / "cleaned_data.parquet"
    
    if not cleaned_file.exists():
        print("Error: cleaned_data.parquet missing")
        return
    
    df = pd.read_parquet(cleaned_file)
    print(f"Loaded: {df.shape}")
    
    if run_data_validation(df, config):
        preprocess_data(config)
        engineer_features(config)
        print("Done – check features.parquet")
    else:
        print("Validation failed")


if __name__ == "__main__":
    main()