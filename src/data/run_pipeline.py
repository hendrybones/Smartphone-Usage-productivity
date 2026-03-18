from src.data.ingestion import ingest_data, load_config
from src.data.cleaning import clean_data
from src.data.validation import run_data_validation
from src.data.preprocessors import preprocess_data

def main():
    config = load_config()
    
    ingest_data(config)
    clean_data(config)
    
    # Optional: stop if validation fails
    df = clean_data(config)  # if your clean_data returns df
    if run_data_validation(df, config):
        preprocess_data(config)


if __name__ == "__main__":
    main()