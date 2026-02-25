from src.data.ingestation import ingest_data, load_config   
from src.data.validation import run_data_validation
from src.data.preprocessors import preprocess_data

if __name__ == "__main__":
    config = load_config()
    
    df = ingest_data(config)
    
    if run_data_validation(df, config):
        preprocess_data(config)
    print