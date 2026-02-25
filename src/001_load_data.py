import pandas as pd
def load_data(file_path):
    """
    Load data from a CSV file.

    Parameters:
    file_path (str): The path to the CSV file.

    Returns:;l
    pd.DataFrame: A DataFrame containing the loaded data.
    """
    try:
        data = pd.read_csv("../../Raw Data/Smartphone_Usage_Productivity_Dataset_50000.csv")
        print(f"Data loaded successfully from {file_path}")
        return data
    except Exception as e:
        print(f"An error occurred while loading data: {e}")
        return None
    