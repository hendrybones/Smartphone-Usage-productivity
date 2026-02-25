import pandas as pd
def clean_data(df):
    """
    Clean the DataFrame by handling missing values and removing duplicates.

    Parameters:
    df (pd.DataFrame): The DataFrame to be cleaned.

    Returns:
    pd.DataFrame: A cleaned DataFrame.
    """
    try:
        # Handle missing values by filling them with the mean of the column
        df_filled = df.fillna(df.mean())
        
        # Remove duplicate rows
        df_cleaned = df_filled.drop_duplicates()
        
        print("Data cleaned successfully.")
        return df_cleaned
    except Exception as e:
        print(f"An error occurred while cleaning data: {e}")
        return None