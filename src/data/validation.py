import pandas as pd
import logging
from pathlib import Path

logging.basicConfig(level=logging.INFO)

def run_data_validation(df: pd.DataFrame, config: dict):
    logging.info("=== Starting data validation ===")
    
    # Schema validation
    expected_columns =[
        "User_ID", "Age", "Gender", "Occupation", "Device_Type",
        "Daily_Phone_Hours", "Social_Media_Hours", "Work_Productivity_Score",
        "Sleep_Hours", "Stress_Level", "App_Usage_Count", "Caffeine_Intake_Cups",
        "Weekend_Screen_Time_Hours"
    ]
    
    if list(df.columns) != expected_columns:
        logging.error(f"Schema validation failed! Expected columns: {expected_columns}, but got: {list(df.columns)}")
        raise ValueError("Schema validation failed!")
    
    # Data types & range checks
    assert df["Age"].between(18, 60).all(), "Age outside 18-60"
    assert df["Daily_Phone_Hours"].between(0, 24).all(), "Invalid phone hours"
    assert df["Weekend_Screen_Time_Hours"].between(0, 24).all(), "Invalid weekend hours"
    assert df["Work_Productivity_Score"].between(1, 10).all(), "Score not in 1-10"
    assert df["Stress_Level"].between(1, 10).all(), "Stress level invalid"
    assert df["Caffeine_Intake_Cups"].between(0, 10).all(), "Caffeine cups invalid"
    
    # Categorical value checks
    assert df["Gender"].isin(["Male", "Female", "Other"]).all(), "Invalid Gender"
    assert df["Device_Type"].isin(["Android", "iOS"]).all(), "Invalid Device_Type"
    valid_occ = ["Professional", "Student", "Business Owner", "Freelancer"]
    assert df["Occupation"].isin(valid_occ).all(), "Invalid Occupation"
    
    # Businness rules
    assert not df["User_ID"].duplicated().any(), "Duplicate User_IDs found"
    assert df.isnull().sum().sum() == 0, f"Missing values found: {df.isnull().sum()}"
    logging.info("✅ All validations PASSED!")
    return True