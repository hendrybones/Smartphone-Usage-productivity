import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import joblib
from pathlib import Path
import logging


logging.basicConfig(level=logging.INFO)

def build_preprocessor(config: dict):
    logging.info("Building preprocessing pipeline...")
    numeric_features = ["Age", "Daily_Phone_Hours", "Social_Media_Hours", "Sleep_Hours",
        "Stress_Level", "App_Usage_Count", "Caffeine_Intake_Cups",
        "Weekend_Screen_Time_Hours", "Total_Screen_Time"]
    categorical_features = ["Gender", "Occupation", "Device_Type"]
    
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), numeric_features),
            ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), categorical_features)
        ]
    )
    return preprocessor

def preprocess_data(config: dict):
    logging.info("Preprocessing data...")
    interim_path = Path(config['data']['interim_path'])/"raw_data.parquet"
    df = pd.read_parquet(interim_path)
    
    # === FEATURE ENGINEERING (Field Processor) ===
    df = df.copy()
    df["Total_Screen_Time"] = df["Daily_Phone_Hours"] + df["Weekend_Screen_Time_Hours"]
    df["Social_Media_Ratio"] = df["Social_Media_Hours"] / (df["Total_Screen_Time"] + 1e-6)
    
    target = config["preprocessing"]["target"]
    X = df.drop(columns=[target, "User_ID"])    
    y = df[target]
    
    # Train-test split
    X_train,X_test, y_train,y_test = train_test_split(X,y, 
                                                      test_size=config["preprocessing"]["test_size"],
                                                      random_state=config["preprocessing"]["random_state"],
                                                      stratify=None)
    
    # Build & fit preprocessor
    preprocessor= build_preprocessor(config)
    X_train_processed = preprocessor.fit_transform(X_train)
    X_test_processed = preprocessor.transform(X_test)
    
    # Save preprocessor
    processed_dir = Path(config['data']['processed_path'])
    processed_dir.mkdir(parents=True, exist_ok=True)    
    
    np.save(processed_dir/"X_train.npy", X_train_processed)
    np.save(processed_dir/"X_test.npy", X_test_processed)
    np.save(processed_dir/"y_train.npy", y_train)
    np.save(processed_dir/"y_test.npy", y_test)
    joblib.dump(preprocessor, processed_dir/"preprocessor.joblib")
    logging.info(f"Preprocessing complete! Processed data saved to {processed_dir}") 
    logging.info(f"X_train shape: {X_train_processed.shape}, X_test shape: {X_test_processed.shape}")
    return X_train_processed, X_test_processed, y_train, y_test   
    