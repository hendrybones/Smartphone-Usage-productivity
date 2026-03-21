import pandas as pd
from pathlib import Path
import yaml
import logging

logging.basicConfig(level=logging.INFO)

def load_config(config_path="config/config.yaml"):
    with open(config_path) as f:
        return yaml.safe_load(f)

def engineer_features(config):

    input_path = Path(config['data']['processed_path']) / "cleaned_data.parquet"
    output_dir = Path(config['data']['processed_path'])

    df = pd.read_parquet(input_path)

    logging.info("Creating new features")

    # Social media usage ratio
    df["social_media_ratio"] = df["Social_Media_Hours"] / df["Daily_Phone_Hours"]

    # Weekend usage increase
    df["weekend_usage_increase"] = (
        df["Weekend_Screen_Time_Hours"] - df["Daily_Phone_Hours"]
    )

    # Productivity efficiency
    df["productivity_efficiency"] = (
        df["Work_Productivity_Score"] / df["Daily_Phone_Hours"]
    )

    # Stress per phone hour
    df["stress_per_hour"] = df["Stress_Level"] / df["Daily_Phone_Hours"]

    # Sleep deficit
    df["sleep_deficit"] = 8 - df["Sleep_Hours"]

    # App usage intensity
    df["app_usage_intensity"] = (
        df["App_Usage_Count"] / df["Daily_Phone_Hours"]
    )
    # digital addiction risk score
    df["digital_addiction_score"] = (
    df["Daily_Phone_Hours"] * 0.4 +
    df["Social_Media_Hours"] * 0.3 +
    (df["App_Usage_Count"] / 50) * 0.3
    )
    # Work life balance  index
    df["work_life_balance"] = (
    (df["Sleep_Hours"] / 8) *
    df["Work_Productivity_Score"] /
    df["Daily_Phone_Hours"]
    )
    # Night time phone usage
    df["late_night_user"] = (df["Daily_Phone_Hours"] > 6).astype(int)
    
    # Stress risk index
    df["stress_risk_index"] = (
    df["Stress_Level"] +
    (8 - df["Sleep_Hours"]) +
    (df["Daily_Phone_Hours"] / 2)
    )
    
    # Productivity Loss Score (This measures how much productivity might be lost due to phone usage)
    df["productivity_loss_score"] = (
    df["Daily_Phone_Hours"] *
    df["Social_Media_Hours"] /
    df["Work_Productivity_Score"]
    )
    
    # Weekend Behaviour shift (This captures how much more they use their phone on weekends compared to weekdays)
    df["weekend_behavior_shift"] = (
    df["Weekend_Screen_Time_Hours"] /
    df["Daily_Phone_Hours"])
    
    ## adding interaction features
    df["distraction_index"] = (
    df["Daily_Phone_Hours"] * df["Social_Media_Hours"]
    )
    # Stress - sleep interaction
    df["stress_sleep_interaction"] = (
    df["Stress_Level"] * (8 - df["Sleep_Hours"]))
    
    # Phone usage per age
    df["phone_usage_per_age"] = (
    df["Daily_Phone_Hours"] / df["Age"])
    
    # productivity vs phone usage
    df["productivity_usage_ratio"] = (
    df["Work_Productivity_Score"] / df["Daily_Phone_Hours"])
    
    # Social Media addiction flag
    df["social_media_addict"] = (
    df["Social_Media_Hours"] > 3).astype(int)
    

    output_path = output_dir / "features.parquet"

    df.to_parquet(output_path, index=False)

    logging.info(f"Feature dataset saved → {output_path}")
    print(df.columns)

    return df


if __name__ == "__main__":
    config = load_config()
    engineer_features(config)
    