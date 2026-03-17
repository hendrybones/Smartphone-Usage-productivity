import pandas as pd
from pathlib import Path
import yaml
import logging
import sys

from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, ExtraTreesRegressor
from sklearn.svm import SVR

from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split


# ─────────────────────────────────────────────────────────────
# FORCE logging to show in Jupyter / VS Code / scripts
# ─────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)s | %(message)s',
    datefmt='%H:%M:%S',
    handlers=[logging.StreamHandler(sys.stdout)],
    force=True
)
logger = logging.getLogger(__name__)


# -------------------------
# Load configuration
# -------------------------
def load_config(config_path="config/config.yaml"):
    """Load YAML config."""
    logger.info(f"Loading config from → {config_path}")
    with open(config_path, encoding="utf-8") as f:
        config = yaml.safe_load(f)
    logger.info("✅ Config loaded successfully")
    return config


# -------------------------
# Load dataset
# -------------------------
def load_data(config):
    """Load preprocessed features dataset."""
    data_path = Path(config['data']['processed_path']) / "features.parquet"

    logger.info(f"Loading data → {data_path}")
    df = pd.read_parquet(data_path)
    logger.info(f"✅ Data loaded: {df.shape[0]:,} rows × {df.shape[1]} columns")
    return df


# -------------------------
# Prepare train/test data (FIXED - Bulletproof One-Hot Encoding)
# -------------------------
def prepare_data(df, config):
    """Split + Drop IDs + Bool→int + One-Hot Encode (drop_first)"""
    target = config["preprocessing"]["target"]
    test_size = config["preprocessing"]["test_size"]
    random_state = config["preprocessing"]["random_state"]

    X = df.drop(columns=[target])
    y = df[target]

    # 1. Drop pure ID columns (User_ID, index, etc.)
    id_cols = [col for col in X.columns if 
               (X[col].nunique() == len(X)) or 
               any(word in col.lower() for word in ['id', 'user', 'index'])]
    if id_cols:
        logger.warning(f"🗑️  Dropping ID columns: {id_cols}")
        X = X.drop(columns=id_cols)

    # 2. Convert boolean columns to 0/1
    bool_cols = X.select_dtypes(include=['bool']).columns.tolist()
    if bool_cols:
        X[bool_cols] = X[bool_cols].astype(int)
        logger.info(f"✅ Converted boolean columns to 0/1: {bool_cols}")

    # 3. One-hot encode ALL categorical columns (object + category + string dtypes)
    #    This catches 'Gender' ('Male'), 'OS', etc. even if dtype is not plain 'object'
    logger.info("🔥 Applying one-hot encoding (drop_first=True + dtype=int)")
    X = pd.get_dummies(X, drop_first=True, dtype=int)

    # 4. Safety check - make sure NO strings left
    non_numeric = X.select_dtypes(exclude=['number']).columns.tolist()
    if non_numeric:
        logger.error(f"❌ STILL have non-numeric columns: {non_numeric}")
        raise ValueError(f"Data still contains strings after encoding: {non_numeric}")

    logger.info(f"✅ After one-hot encoding: {X.shape[1]} features (all numeric)")

    # Final split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )

    logger.info(f"✅ Data ready → Train: {X_train.shape[0]:,} rows × {X_train.shape[1]} cols | Test: {X_test.shape[0]:,}")
    return X_train, X_test, y_train, y_test


# -------------------------
# Define models
# -------------------------
def get_models():
    """Return dictionary of regression models."""
    return {
        "Linear Regression": LinearRegression(),
        "Ridge": Ridge(),
        "Lasso": Lasso(),
        "ElasticNet": ElasticNet(),
        "Decision Tree": DecisionTreeRegressor(random_state=42),
        "Random Forest": RandomForestRegressor(random_state=42, n_jobs=-1),
        "Extra Trees": ExtraTreesRegressor(random_state=42, n_jobs=-1),
        "Gradient Boosting": GradientBoostingRegressor(random_state=42),
        "SVR": SVR()
    }


# -------------------------
# Evaluate a single model
# -------------------------
def evaluate_model(model, X_train, y_train, X_test, y_test):
    """Fit model and calculate evaluation metrics."""
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    mse = mean_squared_error(y_test, y_pred)
    rmse = mse ** 0.5
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    return {"RMSE": round(rmse, 4), "MAE": round(mae, 4), "R2 Score": round(r2, 4)}


# -------------------------
# Compare all models
# -------------------------
def compare_models(X_train, y_train, X_test, y_test):
    """Train and evaluate all models, return sorted DataFrame of results."""
    models = get_models()
    results = []

    logger.info("Starting model comparison...")

    for name, model in models.items():
        logger.info(f"Training → {name}")
        metrics = evaluate_model(model, X_train, y_train, X_test, y_test)
        results.append({"Model": name, **metrics})
        logger.info(f"   → RMSE: {metrics['RMSE']:.4f} | MAE: {metrics['MAE']:.4f} | R²: {metrics['R2 Score']:.4f}")

    results_df = pd.DataFrame(results).sort_values(by="R2 Score", ascending=False)

    print("\n" + "="*80)
    print("FINAL MODEL COMPARISON (Best → Worst)")
    print("="*80)
    print(results_df.round(4))
    print("="*80)

    logger.info("Model comparison completed!")
    return results_df


# =========================
# RUN EVERYTHING
# =========================
if __name__ == "__main__":
    logger.info("🚀 Starting Smartphone Usage → Productivity Model Comparison")

    config = load_config()
    df = load_data(config)
    X_train, X_test, y_train, y_test = prepare_data(df, config)
    results = compare_models(X_train, y_train, X_test, y_test)

    # Save results
    output_dir = Path("results")
    output_dir.mkdir(exist_ok=True)
    results.to_csv(output_dir / "model_comparison_results.csv", index=False)
    logger.info(f"✅ Results saved → results/model_comparison_results.csv")

    logger.info("🎉 Pipeline completed successfully!")