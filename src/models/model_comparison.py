
import pandas as pd
from pathlib import Path
import yaml
import logging
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, ExtraTreesRegressor
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split

logging.basicConfig(level=logging.INFO)

# -------------------------
# Load configuration
# -------------------------
def load_config(config_path="config/config.yaml"):
    with open(config_path) as f:
        return yaml.safe_load(f)

# -------------------------
# Load preprocessed feature dataset
# -------------------------
def load_data(config):
    data_path = Path(config['data']['processed']) / "features.parquet"
    logging.info(f"Loading data → {data_path}")
    df = pd.read_parquet(data_path)
    return df

# -------------------------
# Define regression models
# -------------------------
def get_models():
    return {
        "Linear Regression": LinearRegression(),
        "Ridge": Ridge(),
        "Lasso": Lasso(),
        "ElasticNet": ElasticNet(),
        "Decision Tree": DecisionTreeRegressor(),
        "Random Forest": RandomForestRegressor(),
        "Extra Trees": ExtraTreesRegressor(),
        "Gradient Boosting": GradientBoostingRegressor(),
        "SVR": SVR()
    }

# -------------------------
# Evaluate a single model
# -------------------------
def evaluate_model(model, X_train, y_train, X_test, y_test):
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    mse = mean_squared_error(y_test, y_pred)
    rmse = mse ** 0.5
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    return {"RMSE": rmse, "MAE": mae, "R2 Score": r2}

# -------------------------
# Compare all models
# -------------------------
def compare_models(X_train, y_train, X_test, y_test):
    models = get_models()
    results = []

    for name, model in models.items():
        logging.info(f"Training and evaluating → {name}")
        metrics = evaluate_model(model, X_train, y_train, X_test, y_test)
        results.append({"Model": name, **metrics})

    results_df = pd.DataFrame(results).sort_values(by="R2 Score", ascending=False)
    return results_df

