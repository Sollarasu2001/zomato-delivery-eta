"""Train And Evaluate The Ridge Regression Model"""
from pathlib import Path
import sys


import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import Pipeline

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from delivery_eta.features.preprocessing import build_preprocessor
from delivery_eta.models.elastic_net import build_elastic_net

DATA_DIR = ROOT / "data/interim"

TARGET_COLUMN = "Time_taken(min)"

def load_training_data():
    """Load Train And Test Datasets."""
    x_train = pd.read_parquet(DATA_DIR/"x_train.parquet")
    x_test = pd.read_parquet(DATA_DIR/"x_test.parquet")
    y_train = pd.read_parquet(DATA_DIR / "y_train.parquet")[TARGET_COLUMN]
    y_test = pd.read_parquet(DATA_DIR / "y_test.parquet")[TARGET_COLUMN]

    return x_train, x_test, y_train, y_test


def main():
    """Train and Evaluate Ridge Regression."""
    x_train, x_test, y_train, y_test = load_training_data()

    pipeline = Pipeline(
        steps = [
            ("preprocessor", build_preprocessor()),
            ("model", build_elastic_net(alpha=0.01, l1_ratio=0.5))
        ]
    )

    pipeline.fit(x_train,y_train)

    predictions = pipeline.predict(x_test)

    mae = mean_absolute_error(y_test, predictions)
    mse = mean_squared_error(y_test, predictions)
    rmse = mse ** 0.5
    r2 = r2_score(y_test, predictions)

    print("\nElasticNet Regression Baseline")
    print("==========================")
    print(f"MAE:  {mae:.4f}")
    print(f"RMSE: {rmse:.4f}")
    print(f"R²:   {r2:.4f}")


if __name__ == "__main__":
    main()