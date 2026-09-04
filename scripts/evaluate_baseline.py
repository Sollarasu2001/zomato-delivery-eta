"""Evaluate the naive mean regression baseline."""

from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score


ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data/interim"

TARGET_COLUMN = "Time_taken(min)"


def main():
    """Evaluate the training-target mean on the test set."""
    y_train = pd.read_parquet(DATA_DIR / "y_train.parquet")[TARGET_COLUMN]

    y_test = pd.read_parquet(DATA_DIR / "y_test.parquet")[TARGET_COLUMN]

    prediction = float(y_train.mean())

    predictions = np.full(shape=len(y_test),fill_value=prediction)

    mae = mean_absolute_error(y_test,predictions)
    mse = mean_squared_error(y_test,predictions)
    rmse = mse ** 0.5
    r2 = r2_score(y_test,predictions)

    print("\nNaive Mean Baseline")
    print("===================")
    print(f"Prediction: {prediction:.4f}")
    print(f"MAE:        {mae:.4f}")
    print(f"RMSE:       {rmse:.4f}")
    print(f"R²:         {r2:.4f}")


if __name__ == "__main__":
    main()