"""Verify the persisted Polynomial Ridge model."""

from pathlib import Path
import sys

import joblib
import pandas as pd
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

DATA_DIR = ROOT / "data/interim"
MODEL_PATH = ROOT / "models/polynomial_ridge.joblib"

TARGET_COLUMN = "Time_taken(min)"


def main():
    """Load the saved model and verify inference."""
    model = joblib.load(
        MODEL_PATH
    )

    x_test = pd.read_parquet(
        DATA_DIR / "x_test.parquet"
    )

    y_test = pd.read_parquet(
        DATA_DIR / "y_test.parquet"
    )[TARGET_COLUMN]

    predictions = model.predict(
        x_test
    )

    print("\nModel Verification")
    print("===================")

    print(
        f"Model type: {type(model).__name__}"
    )

    print(
        f"Test rows: {len(x_test)}"
    )

    print(
        f"Predictions: {len(predictions)}"
    )

    print(
        f"Prediction dtype: {predictions.dtype}"
    )

    print("\nFirst 10 predictions:")

    comparison = pd.DataFrame(
        {
            "actual": y_test.iloc[:10].values,
            "predicted": predictions[:10],
        }
    )

    print(
        comparison.to_string(
            index=False
        )
    )

    print("\nValidation checks:")

    print(
        f"Contains NaN predictions: "
        f"{np.isnan(predictions).any()}"
    )

    print(
        f"Contains infinite predictions: "
        f"{np.isinf(predictions).any()}"
    )

    print(
        f"Minimum prediction: "
        f"{predictions.min():.4f}"
    )

    print(
        f"Maximum prediction: "
        f"{predictions.max():.4f}"
    )


if __name__ == "__main__":
    main()