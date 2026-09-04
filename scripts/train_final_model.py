"""Train and persist the final Polynomial Ridge model."""

from pathlib import Path
import sys

import joblib
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from delivery_eta.features.polynomial_preprocessing import (
    build_polynomial_preprocessor,
)


DATA_DIR = ROOT / "data/interim"
MODEL_DIR = ROOT / "models"
MODEL_PATH = MODEL_DIR / "polynomial_ridge.joblib"

TARGET_COLUMN = "Time_taken(min)"


def main():
    """Train the final model on the full training dataset."""
    x_train = pd.read_parquet(
        DATA_DIR / "x_train.parquet"
    )

    y_train = pd.read_parquet(
        DATA_DIR / "y_train.parquet"
    )[TARGET_COLUMN]

    pipeline = Pipeline(
        steps=[
            (
                "preprocessor",
                build_polynomial_preprocessor(),
            ),
            (
                "model",
                Ridge(alpha=0.1),
            ),
        ]
    )

    pipeline.fit(
        x_train,
        y_train,
    )

    MODEL_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    joblib.dump(
        pipeline,
        MODEL_PATH,
    )

    print("\nFinal Model Training")
    print("====================")

    print(
        f"Training rows: {len(x_train)}"
    )

    print(
        "Polynomial degree: 3"
    )

    print(
        "Ridge alpha: 0.1"
    )

    print(
        f"\nModel saved to:\n{MODEL_PATH}"
    )


if __name__ == "__main__":
    main()