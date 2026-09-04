"""Evaluate the final Polynomial Ridge regression model."""

from pathlib import Path
import sys

import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.pipeline import Pipeline

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from delivery_eta.features.polynomial_preprocessing import (
    build_polynomial_preprocessor,
)


DATA_DIR = ROOT / "data/interim"
TARGET_COLUMN = "Time_taken(min)"


def main():
    """Train and evaluate the final Polynomial Ridge model."""
    x_train = pd.read_parquet(
        DATA_DIR / "x_train.parquet"
    )

    x_test = pd.read_parquet(
        DATA_DIR / "x_test.parquet"
    )

    y_train = pd.read_parquet(
        DATA_DIR / "y_train.parquet"
    )[TARGET_COLUMN]

    y_test = pd.read_parquet(
        DATA_DIR / "y_test.parquet"
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

    cv = KFold(
        n_splits=5,
        shuffle=True,
        random_state=42,
    )

    cv_scores = cross_val_score(
        pipeline,
        x_train,
        y_train,
        scoring="neg_root_mean_squared_error",
        cv=cv,
        n_jobs=-1,
    )

    cv_rmse = -cv_scores

    pipeline.fit(
        x_train,
        y_train,
    )

    predictions = pipeline.predict(
        x_test,
    )

    mae = mean_absolute_error(
        y_test,
        predictions,
    )

    mse = mean_squared_error(
        y_test,
        predictions,
    )

    rmse = mse ** 0.5

    r2 = r2_score(
        y_test,
        predictions,
    )

    print("\nFinal Polynomial Ridge Evaluation")
    print("==================================")

    print("\nConfiguration:")
    print("  Polynomial degree: 3")
    print("  Ridge alpha: 0.1")

    print("\nCross-Validation:")
    print(
        f"  RMSE: {cv_rmse.mean():.4f}"
        f" ± {cv_rmse.std():.4f}"
    )

    print("\nTest Set:")
    print(f"  MAE:  {mae:.4f}")
    print(f"  RMSE: {rmse:.4f}")
    print(f"  R²:   {r2:.4f}")


if __name__ == "__main__":
    main()