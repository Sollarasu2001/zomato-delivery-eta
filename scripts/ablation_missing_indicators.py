"""Evaluate the impact of missing-value indicators."""

from pathlib import Path
import sys

import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.pipeline import Pipeline

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from delivery_eta.features.preprocessing import (
    build_preprocessor,
)
from delivery_eta.features.missing_indicator_preprocessing import (
    build_missing_indicator_preprocessor,
)


DATA_DIR = ROOT / "data/interim"
TARGET_COLUMN = "Time_taken(min)"


def evaluate_model(preprocessor, x_train, y_train):
    """Evaluate Ridge using 5-fold cross-validation."""
    pipeline = Pipeline(
        steps=[
            (
                "preprocessor",
                preprocessor,
            ),
            (
                "model",
                Ridge(alpha=1.0),
            ),
        ]
    )

    cv = KFold(
        n_splits=5,
        shuffle=True,
        random_state=42,
    )

    scores = cross_val_score(
        pipeline,
        x_train,
        y_train,
        scoring="neg_root_mean_squared_error",
        cv=cv,
        n_jobs=-1,
    )

    rmse_scores = -scores

    return (
        rmse_scores.mean(),
        rmse_scores.std(),
    )


def main():
    """Run the missing-indicator ablation."""
    x_train = pd.read_parquet(
        DATA_DIR / "x_train.parquet"
    )

    y_train = pd.read_parquet(
        DATA_DIR / "y_train.parquet"
    )[TARGET_COLUMN]

    baseline_rmse, baseline_std = evaluate_model(
        build_preprocessor(),
        x_train,
        y_train,
    )

    indicator_rmse, indicator_std = evaluate_model(
        build_missing_indicator_preprocessor(),
        x_train,
        y_train,
    )

    improvement = (
        (baseline_rmse - indicator_rmse)
        / baseline_rmse
        * 100
    )

    print("\nMissing-Value Indicator Ablation")
    print("=================================")

    print(
        "Baseline:"
        f" {baseline_rmse:.4f}"
        f" ± {baseline_std:.4f}"
    )

    print(
        "With indicators:"
        f" {indicator_rmse:.4f}"
        f" ± {indicator_std:.4f}"
    )

    print(
        f"Improvement: {improvement:.2f}%"
    )


if __name__ == "__main__":
    main()