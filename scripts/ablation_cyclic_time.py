"""Evaluate cyclic time features using Ridge regression."""

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
from delivery_eta.features.cyclic_preprocessing import (
    build_cyclic_preprocessor,
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
    """Run cyclic time feature ablation."""
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

    cyclic_rmse, cyclic_std = evaluate_model(
        build_cyclic_preprocessor(),
        x_train,
        y_train,
    )

    improvement = (
        (baseline_rmse - cyclic_rmse)
        / baseline_rmse
        * 100
    )

    print("\nCyclic Time Feature Ablation")
    print("=============================")

    print(
        "Baseline:"
        f" {baseline_rmse:.4f}"
        f" ± {baseline_std:.4f}"
    )

    print(
        "Cyclic features:"
        f" {cyclic_rmse:.4f}"
        f" ± {cyclic_std:.4f}"
    )

    print(
        f"Improvement: {improvement:.2f}%"
    )


if __name__ == "__main__":
    main()