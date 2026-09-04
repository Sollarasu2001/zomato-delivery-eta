"""Compare coordinate-based and distance-based representations."""

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
from delivery_eta.features.distance_preprocessing import (
    build_distance_preprocessor,
)


DATA_DIR = ROOT / "data/interim"
TARGET_COLUMN = "Time_taken(min)"


def evaluate_model(preprocessor, x_train, y_train):
    """Evaluate Ridge using 5-fold CV."""
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
    """Run coordinate versus distance ablation."""
    x_train = pd.read_parquet(
        DATA_DIR / "x_train.parquet"
    )

    y_train = pd.read_parquet(
        DATA_DIR / "y_train.parquet"
    )[TARGET_COLUMN]

    coordinate_rmse, coordinate_std = evaluate_model(
        build_preprocessor(),
        x_train,
        y_train,
    )

    distance_rmse, distance_std = evaluate_model(
        build_distance_preprocessor(),
        x_train,
        y_train,
    )

    print("\nCoordinate vs Distance Ablation")
    print("================================")

    print(
        "Coordinates + distance:"
        f" {coordinate_rmse:.4f}"
        f" ± {coordinate_std:.4f}"
    )

    print(
        "Distance only:"
        f" {distance_rmse:.4f}"
        f" ± {distance_std:.4f}"
    )

    improvement = (
        (coordinate_rmse - distance_rmse)
        / coordinate_rmse
        * 100
    )

    print(
        f"Distance-only change:"
        f" {improvement:.2f}%"
    )


if __name__ == "__main__":
    main()