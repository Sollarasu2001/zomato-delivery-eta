"""Tune the final Polynomial Ridge candidate."""

from pathlib import Path
import sys

import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import KFold
from sklearn.pipeline import Pipeline

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from delivery_eta.features.final_preprocessing import (
    build_final_preprocessor,
)


DATA_DIR = ROOT / "data/interim"
TARGET_COLUMN = "Time_taken(min)"


def main():
    """Tune polynomial degree and Ridge regularization."""
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
                build_final_preprocessor(),
            ),
            (
                "model",
                Ridge(),
            ),
        ]
    )

    param_grid = {
        "preprocessor__"
        "polynomial_numeric__"
        "polynomial__degree": [
            2,
            3,
        ],
        "model__alpha": [
            0.01,
            0.1,
            1.0,
            10.0,
            100.0,
        ],
    }

    cv = KFold(
        n_splits=5,
        shuffle=True,
        random_state=42,
    )

    search = GridSearchCV(
        estimator=pipeline,
        param_grid=param_grid,
        scoring="neg_root_mean_squared_error",
        cv=cv,
        n_jobs=-1,
        return_train_score=True,
    )

    search.fit(
        x_train,
        y_train,
    )

    best_degree = search.best_params_[
        "preprocessor__"
        "polynomial_numeric__"
        "polynomial__degree"
    ]

    best_alpha = search.best_params_[
        "model__alpha"
    ]

    print("\nFinal Polynomial Ridge Tuning")
    print("==============================")

    print(
        f"Best degree: {best_degree}"
    )

    print(
        f"Best alpha: {best_alpha}"
    )

    print(
        f"Best CV RMSE: {-search.best_score_:.4f}"
    )


if __name__ == "__main__":
    main()