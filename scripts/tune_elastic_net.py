"""Tune Ridge Regression Using Cross-Validation."""

from pathlib import Path
import sys

import pandas as pd
from sklearn.model_selection import KFold
from sklearn.pipeline import Pipeline


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from delivery_eta.features.preprocessing import build_preprocessor
from delivery_eta.models.elastic_net import build_elastic_net
from delivery_eta.training.tuning import tune_model


DATA_DIR = ROOT / "data/interim"
TARGET_COLUMN = "Time_taken(min)"


def main():
    """Run Ridge hyperparameter Tuning."""
    x_train = pd.read_parquet(DATA_DIR/"x_train.parquet")
    y_train = pd.read_parquet(DATA_DIR/"y_train.parquet")[TARGET_COLUMN]

    pipeline = Pipeline(
        steps =[
            ("preprocessor", build_preprocessor()),
            ("model", build_elastic_net())
        ]
    )

    param_grid = {
        "model__alpha": [
            0.0001,
            0.001,
            0.01,
            0.05,
            0.1,
            0.5,
            1.0,
        ],
        "model__l1_ratio": [
            0.1,
            0.25,
            0.5,
            0.75,
            0.9,
        ],
    }

    cv = KFold(n_splits=5, shuffle=True, random_state=42)
    search = tune_model(pipeline=pipeline, param_grid=param_grid, features=x_train, target=y_train, cv=cv)

    print("\nElasticNet Hyperparameter Tuning")
    print("=================================")
    print(
        f"Best alpha: "
        f"{search.best_params_['model__alpha']}"
    )
    print(
        f"Best l1_ratio: "
        f"{search.best_params_['model__l1_ratio']}"
    )
    print(
        f"Best CV RMSE: "
        f"{-search.best_score_:.4f}"
    )


if __name__ == "__main__":
    main()