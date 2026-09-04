"""Tune Polynomial Ridge Regression Using Cross-Validation."""

from pathlib import Path
import sys


import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.model_selection import GridSearchCV,KFold
from sklearn.pipeline import Pipeline

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from delivery_eta.features.polynomial_preprocessing import build_polynomial_preprocessor


DATA_DIR = ROOT / "data/interim"

TARGET_COLUMN = "Time_taken(min)"

def main():
    """Tune polynomial degree and Ridge regularization."""
    x_train = pd.read_parquet(DATA_DIR / "x_train.parquet")
    y_train = pd.read_parquet(DATA_DIR / "y_train.parquet")[TARGET_COLUMN]

    pipeline = Pipeline(
        steps=[
            ("preprocessor",build_polynomial_preprocessor()),
            ("model",Ridge())
        ]
    )

    param_grid = {
        "preprocessor__polynomial_numeric__polynomial__degree": [
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

    cv = KFold(n_splits=5,shuffle=True,random_state=42,)

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

    print("\nPolynomial Ridge Hyperparameter Tuning")
    print("=======================================")

    print(
        "Best degree: "
        f"{search.best_params_['preprocessor__polynomial_numeric__polynomial__degree']}"
    )

    print(
        "Best alpha: "
        f"{search.best_params_['model__alpha']}"
    )

    print(
        "Best CV RMSE: "
        f"{-search.best_score_:.4f}"
    )


if __name__ == "__main__":
    main()