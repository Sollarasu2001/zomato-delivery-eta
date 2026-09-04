"""Cross-validate the Linear Regression baseline."""

from pathlib import Path
import sys

import pandas as pd
from sklearn.pipeline import Pipeline


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from delivery_eta.features.preprocessing import build_preprocessor
from delivery_eta.models.linear import build_linear_regression
from delivery_eta.training.cross_validation import (
    evaluate_with_cross_validation,
)


DATA_DIR = ROOT / "data/interim"


def main():
    """Run five-fold cross-validation."""
    x_train = pd.read_parquet(DATA_DIR / "x_train.parquet")

    y_train = pd.read_parquet(DATA_DIR / "y_train.parquet")["Time_taken(min)"]

    pipeline = Pipeline(
        steps=[
            ("preprocessor",build_preprocessor()),
            ("model",build_linear_regression()),
        ]
    )

    metrics = evaluate_with_cross_validation(pipeline,x_train,y_train)

    print("\nLinear Regression - 5 Fold CV")
    print("==============================")
    print(
        f"MAE:  "
        f"{metrics['mae_mean']:.4f} "
        f"+/- {metrics['mae_std']:.4f}"
    )
    print(
        f"RMSE: "
        f"{metrics['rmse_mean']:.4f} "
        f"+/- {metrics['rmse_std']:.4f}"
    )
    print(
        f"R2:   "
        f"{metrics['r2_mean']:.4f} "
        f"+/- {metrics['r2_std']:.4f}"
    )


if __name__ == "__main__":
    main()