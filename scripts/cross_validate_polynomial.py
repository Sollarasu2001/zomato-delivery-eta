"""Cross-validate Polynomial Ridge regression."""

from pathlib import Path
import sys

import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_validate
from sklearn.pipeline import Pipeline


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from delivery_eta.features.polynomial_preprocessing import build_polynomial_preprocessor


DATA_DIR = ROOT / "data/interim"

TARGET_COLUMN = "Time_taken(min)"


def main():
    """Run five-fold cross-validation."""
    x_train = pd.read_parquet(DATA_DIR / "x_train.parquet")

    y_train = pd.read_parquet(DATA_DIR / "y_train.parquet")[TARGET_COLUMN]

    pipeline = Pipeline(
        steps=[
            ("preprocessor",build_polynomial_preprocessor(degree=2)),
            ("model",Ridge(alpha=1.0))
        ]
    )

    cv = KFold(n_splits=5,shuffle=True,random_state=42)

    results = cross_validate(
        pipeline,
        x_train,
        y_train,
        cv=cv,
        scoring={
            "mae": "neg_mean_absolute_error",
            "rmse": "neg_root_mean_squared_error",
            "r2": "r2",
        },
        n_jobs=-1,
    )

    mae = -results["test_mae"]
    rmse = -results["test_rmse"]
    r2 = results["test_r2"]

    print("\nPolynomial Ridge - 5 Fold CV")
    print("============================")
    print(
        f"MAE:  {mae.mean():.4f} "
        f"+/- {mae.std():.4f}"
    )
    print(
        f"RMSE: {rmse.mean():.4f} "
        f"+/- {rmse.std():.4f}"
    )
    print(
        f"R²:   {r2.mean():.4f} "
        f"+/- {r2.std():.4f}"
    )


if __name__ == "__main__":
    main()