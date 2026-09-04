"""Interpret coefficients of the final Polynomial Ridge model."""

from pathlib import Path
import sys

import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from delivery_eta.features.polynomial_preprocessing import (
    build_polynomial_preprocessor,
)


DATA_DIR = ROOT / "data/interim"
TARGET_COLUMN = "Time_taken(min)"


def main():
    """Train the final model and inspect transformed coefficients."""
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

    preprocessor = pipeline.named_steps[
        "preprocessor"
    ]

    model = pipeline.named_steps[
        "model"
    ]

    feature_names = (
        preprocessor.get_feature_names_out()
    )

    coefficients = model.coef_

    coefficient_table = pd.DataFrame(
        {
            "feature": feature_names,
            "coefficient": coefficients,
        }
    )

    coefficient_table["absolute_coefficient"] = (
        coefficient_table["coefficient"].abs()
    )

    coefficient_table = coefficient_table.sort_values(
        "absolute_coefficient",
        ascending=False,
    )

    print("\nFinal Polynomial Ridge Interpretation")
    print("=======================================")

    print("\nTotal transformed features:")
    print(len(coefficient_table))

    print("\nTop 20 by absolute coefficient")
    print("================================")

    print(
        coefficient_table.head(20).to_string(
            index=False
        )
    )

    print("\nTop 10 Positive Coefficients")
    print("=============================")

    positive = (
        coefficient_table
        .sort_values(
            "coefficient",
            ascending=False,
        )
        .head(10)
    )

    print(
        positive[
            [
                "feature",
                "coefficient",
            ]
        ].to_string(
            index=False
        )
    )

    print("\nTop 10 Negative Coefficients")
    print("=============================")

    negative = (
        coefficient_table
        .sort_values(
            "coefficient",
            ascending=True,
        )
        .head(10)
    )

    print(
        negative[
            [
                "feature",
                "coefficient",
            ]
        ].to_string(
            index=False
        )
    )

    print("\nZero / Near-Zero Coefficients")
    print("==============================")

    near_zero = (
        coefficient_table[
            coefficient_table[
                "absolute_coefficient"
            ] < 1e-6
        ]
    )

    print(
        f"Count: {len(near_zero)}"
    )


if __name__ == "__main__":
    main()