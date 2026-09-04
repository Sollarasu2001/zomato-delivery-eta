"""Analyze errors from the Linear Regression model."""

from pathlib import Path
import sys

import pandas as pd
from sklearn.pipeline import Pipeline


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from delivery_eta.features.preprocessing import build_preprocessor
from delivery_eta.models.linear import build_linear_regression


DATA_DIR = ROOT / "data/interim"


def load_data():
    """Load train and test datasets."""
    x_train = pd.read_parquet(
        DATA_DIR / "x_train.parquet"
    )

    x_test = pd.read_parquet(
        DATA_DIR / "x_test.parquet"
    )

    y_train = pd.read_parquet(
        DATA_DIR / "y_train.parquet"
    )["Time_taken(min)"]

    y_test = pd.read_parquet(
        DATA_DIR / "y_test.parquet"
    )["Time_taken(min)"]

    return x_train, x_test, y_train, y_test


def main():
    """Train Linear Regression and analyze prediction errors."""
    x_train, x_test, y_train, y_test = load_data()

    pipeline = Pipeline(
        steps=[
            (
                "preprocessor",
                build_preprocessor(),
            ),
            (
                "model",
                build_linear_regression(),
            ),
        ]
    )

    pipeline.fit(
        x_train,
        y_train,
    )

    predictions = pipeline.predict(x_test)

    results = x_test.copy()

    results["actual"] = y_test.to_numpy()
    results["predicted"] = predictions
    results["residual"] = (
        results["actual"]
        - results["predicted"]
    )
    results["absolute_error"] = (
        results["residual"].abs()
    )

    print("\n1. Largest Prediction Errors")
    print("============================")

    columns = [
        "actual",
        "predicted",
        "residual",
        "absolute_error",
        "delivery_distance_km",
        "Weather_conditions",
        "Road_traffic_density",
        "City",
        "order_hour",
    ]

    print(
        results.nlargest(
            10,
            "absolute_error",
        )[columns].to_string(index=False)
    )

    print("\n2. Error by Traffic")
    print("====================")

    traffic_error = (
        results.groupby(
            "Road_traffic_density",
            dropna=False,
        )["absolute_error"]
        .mean()
        .sort_values(ascending=False)
    )

    print(
        traffic_error.to_string()
    )

    print("\n3. Error by Weather")
    print("====================")

    weather_error = (
        results.groupby(
            "Weather_conditions",
            dropna=False,
        )["absolute_error"]
        .mean()
        .sort_values(ascending=False)
    )

    print(
        weather_error.to_string()
    )

    print("\n4. Error by Target Range")
    print("========================")

    results["target_range"] = pd.cut(
        results["actual"],
        bins=[
            0,
            20,
            30,
            40,
            50,
            60,
        ],
        labels=[
            "10-20",
            "21-30",
            "31-40",
            "41-50",
            "51-60",
        ],
    )

    target_error = (
        results.groupby(
            "target_range",
            observed=True,
        )["absolute_error"]
        .agg(
            count="count",
            mae="mean",
        )
    )

    print(
        target_error.to_string()
    )

    print("\n5. Prediction Distribution")
    print("==========================")

    prediction_stats = pd.Series(
        predictions,
        name="predicted",
    ).describe(
        percentiles=[
            0.01,
            0.05,
            0.25,
            0.50,
            0.75,
            0.95,
            0.99,
        ]
    )

    print(
        prediction_stats.to_string()
    )

    print("\n6. Actual Distribution")
    print("======================")

    actual_stats = y_test.describe(
        percentiles=[
            0.01,
            0.05,
            0.25,
            0.50,
            0.75,
            0.95,
            0.99,
        ]
    )

    print(
        actual_stats.to_string()
    )


if __name__ == "__main__":
    main()