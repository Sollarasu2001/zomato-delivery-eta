"""Perform error analysis for the final Polynomial Ridge model."""

from pathlib import Path
import sys

import matplotlib.pyplot as plt
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
from sklearn.pipeline import Pipeline

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from delivery_eta.features.polynomial_preprocessing import (
    build_polynomial_preprocessor,
)


DATA_DIR = ROOT / "data/interim"
OUTPUT_DIR = ROOT / "reports/figures"

TARGET_COLUMN = "Time_taken(min)"


def main():
    """Train final model and analyze prediction errors."""
    x_train = pd.read_parquet(
        DATA_DIR / "x_train.parquet"
    )

    x_test = pd.read_parquet(
        DATA_DIR / "x_test.parquet"
    )

    y_train = pd.read_parquet(
        DATA_DIR / "y_train.parquet"
    )[TARGET_COLUMN]

    y_test = pd.read_parquet(
        DATA_DIR / "y_test.parquet"
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

    predictions = pipeline.predict(
        x_test,
    )

    analysis = x_test.copy()

    analysis["actual"] = y_test.values
    analysis["predicted"] = predictions

    analysis["error"] = (
        analysis["actual"]
        - analysis["predicted"]
    )

    analysis["absolute_error"] = (
        analysis["error"].abs()
    )

    analysis["squared_error"] = (
        analysis["error"] ** 2
    )

    print("\nFinal Model Error Analysis")
    print("===========================")

    mae = mean_absolute_error(
        analysis["actual"],
        analysis["predicted"],
    )

    mse = mean_squared_error(
        analysis["actual"],
        analysis["predicted"],
    )

    rmse = mse ** 0.5

    print(f"MAE:  {mae:.4f}")
    print(f"RMSE: {rmse:.4f}")

    print("\nWorst 20 Predictions")
    print("====================")

    worst_predictions = (
        analysis
        .sort_values(
            "absolute_error",
            ascending=False,
        )
        .head(20)
    )

    columns_to_show = [
        "actual",
        "predicted",
        "error",
        "absolute_error",
        "Road_traffic_density",
        "Weather_conditions",
        "City",
        "delivery_distance_km",
    ]

    print(
        worst_predictions[
            columns_to_show
        ].to_string(index=False)
    )

    print("\nError by Traffic Density")
    print("========================")

    traffic_error = (
        analysis
        .groupby(
            "Road_traffic_density",
            dropna=False,
        )["absolute_error"]
        .agg(
            ["mean", "median", "count"]
        )
        .sort_values(
            "mean",
            ascending=False,
        )
    )

    print(
        traffic_error.to_string()
    )

    print("\nError by Weather")
    print("================")

    weather_error = (
        analysis
        .groupby(
            "Weather_conditions",
            dropna=False,
        )["absolute_error"]
        .agg(
            ["mean", "median", "count"]
        )
        .sort_values(
            "mean",
            ascending=False,
        )
    )

    print(
        weather_error.to_string()
    )

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    plt.figure(
        figsize=(8, 6),
    )

    plt.scatter(
        analysis["actual"],
        analysis["predicted"],
        alpha=0.3,
    )

    minimum = min(
        analysis["actual"].min(),
        analysis["predicted"].min(),
    )

    maximum = max(
        analysis["actual"].max(),
        analysis["predicted"].max(),
    )

    plt.plot(
        [minimum, maximum],
        [minimum, maximum],
    )

    plt.xlabel(
        "Actual Delivery Time (min)"
    )

    plt.ylabel(
        "Predicted Delivery Time (min)"
    )

    plt.title(
        "Actual vs Predicted Delivery Time"
    )

    plt.tight_layout()

    plt.savefig(
        OUTPUT_DIR
        / "actual_vs_predicted.png",
        dpi=200,
    )

    plt.close()

    plt.figure(
        figsize=(8, 6),
    )

    plt.hist(
        analysis["error"],
        bins=40,
    )

    plt.xlabel(
        "Prediction Error (min)"
    )

    plt.ylabel(
        "Frequency"
    )

    plt.title(
        "Prediction Error Distribution"
    )

    plt.tight_layout()

    plt.savefig(
        OUTPUT_DIR
        / "residual_distribution.png",
        dpi=200,
    )

    plt.close()

    print(
        "\nSaved figures to:"
        f" {OUTPUT_DIR}"
    )


if __name__ == "__main__":
    main()