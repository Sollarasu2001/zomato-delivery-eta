"""Train and track the final model with MLflow."""

from pathlib import Path
import sys

import mlflow
import mlflow.sklearn
from mlflow.models import infer_signature
import pandas as pd

from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.pipeline import Pipeline

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from delivery_eta.features.polynomial_preprocessing import (
    build_polynomial_preprocessor,
)


DATA_DIR = ROOT / "data/interim"
TARGET_COLUMN = "Time_taken(min)"

EXPERIMENT_NAME = "zomato-delivery-eta"


def main():
    """Train, evaluate, and track the final model."""
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

    cv = KFold(
        n_splits=5,
        shuffle=True,
        random_state=42,
    )

    cv_scores = cross_val_score(
        pipeline,
        x_train,
        y_train,
        scoring="neg_root_mean_squared_error",
        cv=cv,
        n_jobs=-1,
    )

    cv_rmse = -cv_scores

    pipeline.fit(
        x_train,
        y_train,
    )

    predictions = pipeline.predict(
        x_test,
    )

    input_example = x_train.head(10).copy()

    numeric_columns = [
        "Delivery_person_Age",
        "Delivery_person_Ratings",
        "Restaurant_latitude",
        "Restaurant_longitude",
        "Delivery_location_latitude",
        "Delivery_location_longitude",
        "Vehicle_condition",
        "multiple_deliveries",
        "order_hour",
        "order_day_of_week",
        "order_month",
        "is_weekend",
        "order_hour_sin",
        "order_hour_cos",
        "order_day_sin",
        "order_day_cos",
        "delivery_distance_km",
    ]

    input_example[numeric_columns] = (
        input_example[numeric_columns]
        .astype("float64")
    )

    example_predictions = pipeline.predict(
        input_example,
    )

    signature = infer_signature(
        input_example,
        example_predictions,
    )

    mae = mean_absolute_error(
        y_test,
        predictions,
    )

    mse = mean_squared_error(
        y_test,
        predictions,
    )

    rmse = mse ** 0.5

    r2 = r2_score(
        y_test,
        predictions,
    )

    mlflow.set_experiment(
        EXPERIMENT_NAME
    )

    with mlflow.start_run():
        mlflow.log_param(
            "model_type",
            "Polynomial Ridge",
        )

        mlflow.log_param(
            "polynomial_degree",
            3,
        )

        mlflow.log_param(
            "ridge_alpha",
            0.1,
        )

        mlflow.log_param(
            "cv_folds",
            5,
        )

        mlflow.log_param(
            "random_state",
            42,
        )

        mlflow.log_metric(
            "cv_rmse_mean",
            cv_rmse.mean(),
        )

        mlflow.log_metric(
            "cv_rmse_std",
            cv_rmse.std(),
        )

        mlflow.log_metric(
            "test_mae",
            mae,
        )

        mlflow.log_metric(
            "test_rmse",
            rmse,
        )

        mlflow.log_metric(
            "test_r2",
            r2,
        )


        mlflow.sklearn.log_model(
            pipeline,
            name="polynomial_ridge",
            signature=signature,
            input_example=input_example,
            skops_trusted_types=[
                "numpy.dtype",
            ],
        )

        print("\nMLflow Run")
        print("===========")

        print(
            f"CV RMSE: "
            f"{cv_rmse.mean():.4f}"
            f" ± {cv_rmse.std():.4f}"
        )

        print(
            f"Test MAE: {mae:.4f}"
        )

        print(
            f"Test RMSE: {rmse:.4f}"
        )

        print(
            f"Test R²: {r2:.4f}"
        )

        print(
            "\nMLflow tracking completed."
        )


if __name__ == "__main__":
    main()