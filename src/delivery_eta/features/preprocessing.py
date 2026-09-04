
"""Preprocessing pipeline for ETA regression."""

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import StandardScaler


NUMERIC_COLUMNS = [
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
    "delivery_distance_km",
]


CATEGORICAL_COLUMNS = [
    "Weather_conditions",
    "Road_traffic_density",
    "Type_of_order",
    "Type_of_vehicle",
    "Festival",
    "City",
]


def build_preprocessor():
    """Build the leakage-safe preprocessing pipeline."""
    numeric_pipeline = Pipeline(
        steps=[
            ("imputer",SimpleImputer(strategy="median",)),
            ("scaler",StandardScaler()),
        ]
    )

    categorical_pipeline = Pipeline(
        steps=[
            ("imputer",SimpleImputer(strategy="most_frequent")),
            ("encoder",OneHotEncoder(handle_unknown="ignore",sparse_output=False)),
        ]
    )

    return ColumnTransformer(
        transformers=[
            ("numeric",numeric_pipeline,NUMERIC_COLUMNS),
            ("categorical",categorical_pipeline,CATEGORICAL_COLUMNS),
        ]
    )