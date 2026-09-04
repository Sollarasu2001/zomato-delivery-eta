"""Preprocessing Pipeline For Polynomial Regression."""

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler, PolynomialFeatures

from delivery_eta.models.polynomial import build_polynomial_features

POLYNOMIAL_NUMERIC_COLUMNS = [
    "Delivery_person_Age",
    "Delivery_person_Ratings",
    "Vehicle_condition",
    "multiple_deliveries",
    "order_hour",
    "order_day_of_week",
    "order_month",
    "is_weekend",
    "delivery_distance_km",
]


LINEAR_NUMERIC_COLUMNS = [
    "Restaurant_latitude",
    "Restaurant_longitude",
    "Delivery_location_latitude",
    "Delivery_location_longitude",
]


CATEGORICAL_COLUMNS = [
    "Weather_conditions",
    "Road_traffic_density",
    "Type_of_order",
    "Type_of_vehicle",
    "Festival",
    "City",
]


def build_polynomial_preprocessor(degree=3):
    """Build Preprocessing With Selective Polynomial Expansion."""
    polynomial_pipeline = Pipeline(
        steps =[
            ("imputer",SimpleImputer(strategy="median",)),
            ("polynomial",build_polynomial_features(degree=degree)),
            ("scaler",StandardScaler())
        ]

    )

    linear_numeric_pipeline = Pipeline(
        steps =[
            ("imputer",SimpleImputer(strategy="median",)),
            ("scaler",StandardScaler())
        ]

    )

    categorical_pipeline = Pipeline(
        steps =[
            ("imputer",SimpleImputer(strategy="most_frequent")),
            ("encoder",OneHotEncoder(handle_unknown="ignore",sparse_output=False))
        ]

    )

    return ColumnTransformer(
        transformers=[
            ("polynomial_numeric",polynomial_pipeline,POLYNOMIAL_NUMERIC_COLUMNS),
            ("linear_numeric",linear_numeric_pipeline,LINEAR_NUMERIC_COLUMNS),
            ("categorical",categorical_pipeline,CATEGORICAL_COLUMNS),
        ]
    )