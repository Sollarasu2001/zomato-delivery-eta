"""Cleaning utilities for the delivery ETA dataset."""

from pathlib import Path

import pandas as pd


AGE_MIN = 18
AGE_MAX = 50

RATING_MIN = 1
RATING_MAX = 5

COORDINATE_TOLERANCE_DEGREES = 0.15


def clean_data(dataframe):
    """Apply deterministic cleaning and coordinate repairs."""
    dataframe = dataframe.copy()

    if dataframe.empty:
        return dataframe

    dataframe = dataframe.drop_duplicates()

    if "Delivery_person_Age" in dataframe.columns:
        dataframe.loc[
            ~dataframe["Delivery_person_Age"].between(
                AGE_MIN,
                AGE_MAX,
                inclusive="both",
            ),
            "Delivery_person_Age",
        ] = pd.NA

    if "Delivery_person_Ratings" in dataframe.columns:
        dataframe.loc[
            ~dataframe["Delivery_person_Ratings"].between(
                RATING_MIN,
                RATING_MAX,
                inclusive="both",
            ),
            "Delivery_person_Ratings",
        ] = pd.NA

    required_coordinate_columns = {
        "Restaurant_latitude",
        "Restaurant_longitude",
        "Delivery_location_latitude",
        "Delivery_location_longitude",
    }

    if not required_coordinate_columns.issubset(
        dataframe.columns
    ):
        dataframe["coordinate_repaired"] = False
        return dataframe

    dataframe["coordinate_repaired"] = False

    restaurant_latitude = dataframe[
        "Restaurant_latitude"
    ]

    restaurant_longitude = dataframe[
        "Restaurant_longitude"
    ]

    delivery_latitude = dataframe[
        "Delivery_location_latitude"
    ]

    delivery_longitude = dataframe[
        "Delivery_location_longitude"
    ]

    latitude_match = (
        (
            restaurant_latitude.abs()
            - delivery_latitude
        ).abs()
        <= COORDINATE_TOLERANCE_DEGREES
    )

    longitude_match = (
        (
            restaurant_longitude.abs()
            - delivery_longitude
        ).abs()
        <= COORDINATE_TOLERANCE_DEGREES
    )

    sign_error = (
        (
            (restaurant_latitude < 0)
            & (delivery_latitude > 0)
        )
        |
        (
            (restaurant_longitude < 0)
            & (delivery_longitude > 0)
        )
    )

    repair_mask = (
        sign_error
        & latitude_match
        & longitude_match
    )

    dataframe.loc[
        repair_mask,
        "Restaurant_latitude",
    ] = (
        dataframe.loc[
            repair_mask,
            "Restaurant_latitude",
        ].abs()
    )

    dataframe.loc[
        repair_mask,
        "Restaurant_longitude",
    ] = (
        dataframe.loc[
            repair_mask,
            "Restaurant_longitude",
        ].abs()
    )

    dataframe.loc[
        repair_mask,
        "coordinate_repaired",
    ] = True

    return dataframe


def save_clean_data(dataframe, output_path):
    """Persist cleaned data as Parquet."""
    output_path = Path(output_path)

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    dataframe.to_parquet(
        output_path,
        index=False,
    )