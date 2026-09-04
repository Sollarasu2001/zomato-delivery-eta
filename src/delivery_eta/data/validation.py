# """ Data validation utilities for the ETA prediction pipeline. """

# from pathlib import Path

# import pandas as pd

# REQUIRED_COLUMNS = {
#     "ID",
#     "Delivery_person_ID",
#     "Delivery_person_Age",
#     "Delivery_person_Ratings",
#     "Restaurant_latitude",
#     "Restaurant_longitude",
#     "Delivery_location_latitude",
#     "Delivery_location_longitude",
#     "Order_Date",
#     "Time_Orderd",
#     "Time_Order_picked",
#     "Weather_conditions",
#     "Road_traffic_density",
#     "Vehicle_condition",
#     "Type_of_order",
#     "Type_of_vehicle",
#     "multiple_deliveries",
#     "Festival",
#     "City",
#     "Time_taken(min)",
# }


# def validate_required_columns(dataframe):
#     """Validate That All Required Dataset Columns Exists:"""
#     missing_columns = REQUIRED_COLUMNS.difference(dataframe.columns)

#     if missing_columns:
#         raise ValueError(f"Missing Required Columns: {sorted(missing_columns)}")
    

# def validate_numeric_ranges(dataframe):
#     """Validate Know Numeric Business Constraints."""
#     violations = {}

#     age_mask = ~dataframe["Delivery_person_Age"].between(18,65, inclusive="both") & dataframe["Delivery_person_Age"].notna()

#     rating_mask = ~dataframe["Delivery_person_Ratings"].between(0,5,inclusive="both") & dataframe["Delivery_person_Ratings"].notna()

#     vehicle_condition_mask = ~dataframe["Vehicle_condition"].between(1,5,inclusive="both") & dataframe["Vehicle_condition"].notna()

#     delivery_time_mask = (
#         dataframe["Time_taken(min)"] <= 0
#     ) & dataframe["Time_taken(min)"].notna()

#     violations["invalid_age"] = int(age_mask.sum())
#     violations["invalid_rating"] = int(rating_mask.sum())
#     violations["invalid_vehicle_condition"] = int(vehicle_condition_mask.sum())
#     violations["invalid_delivery_time"] = int(delivery_time_mask.sum())

#     return violations


# def validate_coordinates(dataframe):
#     """Validate Latitude and Longitdue Ranges."""
#     latitude_columns = ["Restaurant_latitude", "Delivery_location_latitude"]
#     longitude_columns = ["Restaurant_longitude", "Delivery_location_longitude"]

#     violations = {}

#     for column in latitude_columns:
#         mask = ~dataframe[column].between(-90,90, inclusive="both") & dataframe[column].notna()
#         violations[f"invalid_{column}"] = int(mask.sum())

#     for column in longitude_columns:
#         mask = ~dataframe[column].between(-180,180, inclusive="both") & dataframe[column].notna()
#         violations[f"invalid_{column}"] = int(mask.sum())

#     return violations


# def validate_dataframe(dataframe):
#     """Run dataset-level validation checks."""
#     validate_required_columns(dataframe)

#     numeric_violations = validate_numeric_ranges(dataframe)
#     coordinate_violations = validate_coordinates(dataframe)

#     return {
#         "rows": len(dataframe),
#         "columns": len(dataframe.columns),
#         "duplicate_rows": int(dataframe.duplicated().sum()),
#         "numeric_violations": numeric_violations,
#         "coordinate_violations": coordinate_violations,
#     }



####VERSION 2
"""Validation utilities for delivery ETA data."""

import pandas as pd


REQUIRED_COLUMNS = {
    "ID",
    "Delivery_person_ID",
    "Delivery_person_Age",
    "Delivery_person_Ratings",
    "Restaurant_latitude",
    "Restaurant_longitude",
    "Delivery_location_latitude",
    "Delivery_location_longitude",
    "Order_Date",
    "Time_Orderd",
    "Time_Order_picked",
    "Weather_conditions",
    "Road_traffic_density",
    "Vehicle_condition",
    "Type_of_order",
    "Type_of_vehicle",
    "multiple_deliveries",
    "Festival",
    "City",
    "Time_taken(min)",
}


def validate_required_columns(dataframe):
    """Validate that required columns exist."""
    missing_columns = REQUIRED_COLUMNS.difference(dataframe.columns)

    if missing_columns:
        raise ValueError(f"Missing required columns: {sorted(missing_columns)}")


def validate_numeric_ranges(dataframe):
    """Validate known numeric constraints."""
    violations = {}

    age = dataframe["Delivery_person_Age"]
    rating = dataframe["Delivery_person_Ratings"]
    delivery_time = dataframe["Time_taken(min)"]

    violations["invalid_age"] = int(
        (
            ~age.between(18, 80)
            & age.notna()
        ).sum()
    )

    violations["invalid_rating"] = int(
        (
            ~rating.between(0, 5)
            & rating.notna()
        ).sum()
    )

    violations["invalid_delivery_time"] = int(
        (
            (delivery_time <= 0)
            & delivery_time.notna()
        ).sum()
    )

    return violations


def validate_coordinates(dataframe):
    """Validate geographic coordinate ranges."""
    latitude_columns = [
        "Restaurant_latitude",
        "Delivery_location_latitude",
    ]

    longitude_columns = [
        "Restaurant_longitude",
        "Delivery_location_longitude",
    ]

    violations = {}

    for column in latitude_columns:
        violations[f"invalid_{column}"] = int(
            (
                ~dataframe[column].between(-90, 90)
                & dataframe[column].notna()
            ).sum()
        )

    for column in longitude_columns:
        violations[f"invalid_{column}"] = int(
            (
                ~dataframe[column].between(-180, 180)
                & dataframe[column].notna()
            ).sum()
        )

    return violations


def validate_dataframe(dataframe):
    """Run all dataset-level validation checks."""
    validate_required_columns(dataframe)

    return {
        "rows": len(dataframe),
        "columns": len(dataframe.columns),
        "duplicate_rows": int(dataframe.duplicated().sum()),
        "numeric_violations": validate_numeric_ranges(dataframe),
        "coordinate_violations": validate_coordinates(dataframe),
    }