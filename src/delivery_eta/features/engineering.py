"""Feature engineering for delivery ETA prediction."""

import numpy as np
import pandas as pd


EARTH_RADIUS_KM = 6371.0


def calculate_haversine_distance(
    latitude_1,
    longitude_1,
    latitude_2,
    longitude_2,
):
    """Calculate great-circle distance between two coordinates."""
    latitude_1 = np.radians(latitude_1)
    longitude_1 = np.radians(longitude_1)
    latitude_2 = np.radians(latitude_2)
    longitude_2 = np.radians(longitude_2)

    delta_latitude = latitude_2 - latitude_1
    delta_longitude = longitude_2 - longitude_1

    a = (
        np.sin(delta_latitude / 2.0) ** 2
        + np.cos(latitude_1)
        * np.cos(latitude_2)
        * np.sin(delta_longitude / 2.0) ** 2
    )

    return (
        EARTH_RADIUS_KM
        * 2.0
        * np.arcsin(np.sqrt(a))
    )


def parse_order_time(series):
    """Normalize HH:MM and fractional-day time representations."""
    values = series.astype("string").str.strip()

    result = pd.Series(
        pd.NaT,
        index=series.index,
        dtype="datetime64[ns]",
    )

    hhmm_mask = values.str.match(
        r"^\d{1,2}:\d{2}$",
        na=False,
    )

    result.loc[hhmm_mask] = pd.to_datetime(
        values.loc[hhmm_mask],
        format="%H:%M",
        errors="coerce",
    )

    numeric_values = pd.to_numeric(
        values.loc[~hhmm_mask],
        errors="coerce",
    )

    numeric_values = numeric_values.mod(1)

    result.loc[~hhmm_mask] = (
        pd.Timestamp("1900-01-01")
        + pd.to_timedelta(
            numeric_values,
            unit="D",
        )
    )

    return result


def add_features(dataframe):
    """Create deterministic features available at order placement."""
    dataframe = dataframe.copy()

    order_date = pd.to_datetime(
        dataframe["Order_Date"],
        format="%d-%m-%Y",
        errors="coerce",
    )

    order_time = parse_order_time(
        dataframe["Time_Orderd"]
    )

    order_datetime = (
        order_date.dt.normalize()
        + (
            order_time
            - pd.Timestamp("1900-01-01")
        )
    )

    dataframe["order_hour"] = order_datetime.dt.hour

    dataframe["order_day_of_week"] = (
        order_datetime.dt.dayofweek
    )

    dataframe["order_month"] = order_datetime.dt.month

    dataframe["is_weekend"] = (
        order_datetime.dt.dayofweek >= 5
    ).astype("float")

    dataframe.loc[
        order_datetime.isna(),
        "is_weekend",
    ] = np.nan

    dataframe["order_hour_sin"] = np.sin(
        2 * np.pi * dataframe["order_hour"] / 24
    )

    dataframe["order_hour_cos"] = np.cos(
        2 * np.pi * dataframe["order_hour"] / 24
    )

    dataframe["order_day_sin"] = np.sin(
        2 * np.pi
        * dataframe["order_day_of_week"]
        / 7
    )

    dataframe["order_day_cos"] = np.cos(
        2 * np.pi
        * dataframe["order_day_of_week"]
        / 7
    )

    dataframe["delivery_distance_km"] = (
        calculate_haversine_distance(
            dataframe["Restaurant_latitude"],
            dataframe["Restaurant_longitude"],
            dataframe["Delivery_location_latitude"],
            dataframe["Delivery_location_longitude"],
        )
    )

    dataframe = dataframe.drop(
        columns=[
            "ID",
            "Delivery_person_ID",
            "Time_Orderd",
            "Time_Order_picked",
            "Order_Date",
            "coordinate_repaired",
        ],
        errors="ignore",
    )

    return dataframe