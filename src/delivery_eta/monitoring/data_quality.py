"""Input data quality monitoring."""

import math
import threading


MONITORED_FIELDS = (
    "delivery_person_age",
    "delivery_person_ratings",
    "weather_conditions",
    "road_traffic_density",
    "multiple_deliveries",
    "festival",
    "city",
    "time_ordered",
)

MAX_EXPECTED_DISTANCE_KM = 50.0


class DataQualityMetrics:
    """Thread-safe input data quality collector."""

    def __init__(self):
        """Initialize data quality metrics."""
        self._lock = threading.Lock()
        self._requests_checked = 0
        self._requests_with_missing_values = 0
        self._missing_value_count = 0
        self._distance_over_threshold = 0
        self._missing_fields = {
            field: 0
            for field in MONITORED_FIELDS
        }

    def record(self, data: dict) -> None:
        """Record quality metrics for an accepted request."""
        missing_count = 0

        for field in MONITORED_FIELDS:
            value = data.get(field)

            if _is_missing(value):
                missing_count += 1
                self._missing_fields[field] += 1

        distance_km = _calculate_distance_km(data)

        with self._lock:
            self._requests_checked += 1
            self._missing_value_count += missing_count

            if missing_count > 0:
                self._requests_with_missing_values += 1

            if (
                distance_km is not None
                and distance_km > MAX_EXPECTED_DISTANCE_KM
            ):
                self._distance_over_threshold += 1

    def snapshot(self) -> dict:
        """Return a data quality snapshot."""
        with self._lock:
            requests = self._requests_checked

            return {
                "requests_checked": requests,
                "requests_with_missing_values": (
                    self._requests_with_missing_values
                ),
                "missing_value_count": (
                    self._missing_value_count
                ),
                "missing_value_rate": (
                    (
                        self._missing_value_count
                        / (
                            requests
                            * len(MONITORED_FIELDS)
                        )
                    )
                    if requests
                    else 0.0
                ),
                "missing_fields": dict(
                    self._missing_fields
                ),
                "distance_over_50km": (
                    self._distance_over_threshold
                ),
            }


def _is_missing(value) -> bool:
    """Return whether a value should be treated as missing."""
    return value is None or (
        isinstance(value, float)
        and math.isnan(value)
    )


def _calculate_distance_km(data: dict):
    """Calculate straight-line distance using Haversine."""
    latitude_1 = data.get(
        "restaurant_latitude"
    )
    longitude_1 = data.get(
        "restaurant_longitude"
    )
    latitude_2 = data.get(
        "delivery_location_latitude"
    )
    longitude_2 = data.get(
        "delivery_location_longitude"
    )

    if any(
        _is_missing(value)
        for value in (
            latitude_1,
            longitude_1,
            latitude_2,
            longitude_2,
        )
    ):
        return None

    radius_km = 6371.0

    lat1 = math.radians(latitude_1)
    lat2 = math.radians(latitude_2)

    delta_lat = math.radians(
        latitude_2 - latitude_1
    )
    delta_lon = math.radians(
        longitude_2 - longitude_1
    )

    value = (
        math.sin(delta_lat / 2) ** 2
        + math.cos(lat1)
        * math.cos(lat2)
        * math.sin(delta_lon / 2) ** 2
    )

    return 2 * radius_km * math.asin(
        math.sqrt(value)
    )


data_quality_metrics = DataQualityMetrics()