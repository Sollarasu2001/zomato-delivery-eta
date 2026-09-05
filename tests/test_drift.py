"""Tests for feature drift monitoring."""

from delivery_eta.monitoring.drift import (
    DriftMonitor,
)


def test_drift_monitor_starts_without_data():
    """Test empty drift monitor state."""
    monitor = DriftMonitor(
        max_observations=100,
    )

    result = monitor.snapshot()

    assert result["drift_status"] == (
        "insufficient_data"
    )
    assert result["drift_observations"] == 0
    assert result["feature_psi"] == {}


def test_drift_monitor_records_observations():
    """Test production observations are recorded."""
    monitor = DriftMonitor(
        max_observations=100,
    )

    observation = {
        "Delivery_person_Age": 30,
        "Delivery_person_Ratings": 4.8,
        "Vehicle_condition": 2,
        "multiple_deliveries": 1,
        "order_hour": 19,
        "order_day_of_week": 1,
        "order_month": 8,
        "is_weekend": 0,
        "delivery_distance_km": 8.5,
        "Weather_conditions": "Sunny",
        "Road_traffic_density": "High",
        "Type_of_order": "Meal",
        "Type_of_vehicle": "motorcycle",
        "Festival": "No",
        "City": "Metropolitian",
    }

    monitor.record(
        observation
    )

    result = monitor.snapshot()

    assert result["drift_observations"] == 1
    assert result["feature_psi"]


def test_drift_monitor_limits_observations():
    """Test the observation buffer is bounded."""
    monitor = DriftMonitor(
        max_observations=2,
    )

    observation = {
        "Delivery_person_Age": 30,
    }

    monitor.record(
        observation
    )
    monitor.record(
        observation
    )
    monitor.record(
        observation
    )

    result = monitor.snapshot()

    assert result["drift_observations"] == 2


def test_drift_status_stable():
    """Test stable drift classification."""
    result = DriftMonitor._overall_status(
        {
            "feature_a": 0.05,
            "feature_b": 0.08,
        }
    )

    assert result == "stable"


def test_drift_status_warning():
    """Test warning drift classification."""
    result = DriftMonitor._overall_status(
        {
            "feature_a": 0.05,
            "feature_b": 0.15,
        }
    )

    assert result == "warning"


def test_drift_status_drift():
    """Test drift classification."""
    result = DriftMonitor._overall_status(
        {
            "feature_a": 0.05,
            "feature_b": 0.25,
        }
    )

    assert result == "drift"


def test_numeric_reference_distribution_is_exact():
    """Test numeric reference distribution is normalized."""
    monitor = DriftMonitor()

    for profile in (
        monitor.reference_profile[
            "numeric"
        ].values()
    ):
        distribution = profile[
            "reference_distribution"
        ]

        assert distribution
        assert abs(
            sum(distribution) - 1.0
        ) < 1e-6