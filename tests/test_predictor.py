# """Tests for the predictor service."""

# from delivery_eta.api.services.predictor import (
#     DeliveryETAPredictor,
# )


# VALID_DATA = {
#     "delivery_person_age": 30,
#     "delivery_person_ratings": 4.8,
#     "restaurant_latitude": 12.9716,
#     "restaurant_longitude": 77.5946,
#     "delivery_location_latitude": 12.9352,
#     "delivery_location_longitude": 77.6245,
#     "weather_conditions": "Sunny",
#     "road_traffic_density": "High",
#     "vehicle_condition": 2,
#     "type_of_order": "Meal",
#     "type_of_vehicle": "Motorcycle",
#     "multiple_deliveries": 1,
#     "festival": "No",
#     "city": "Metropolitian",
#     "order_date": "15-08-2022",
#     "time_ordered": "19:30",
# }


# def test_predictor_returns_float():
#     """Test predictor output type."""
#     predictor = DeliveryETAPredictor()

#     prediction = predictor.predict(VALID_DATA)

#     assert isinstance(prediction,float)
    
#     assert prediction > 0


####VERSIOn 2
"""Tests for the predictor service."""

from pathlib import Path

import pytest

from delivery_eta.api.services.predictor import (
    DeliveryETAPredictor,
)


def test_predictor_missing_model():
    """Test missing model artifact handling."""
    predictor = DeliveryETAPredictor(
        model_path=(
            Path("nonexistent")
            / "model.joblib"
        )
    )

    with pytest.raises(
        FileNotFoundError,
        match="Model not found",
    ):
        predictor.predict(
            {
                "delivery_person_age": 30,
            }
        )