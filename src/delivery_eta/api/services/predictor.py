"""Model inference service."""

from pathlib import Path

import joblib
import pandas as pd

from delivery_eta.data.cleaning import (
    clean_data,
)
from delivery_eta.features.engineering import (
    add_features,
)

from delivery_eta.api.config import Config

MODEL_PATH = Config.MODEL_PATH

class DeliveryETAPredictor:
    """Load and execute the delivery ETA model."""

    def __init__(self, model_path=MODEL_PATH):
        """Initialize the predictor."""
        self.model_path = model_path

        self.model = joblib.load(
            self.model_path
        )

    def predict(self, data):
        """Generate an ETA prediction."""
        dataframe = pd.DataFrame(
            [
                {
                    "Delivery_person_Age": data.get(
                        "delivery_person_age"
                    ),
                    "Delivery_person_Ratings": data.get(
                        "delivery_person_ratings"
                    ),
                    "Restaurant_latitude": data[
                        "restaurant_latitude"
                    ],
                    "Restaurant_longitude": data[
                        "restaurant_longitude"
                    ],
                    "Delivery_location_latitude": data[
                        "delivery_location_latitude"
                    ],
                    "Delivery_location_longitude": data[
                        "delivery_location_longitude"
                    ],
                    "Weather_conditions": data.get(
                        "weather_conditions"
                    ),
                    "Road_traffic_density": data.get(
                        "road_traffic_density"
                    ),
                    "Vehicle_condition": data[
                        "vehicle_condition"
                    ],
                    "Type_of_order": data[
                        "type_of_order"
                    ],
                    "Type_of_vehicle": data[
                        "type_of_vehicle"
                    ],
                    "multiple_deliveries": data.get(
                        "multiple_deliveries"
                    ),
                    "Festival": data.get(
                        "festival"
                    ),
                    "City": data.get(
                        "city"
                    ),
                    "Order_Date": data[
                        "order_date"
                    ],
                    "Time_Orderd": data.get(
                        "time_ordered"
                    ),
                }
            ]
        )

        dataframe = clean_data(
            dataframe
        )

        features = add_features(
            dataframe
        )

        prediction = self.model.predict(
            features
        )[0]

        return float(prediction)