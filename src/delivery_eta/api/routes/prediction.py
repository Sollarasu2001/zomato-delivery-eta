"""Prediction routes."""

import logging

from flask import Blueprint
from flask import jsonify
from flask import request

from delivery_eta.api.schemas import (
    DeliveryPredictionSchema,
)
from delivery_eta.api.services.predictor import (
    DeliveryETAPredictor,
)


logger = logging.getLogger(
    __name__
)

prediction_bp = Blueprint(
    "prediction",
    __name__,
)

predictor = DeliveryETAPredictor()
prediction_schema = DeliveryPredictionSchema()


@prediction_bp.route(
    "/predict",
    methods=["POST"],
)
def predict():
    """Predict delivery ETA."""
    if not request.is_json:
        logger.warning(
            "prediction_invalid_content_type"
        )

        return jsonify(
            {
                "error": (
                    "Content-Type must be "
                    "application/json."
                )
            }
        ), 400

    payload = request.get_json(
        silent=True
    )

    if payload is None:
        logger.warning(
            "prediction_invalid_json"
        )

        return jsonify(
            {
                "error": (
                    "Request body must contain "
                    "valid JSON."
                )
            }
        ), 400

    data = prediction_schema.load(
        payload
    )

    prediction = predictor.predict(
        data
    )

    logger.info(
        "prediction_success"
    )

    return jsonify(
        {
            "predicted_delivery_time_minutes": (
                round(
                    prediction,
                    2,
                )
            )
        }
    )