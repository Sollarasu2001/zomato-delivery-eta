"""Prediction routes."""

import logging
import time

from flask import Blueprint
from flask import g
from flask import jsonify
from flask import request

from delivery_eta.api.schemas import (
    DeliveryPredictionSchema,
)
from delivery_eta.api.services.predictor import (
    DeliveryETAPredictor,
)
from delivery_eta.monitoring.data_quality import (
    data_quality_metrics,
)
from delivery_eta.monitoring.events import (
    log_prediction_event,
)
from delivery_eta.monitoring.predictions import (
    prediction_metrics,
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

    data_quality_metrics.record(
        data
    )

    start_time = time.perf_counter()

    prediction = predictor.predict(
        data
    )

    prediction_metrics.record(
        prediction
    )

    latency_ms = (
        time.perf_counter() - start_time
    ) * 1000

    log_prediction_event(
        prediction=prediction,
        latency_ms=latency_ms,
        status="success",
        model_version="polynomial_ridge",
        request_id=g.request_id,
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