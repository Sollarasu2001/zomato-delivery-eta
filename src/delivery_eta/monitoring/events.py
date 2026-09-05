"""Structured Prediction Event Logging."""

import json
import logging
from datetime import datetime, timezone
from uuid import uuid4


LOGGER = logging.getLogger("delivery_eta.monitoring")


def log_prediction_event(
    *,
    prediction: float,
    latency_ms: float,
    status: str,
    model_version: str,
    request_id: str | None = None,
) -> str:
    """Log a structured prediction event."""
    event_request_id = request_id or str(uuid4())

    event = {
        "event": "prediction",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "request_id": event_request_id,
        "model_version": model_version,
        "prediction": round(float(prediction), 4),
        "latency_ms": round(float(latency_ms), 2),
        "status": status,
    }

    LOGGER.info(json.dumps(event))

    return event_request_id