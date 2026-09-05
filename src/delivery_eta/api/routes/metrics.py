"""Metrics routes."""

from flask import Blueprint
from flask import jsonify

from delivery_eta.monitoring.data_quality import (
    data_quality_metrics,
)
from delivery_eta.monitoring.metrics import (
    metrics,
)
from delivery_eta.monitoring.predictions import (
    prediction_metrics,
)
from delivery_eta.monitoring.drift import (
    drift_monitor,
)


metrics_bp = Blueprint(
    "metrics",
    __name__,
)


@metrics_bp.route(
    "/metrics",
    methods=["GET"],
)
def get_metrics():
    """Return application metrics."""
    response = metrics.snapshot()

    response.update(
        prediction_metrics.snapshot()
    )

    response.update(
        data_quality_metrics.snapshot()
    )
    
    response.update(
        drift_monitor.snapshot()
    )

    return jsonify(response)