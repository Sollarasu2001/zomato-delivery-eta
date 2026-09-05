"""Request correlation and metrics utilities."""

import time
from uuid import uuid4

from flask import g
from flask import request

from delivery_eta.monitoring.metrics import metrics


REQUEST_ID_HEADER = "X-Request-ID"


def register_request_context(app):
    """Register request ID and metrics hooks."""

    @app.before_request
    def set_request_context():
        """Set request ID and request start time."""
        request_id = request.headers.get(
            REQUEST_ID_HEADER
        )

        g.request_id = request_id or str(uuid4())
        g.request_start_time = time.perf_counter()

    @app.after_request
    def record_request_metrics(response):
        """Record request metrics and add request ID."""
        latency_ms = (
            time.perf_counter()
            - g.request_start_time
        ) * 1000

        metrics.record_request(
            status_code=response.status_code,
            latency_ms=latency_ms,
        )

        response.headers[REQUEST_ID_HEADER] = (
            g.request_id
        )

        return response