"""Request correlation utilities."""

from uuid import uuid4

from flask import g
from flask import request


REQUEST_ID_HEADER = "X-Request-ID"


def register_request_id(app):
    """Register request ID hooks on the Flask application."""

    @app.before_request
    def set_request_id():
        """Set or generate a request ID."""
        request_id = request.headers.get(
            REQUEST_ID_HEADER
        )

        g.request_id = request_id or str(uuid4())

    @app.after_request
    def add_request_id(response):
        """Add the request ID to the response."""
        response.headers[REQUEST_ID_HEADER] = (
            g.request_id
        )
        return response