"""Application-wide API error handling."""

import logging

from flask import jsonify
from marshmallow import ValidationError


logger = logging.getLogger(
    __name__
)


def register_error_handlers(app):
    """Register application-wide error handlers."""

    @app.errorhandler(ValidationError)
    def handle_validation_error(error):
        """Return a structured validation error."""
        logger.warning(
            "request_validation_failed"
        )

        return jsonify(
            {
                "error": "Invalid request.",
                "fields": error.messages,
            }
        ), 400

    @app.errorhandler(400)
    def handle_bad_request(error):
        """Return a structured bad-request response."""
        logger.warning(
            "bad_request"
        )

        return jsonify(
            {
                "error": "Bad request.",
            }
        ), 400

    @app.errorhandler(404)
    def handle_not_found(error):
        """Return a structured not-found response."""
        return jsonify(
            {
                "error": "Endpoint not found.",
            }
        ), 404

    @app.errorhandler(405)
    def handle_method_not_allowed(error):
        """Return a structured method error."""
        return jsonify(
            {
                "error": "Method not allowed.",
            }
        ), 405

    @app.errorhandler(500)
    def handle_internal_error(error):
        """Return a safe internal-server-error response."""
        logger.exception(
            "internal_server_error"
        )

        return jsonify(
            {
                "error": "Internal server error.",
            }
        ), 500