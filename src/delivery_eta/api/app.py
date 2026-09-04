"""Flask application factory."""

import logging

from flask import Flask

from delivery_eta.api.config import Config
from delivery_eta.api.errors import (
    register_error_handlers,
)
from delivery_eta.api.logging_config import (
    configure_logging,
)
from delivery_eta.api.routes.health import (
    health_bp,
)
from delivery_eta.api.routes.prediction import (
    prediction_bp,
)


logger = logging.getLogger(
    __name__
)


def create_app():
    """Create and configure the Flask application."""
    configure_logging()

    app = Flask(__name__)

    app.config.from_object(
        Config
    )

    register_error_handlers(
        app
    )

    app.register_blueprint(
        health_bp
    )

    app.register_blueprint(
        prediction_bp
    )

    logger.info(
        "application_started"
    )

    return app


if __name__ == "__main__":
    app = create_app()

    app.run(
        host=Config.HOST,
        port=Config.PORT,
        debug=False,
    )