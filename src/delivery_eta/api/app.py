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
from delivery_eta.api.routes.metrics import (
    metrics_bp,
)
from delivery_eta.api.request_context import (
    register_request_context,
)
from delivery_eta.api.request_context import (
    register_request_context,
)

logger = logging.getLogger(
    __name__
)


def create_app():
    """Create and configure the Flask application."""
    app = Flask(__name__)

    app.config.from_object(Config)

    configure_logging()
    register_request_context(app)
    register_error_handlers(app)
    register_request_context(app)

    app.register_blueprint(health_bp)
    app.register_blueprint(prediction_bp)
    app.register_blueprint(metrics_bp)

    logger.info("application_started")

    return app

if __name__ == "__main__":
    app = create_app()

    app.run(
        host=Config.HOST,
        port=Config.PORT,
        debug=False,
    )