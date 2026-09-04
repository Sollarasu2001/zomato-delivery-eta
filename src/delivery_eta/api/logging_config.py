"""Application logging configuration."""

import logging
import sys

from delivery_eta.api.config import Config


def configure_logging():
    """Configure application logging."""
    log_level = getattr(
        logging,
        Config.LOG_LEVEL.upper(),
        logging.INFO,
    )

    logging.basicConfig(
        level=log_level,
        format=(
            "%(asctime)s "
            "%(levelname)s "
            "%(name)s "
            "%(message)s"
        ),
        stream=sys.stdout,
    )