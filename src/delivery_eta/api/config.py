"""Application configuration."""

import os
from pathlib import Path


ROOT = Path(__file__).resolve().parents[3]


class Config:
    """Base application configuration."""

    MODEL_PATH = Path(
        os.getenv(
            "MODEL_PATH",
            ROOT
            / "models"
            / "polynomial_ridge.joblib",
        )
    )

    HOST = os.getenv(
        "HOST",
        "0.0.0.0",
    )

    PORT = int(
        os.getenv(
            "PORT",
            "8000",
        )
    )

    LOG_LEVEL = os.getenv(
        "LOG_LEVEL",
        "INFO",
    )

    ENVIRONMENT = os.getenv(
        "ENVIRONMENT",
        "development",
    )