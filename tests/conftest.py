# """Pytest Fixtures For The ETA API."""

# import pytest

# from delivery_eta.api.app import create_app

# @pytest.fixture
# def client():
#     """Create a Flask test client."""
#     app = create_app()

#     app.config.update(TESTING=True)

#     with app.test_client() as test_client:
#         yield test_client






####VERSION 2
"""Pytest fixtures for the ETA API."""

import pytest

from delivery_eta.api.app import create_app
from delivery_eta.api.routes import prediction


class FakePredictor:
    """Fake predictor for API tests."""

    def predict(self, data):
        """Return a deterministic test prediction."""
        return 30.0


@pytest.fixture
def client():
    """Create a Flask test client."""
    app = create_app()

    prediction.predictor = (
        FakePredictor()
    )

    app.config.update(
        TESTING=True,
    )

    with app.test_client() as test_client:
        yield test_client