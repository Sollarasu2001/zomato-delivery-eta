"""Pytest Fixtures For The ETA API."""

import pytest

from delivery_eta.api.app import create_app

@pytest.fixture
def client():
    """Create a Flask test client."""
    app = create_app()

    app.config.update(TESTING=True)

    with app.test_client() as test_client:
        yield test_client