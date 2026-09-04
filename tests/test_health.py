"""Tests for the health endpoint."""


def test_health_check(client):
    """Test the health endpoint."""
    response = client.get(
        "/health"
    )

    assert response.status_code == 200

    data = response.get_json()

    assert data["status"] == "healthy"
    assert data["service"] == "zomato_delivery_eta"