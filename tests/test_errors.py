"""Tests for application-wide error handlers."""


def test_unknown_endpoint(client):
    """Test unknown endpoint handling."""
    response = client.get(
        "/does-not-exist"
    )

    assert response.status_code == 404

    data = response.get_json()

    assert data["error"] == (
        "Endpoint not found."
    )