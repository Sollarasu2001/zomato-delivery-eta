"""Tests for the prediction endpoint."""


VALID_PAYLOAD = {
    "delivery_person_age": 30,
    "delivery_person_ratings": 4.8,
    "restaurant_latitude": 12.9716,
    "restaurant_longitude": 77.5946,
    "delivery_location_latitude": 12.9352,
    "delivery_location_longitude": 77.6245,
    "weather_conditions": "Sunny",
    "road_traffic_density": "High",
    "vehicle_condition": 2,
    "type_of_order": "Meal",
    "type_of_vehicle": "Motorcycle",
    "multiple_deliveries": 1,
    "festival": "No",
    "city": "Metropolitian",
    "order_date": "15-08-2022",
    "time_ordered": "19:30",
}


def test_prediction_success(client):
    """Test a valid prediction request."""
    response = client.post(
        "/predict",
        json=VALID_PAYLOAD,
    )

    assert response.status_code == 200

    data = response.get_json()

    assert (
        "predicted_delivery_time_minutes"
        in data
    )

    prediction = (
        data[
            "predicted_delivery_time_minutes"
        ]
    )

    assert isinstance(
        prediction,
        float,
    )

    assert prediction > 0


def test_prediction_missing_required_field(client):
    """Test missing required input."""
    payload = VALID_PAYLOAD.copy()

    del payload[
        "vehicle_condition"
    ]

    response = client.post(
        "/predict",
        json=payload,
    )

    assert response.status_code == 400

    data = response.get_json()

    assert data["error"] == (
        "Invalid request."
    )

    assert (
        "vehicle_condition"
        in data["fields"]
    )


def test_prediction_invalid_vehicle_condition(
    client,
):
    """Test invalid vehicle condition."""
    payload = VALID_PAYLOAD.copy()

    payload[
        "vehicle_condition"
    ] = 10

    response = client.post(
        "/predict",
        json=payload,
    )

    assert response.status_code == 400

    data = response.get_json()

    assert data["error"] == (
        "Invalid request."
    )

    assert (
        "vehicle_condition"
        in data["fields"]
    )


def test_prediction_invalid_latitude(
    client,
):
    """Test invalid latitude."""
    payload = VALID_PAYLOAD.copy()

    payload[
        "restaurant_latitude"
    ] = 150

    response = client.post(
        "/predict",
        json=payload,
    )

    assert response.status_code == 400

    data = response.get_json()

    assert (
        "restaurant_latitude"
        in data["fields"]
    )


def test_prediction_invalid_age(
    client,
):
    """Test invalid delivery person age."""
    payload = VALID_PAYLOAD.copy()

    payload[
        "delivery_person_age"
    ] = 100

    response = client.post(
        "/predict",
        json=payload,
    )

    assert response.status_code == 400

    data = response.get_json()

    assert (
        "delivery_person_age"
        in data["fields"]
    )


def test_prediction_invalid_content_type(
    client,
):
    """Test non-JSON content type."""
    response = client.post(
        "/predict",
        data="not-json",
        content_type="text/plain",
    )

    assert response.status_code == 400

    data = response.get_json()

    assert data["error"] == (
        "Content-Type must be "
        "application/json."
    )


def test_prediction_invalid_json(
    client,
):
    """Test malformed JSON."""
    response = client.post(
        "/predict",
        data="{invalid-json}",
        content_type="application/json",
    )

    assert response.status_code == 400

    data = response.get_json()

    assert data["error"] == (
        "Request body must contain "
        "valid JSON."
    )


def test_prediction_wrong_method(
    client,
):
    """Test unsupported HTTP method."""
    response = client.get(
        "/predict"
    )

    assert response.status_code == 405

    data = response.get_json()

    assert data["error"] == (
        "Method not allowed."
    )


def test_prediction_request_id(client):
    """Test that a request ID is returned."""
    response = client.post(
        "/predict",
        json=VALID_PAYLOAD,
    )

    assert response.status_code == 200
    assert response.headers["X-Request-ID"]


def test_prediction_preserves_request_id(client):
    """Test that a supplied request ID is preserved."""
    request_id = "test-request-123"

    response = client.post(
        "/predict",
        json=VALID_PAYLOAD,
        headers={
            "X-Request-ID": request_id,
        },
    )

    assert response.status_code == 200
    assert (
        response.headers["X-Request-ID"]
        == request_id
    )