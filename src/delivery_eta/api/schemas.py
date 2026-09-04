"""Request and response schemas for the prediction API."""

from marshmallow import Schema
from marshmallow import fields
from marshmallow import validate


class DeliveryPredictionSchema(Schema):
    """Validate raw delivery prediction input."""

    delivery_person_age = fields.Float(
        allow_none=True,
        load_default=None,
        validate=validate.Range(
            min=15,
            max=50,
        ),
    )

    delivery_person_ratings = fields.Float(
        allow_none=True,
        load_default=None,
        validate=validate.Range(
            min=1,
            max=6,
        ),
    )

    restaurant_latitude = fields.Float(
        required=True,
        validate=validate.Range(
            min=-90,
            max=90,
        ),
    )

    restaurant_longitude = fields.Float(
        required=True,
        validate=validate.Range(
            min=-180,
            max=180,
        ),
    )

    delivery_location_latitude = fields.Float(
        required=True,
        validate=validate.Range(
            min=-90,
            max=90,
        ),
    )

    delivery_location_longitude = fields.Float(
        required=True,
        validate=validate.Range(
            min=-180,
            max=180,
        ),
    )

    weather_conditions = fields.String(
        allow_none=True,
        load_default=None,
    )

    road_traffic_density = fields.String(
        allow_none=True,
        load_default=None,
    )

    vehicle_condition = fields.Integer(
        required=True,
        validate=validate.Range(
            min=0,
            max=3,
        ),
    )

    type_of_order = fields.String(
        required=True,
    )

    type_of_vehicle = fields.String(
        required=True,
    )

    multiple_deliveries = fields.Float(
        allow_none=True,
        load_default=None,
        validate=validate.Range(
            min=0,
            max=3,
        ),
    )

    festival = fields.String(
        allow_none=True,
        load_default=None,
    )

    city = fields.String(
        allow_none=True,
        load_default=None,
    )

    order_date = fields.String(
        required=True,
    )

    time_ordered = fields.String(
        allow_none=True,
        load_default=None,
    )