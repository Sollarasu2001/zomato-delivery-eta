"""Health Check Routes"""

from flask import Blueprint, jsonify

health_bp = Blueprint("health", __name__)

@health_bp.route("/health", methods=["GET"])
def health_check():
    """Return API health status."""
    return jsonify({"status": "healthy", "service": "zomato_delivery_eta"}), 200