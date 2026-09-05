"""Prediction monitoring metrics."""

import threading


class PredictionMetrics:
    """Thread-safe prediction metrics collector."""

    def __init__(self):
        """Initialize prediction metrics."""
        self._lock = threading.Lock()
        self._predictions = []

    def record(self, prediction: float) -> None:
        """Record a model prediction."""
        with self._lock:
            self._predictions.append(
                float(prediction)
            )

    def snapshot(self) -> dict:
        """Return prediction statistics."""
        with self._lock:
            predictions = sorted(
                self._predictions
            )

        count = len(predictions)

        if not count:
            return {
                "prediction_count": 0,
                "prediction_mean": 0.0,
                "prediction_min": 0.0,
                "prediction_max": 0.0,
                "prediction_p50": 0.0,
                "prediction_p95": 0.0,
            }

        return {
            "prediction_count": count,
            "prediction_mean": (
                sum(predictions) / count
            ),
            "prediction_min": predictions[0],
            "prediction_max": predictions[-1],
            "prediction_p50": _percentile(
                predictions,
                50,
            ),
            "prediction_p95": _percentile(
                predictions,
                95,
            ),
        }


def _percentile(
    values: list[float],
    percentile: float,
) -> float:
    """Calculate a percentile using linear interpolation."""
    if len(values) == 1:
        return values[0]

    position = (
        (len(values) - 1)
        * percentile
        / 100
    )

    lower = int(position)
    upper = min(
        lower + 1,
        len(values) - 1,
    )

    weight = position - lower

    return (
        values[lower]
        + weight
        * (values[upper] - values[lower])
    )


prediction_metrics = PredictionMetrics()