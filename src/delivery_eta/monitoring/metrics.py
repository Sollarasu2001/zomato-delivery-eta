"""Application metrics."""

import threading
import time
from collections import Counter


class Metrics:
    """Thread-safe in-process API metrics collector."""

    def __init__(self):
        """Initialize metrics."""
        self._lock = threading.Lock()
        self._request_count = 0
        self._status_counts = Counter()
        self._latencies_ms = []

    def record_request(
        self,
        *,
        status_code: int,
        latency_ms: float,
    ) -> None:
        """Record a completed HTTP request."""
        with self._lock:
            self._request_count += 1
            self._status_counts[status_code] += 1
            self._latencies_ms.append(latency_ms)

    def snapshot(self) -> dict:
        """Return a metrics snapshot."""
        with self._lock:
            latencies = list(self._latencies_ms)

            return {
                "request_count": self._request_count,
                "status_counts": dict(
                    self._status_counts
                ),
                "latency_ms": {
                    "count": len(latencies),
                    "mean": (
                        sum(latencies) / len(latencies)
                        if latencies
                        else 0.0
                    ),
                    "max": (
                        max(latencies)
                        if latencies
                        else 0.0
                    ),
                },
            }


metrics = Metrics()