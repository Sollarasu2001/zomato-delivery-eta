"""Feature drift monitoring using Population Stability Index."""

import json
import math
import threading
from pathlib import Path

import pandas as pd


PSI_WARNING_THRESHOLD = 0.10
PSI_DRIFT_THRESHOLD = 0.20

REFERENCE_PROFILE_PATH = (
    Path(__file__).resolve().parents[3]
    / "configs"
    / "monitoring"
    / "reference_profile.json"
)


class DriftMonitor:
    """Track production observations and calculate feature drift."""

    def __init__(
        self,
        reference_profile_path=REFERENCE_PROFILE_PATH,
        max_observations=5000,
    ):
        """Initialize the drift monitor."""
        self.reference_profile_path = Path(
            reference_profile_path
        )
        self.max_observations = max_observations
        self._lock = threading.Lock()
        self._observations = []

        self.reference_profile = (
            self._load_reference_profile()
        )

    def record(self, data: dict) -> None:
        """Record one accepted production observation."""
        observation = dict(data)

        with self._lock:
            self._observations.append(observation)

            if (
                len(self._observations)
                > self.max_observations
            ):
                self._observations.pop(0)

    def snapshot(self) -> dict:
        """Calculate drift statistics."""
        with self._lock:
            observations = list(
                self._observations
            )

        if not observations:
            return {
                "drift_status": "insufficient_data",
                "drift_observations": 0,
                "feature_psi": {},
            }

        dataframe = pd.DataFrame(
            observations
        )

        feature_psi = {}

        for feature, profile in (
            self.reference_profile[
                "numeric"
            ].items()
        ):
            if feature in dataframe.columns:
                feature_psi[feature] = (
                    self._numeric_psi(
                        dataframe[feature],
                        profile,
                    )
                )

        for feature, profile in (
            self.reference_profile[
                "categorical"
            ].items()
        ):
            if feature in dataframe.columns:
                feature_psi[feature] = (
                    self._categorical_psi(
                        dataframe[feature],
                        profile,
                    )
                )

        status = self._overall_status(
            feature_psi
        )

        return {
            "drift_status": status,
            "drift_observations": len(
                observations
            ),
            "feature_psi": feature_psi,
        }

    def _load_reference_profile(self) -> dict:
        """Load the reference profile."""
        if not self.reference_profile_path.exists():
            raise FileNotFoundError(
                "Reference profile not found: "
                f"{self.reference_profile_path}"
            )

        with self.reference_profile_path.open(
            "r",
            encoding="utf-8",
        ) as file:
            return json.load(file)

    @staticmethod
    def _numeric_psi(
        series: pd.Series,
        profile: dict,
    ) -> float:
        """Calculate PSI for a numeric feature."""
        values = pd.to_numeric(
            series,
            errors="coerce",
        ).dropna()

        if values.empty:
            return 0.0

        edges = profile["bin_edges"]

        if len(edges) < 2:
            return 0.0

        reference_distribution = profile[
            "reference_distribution"
        ]

        production_distribution = (
            _numeric_distribution(
                values,
                edges,
            )
        )

        return _psi(
            reference_distribution,
            production_distribution,
        )

    @staticmethod
    def _categorical_psi(
        series: pd.Series,
        profile: dict,
    ) -> float:
        """Calculate PSI for a categorical feature."""
        values = (
            series.astype("string")
            .fillna("__MISSING__")
        )

        frequencies = profile[
            "frequencies"
        ]

        categories = set(
            frequencies
        ).union(values.unique())

        if not categories:
            return 0.0

        reference_distribution = []
        production_distribution = []

        value_counts = (
            values.value_counts(
                normalize=True
            )
        )

        for category in categories:
            reference_distribution.append(
                float(
                    frequencies.get(
                        category,
                        0.0,
                    )
                )
            )

            production_distribution.append(
                float(
                    value_counts.get(
                        category,
                        0.0,
                    )
                )
            )

        return _psi(
            reference_distribution,
            production_distribution,
        )

    @staticmethod
    def _overall_status(
        feature_psi: dict,
    ) -> str:
        """Determine overall drift status."""
        if not feature_psi:
            return "insufficient_data"

        maximum_psi = max(
            feature_psi.values()
        )

        if maximum_psi >= PSI_DRIFT_THRESHOLD:
            return "drift"

        if maximum_psi >= PSI_WARNING_THRESHOLD:
            return "warning"

        return "stable"


def _numeric_distribution(
    values: pd.Series,
    edges: list[float],
) -> list[float]:
    """Calculate production distribution."""
    clipped = values.clip(
        lower=edges[0],
        upper=edges[-1],
    )

    counts = pd.cut(
        clipped,
        bins=edges,
        include_lowest=True,
        duplicates="drop",
    ).value_counts(
        sort=False
    )

    total = counts.sum()

    if total == 0:
        return [0.0] * len(counts)

    return (
        counts / total
    ).tolist()


def _psi(
    reference: list[float],
    production: list[float],
) -> float:
    """Calculate Population Stability Index."""
    if len(reference) != len(production):
        return 0.0

    epsilon = 1e-6
    psi = 0.0

    for reference_value, production_value in zip(
        reference,
        production,
    ):
        reference_value = max(
            reference_value,
            epsilon,
        )

        production_value = max(
            production_value,
            epsilon,
        )

        psi += (
            production_value
            - reference_value
        ) * math.log(
            production_value
            / reference_value
        )

    return float(psi)


drift_monitor = DriftMonitor()