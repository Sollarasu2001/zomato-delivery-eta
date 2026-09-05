"""Create a feature drift reference profile."""

import json
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]

INPUT_PATH = (
    ROOT
    / "data"
    / "interim"
    / "x_train.parquet"
)

OUTPUT_PATH = (
    ROOT
    / "configs"
    / "monitoring"
    / "reference_profile.json"
)

NUMERIC_FEATURES = (
    "Delivery_person_Age",
    "Delivery_person_Ratings",
    "Restaurant_latitude",
    "Restaurant_longitude",
    "Delivery_location_latitude",
    "Delivery_location_longitude",
    "Vehicle_condition",
    "multiple_deliveries",
    "order_hour",
    "order_day_of_week",
    "order_month",
    "is_weekend",
    "delivery_distance_km",
)

CATEGORICAL_FEATURES = (
    "Weather_conditions",
    "Road_traffic_density",
    "Type_of_order",
    "Type_of_vehicle",
    "Festival",
    "City",
)


def build_numeric_profile(
    series: pd.Series,
) -> dict:
    """Build a numeric feature profile."""
    values = pd.to_numeric(
        series,
        errors="coerce",
    ).dropna()

    quantiles = values.quantile(
        [
            0.00,
            0.10,
            0.20,
            0.30,
            0.40,
            0.50,
            0.60,
            0.70,
            0.80,
            0.90,
            1.00,
        ]
    ).tolist()

    bin_edges = _make_unique_edges(
        quantiles
    )

    reference_distribution = (
        _calculate_distribution(
            values,
            bin_edges,
        )
    )

    return {
        "type": "numeric",
        "bin_edges": bin_edges,
        "reference_distribution": (
            reference_distribution
        ),
        "count": int(len(values)),
    }


def build_categorical_profile(
    series: pd.Series,
) -> dict:
    """Build a categorical feature profile."""
    values = (
        series.astype("string")
        .fillna("__MISSING__")
    )

    frequencies = (
        values.value_counts(
            normalize=True
        )
        .to_dict()
    )

    return {
        "type": "categorical",
        "frequencies": frequencies,
        "count": int(len(values)),
    }


def _calculate_distribution(
    values: pd.Series,
    edges: list[float],
) -> list[float]:
    """Calculate the reference distribution."""
    if len(edges) < 2:
        return [1.0]

    counts = pd.cut(
        values,
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
    ).astype(float).tolist()


def _make_unique_edges(
    edges: list[float],
) -> list[float]:
    """Ensure histogram bin edges are strictly increasing."""
    unique_edges = []

    for edge in edges:
        if not unique_edges or edge > unique_edges[-1]:
            unique_edges.append(float(edge))

    if len(unique_edges) == 1:
        value = unique_edges[0]
        unique_edges = [
            value - 0.5,
            value + 0.5,
        ]

    return unique_edges


def main() -> None:
    """Create and save the reference profile."""
    dataframe = pd.read_parquet(
        INPUT_PATH
    )

    profile = {
        "source": str(
            INPUT_PATH.relative_to(ROOT)
        ),
        "numeric": {},
        "categorical": {},
    }

    for column in NUMERIC_FEATURES:
        profile["numeric"][column] = (
            build_numeric_profile(
                dataframe[column]
            )
        )

    for column in CATEGORICAL_FEATURES:
        profile["categorical"][column] = (
            build_categorical_profile(
                dataframe[column]
            )
        )

    OUTPUT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    OUTPUT_PATH.write_text(
        json.dumps(
            profile,
            indent=2,
        ),
        encoding="utf-8",
    )

    print(
        f"Reference profile saved to "
        f"{OUTPUT_PATH}"
    )


if __name__ == "__main__":
    main()