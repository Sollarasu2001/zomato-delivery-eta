"""Build the modeling feature dataset."""

from pathlib import Path
import sys
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from delivery_eta.data.loader import load_parquet
from delivery_eta.features.engineering import add_features

INPUT_PATH = ROOT / "data/interim/cleaned_data.parquet"
OUTPUT_PATH = ROOT / "data/interim/features.parquet"
TARGET_COLUMN = "Time_taken(min)"


def main():
    """Create deterministic modeling features."""
    dataframe = load_parquet(INPUT_PATH)

    features = add_features(dataframe)

    target = features.pop(TARGET_COLUMN)

    features[TARGET_COLUMN] = target

    OUTPUT_PATH.parent.mkdir(parents=True,exist_ok=True)

    features.to_parquet(OUTPUT_PATH,index=False)

    print(f"Rows: {len(features)}")
    print(f"Columns: {len(features.columns)}")

    print("\nColumns:")
    for column in features.columns:
        print(f"  - {column}")

    print("\nMissing values:")
    print(
        features.isna()
        .sum()
        .sort_values(ascending=False)
        .to_string()
    )

    print(f"\nSaved to: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()