""" Clean The Raw Delivery Dataset"""

from pathlib import Path
import sys
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from delivery_eta.data.loader import load_csv
from delivery_eta.data.cleaning import clean_data

RAW_PATH = ROOT / "data/raw/delivery_data.csv"
OUTPUT_PATH = ROOT / "data/interim/cleaned_data.parquet"

def main():
    """Run Deterministic Data Cleaning Pipeline"""
    dataframe = load_csv(RAW_PATH)

    original_rows = len(dataframe)

    cleaned = clean_data(dataframe)

    rows_removed = original_rows - len(cleaned)
    coordinate_repaired = int(cleaned["coordinate_repaired"].sum())

    print(f"Original Rows: {original_rows}")
    print(f"Cleaned Rows: {len(cleaned)}")
    print(f"Rows Removed: {rows_removed}")
    print(f"Coordinates Repaired: {coordinate_repaired}")

    print("\nMissing Values after Cleaning:")
    print(cleaned.isna().sum().sort_values(ascending=False).to_string())

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    cleaned.to_parquet(OUTPUT_PATH, index=False)

    print(f"\nCleaned Data Saved to: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
