"""Generate a compact first-pass data quality report."""

from pathlib import Path
import sys

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from delivery_eta.data.loader import load_csv
from delivery_eta.data.validation import validate_dataframe


DATA_PATH = ROOT / "data/raw/delivery_data.csv"
REPORT_PATH = ROOT / "artifacts/reports/data_quality_report.csv"


def main():
    """
    Create Column-Level Missingness And Dtype Profile.
    """
    dataframe = load_csv(DATA_PATH)
    summary = validate_dataframe(dataframe)

    print("Dataset Summary:")
    for key, value in summary.items():
        print(f"{key}: {value}")

    report = pd.DataFrame(
        {
            "dtype": dataframe.dtypes.astype(str),
            "missing_count": dataframe.isna().sum(),
            "missing_pct": dataframe.isna().mean().mul(100),
            "unique_count": dataframe.nunique(dropna=True),
        }
    ).sort_values(by="missing_count", ascending=False)

    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    report.to_csv(REPORT_PATH, index=True)
    print(f"Data quality report saved to: {REPORT_PATH}")


if __name__ == "__main__":
    main()

