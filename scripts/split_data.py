"""Create reproducible train and test datasets."""

from pathlib import Path
import sys

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from delivery_eta.training.split import split_data


INPUT_PATH = ROOT / "data/interim/features.parquet"
OUTPUT_DIR = ROOT / "data/interim"

TARGET_COLUMN = "Time_taken(min)"


def main():
    """Split the feature dataset into train and test sets."""
    dataframe = pd.read_parquet(INPUT_PATH)

    features = dataframe.drop( columns=[TARGET_COLUMN])
    target = dataframe[TARGET_COLUMN]

    x_train, x_test, y_train, y_test = split_data(
        features,
        target,
        test_size=0.2,
        random_state=42,
    )

    x_train.to_parquet(OUTPUT_DIR / "x_train.parquet",index=False)

    x_test.to_parquet(OUTPUT_DIR / "x_test.parquet",index=False,)

    y_train.to_frame( name=TARGET_COLUMN).to_parquet(OUTPUT_DIR / "y_train.parquet",index=False)

    y_test.to_frame(name=TARGET_COLUMN,).to_parquet(OUTPUT_DIR / "y_test.parquet",index=False)

    print(f"Training Rows: {len(x_train)}")
    print(f"Testing Rows: {len(x_test)}")
    print(f"Training Target Mean: {y_train.mean():.2f}")
    print(f"Testing Target Mean: {y_test.mean():.2f}")


if __name__ == "__main__":
    main()