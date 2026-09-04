"""Validate The Preprocessing Pipeline."""

from pathlib import Path
import sys
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


from delivery_eta.features.preprocessing import build_preprocessor
DATA_DIR = ROOT / "data/interim"

def main():
    """Fit Preprocessing on Training Data And Transform The Test Data."""
    x_train = pd.read_parquet(DATA_DIR / "x_train.parquet")
    x_test = pd.read_parquet(DATA_DIR / "x_test.parquet")

    preprocessor = build_preprocessor()

    x_train_transformed = preprocessor.fit_transform(x_train)
    x_test_transformed = preprocessor.transform(x_test)

    print("Original Training Data Shape:", x_train.shape)
    print("Transformed Training Data Shape:", x_train_transformed.shape)
    print("Original Test Data Shape:", x_test.shape)
    print("Transformed Test Data Shape:", x_test_transformed.shape)
    print("Training NaN count:",pd.isna(x_train_transformed).sum())
    print("Testing NaN count:",pd.isna(x_test_transformed).sum())


if __name__ == "__main__":
    main()