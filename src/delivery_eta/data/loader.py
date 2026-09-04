""" Data loading utitilies."""

from pathlib import Path
import pandas as pd

REQUIRED_TARGET = "Time_taken(min)"

def load_csv(path):
    """ Load a CSV and verify that the target column exists."""
    path = Path(path)

    if not path.exists():
        raise FileNotFoundError(f"Dataset not found at {path}")

    dataframe = pd.read_csv(path)

    if REQUIRED_TARGET not in dataframe.columns:
        raise ValueError(f"Missing target column: {REQUIRED_TARGET}")

    return dataframe

def load_parquet(path):
    """Load a Parquet file into a DataFrame."""
    path = Path(path)

    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")

    return pd.read_parquet(path)