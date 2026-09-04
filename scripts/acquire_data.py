""" Download and stage the raw Delivery ETA dataset."""

from pathlib import Path
RAW_DIR = Path("data/raw")
DATASET_PATH = RAW_DIR / "delivery_data.csv"

SOURCE_URL = (
    "https://www.kaggle.com/datasets/saurabhbadole/"
    "zomato-delivery-operations-analytics-dataset"
)


def main():
    """
    Print acquisition instructions fro the raw dataset.
    """
    RAW_DIR.mkdir(parents=True, exist_ok=True)

    print("Raw Dataset Source:")
    print(SOURCE_URL)
    print()
    print(f" Place The Downloaded CSV at: {DATASET_PATH}")

if __name__ == "__main__":
    main()