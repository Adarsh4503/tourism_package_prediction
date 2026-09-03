"""Validate the tourism dataset and print a short summary."""
import os

import pandas as pd

DATA_PATH = os.path.normpath(
    os.path.join(os.path.dirname(__file__), "..", "data", "tourism.csv")
)

EXPECTED_COLUMNS = [
    "CustomerID", "ProdTaken", "Age", "TypeofContact", "CityTier",
    "Occupation", "Gender", "NumberOfPersonVisiting",
    "PreferredPropertyStar", "MaritalStatus", "NumberOfTrips",
    "Passport", "OwnCar", "NumberOfChildrenVisiting", "Designation",
    "MonthlyIncome", "PitchSatisfactionScore", "ProductPitched",
    "NumberOfFollowups", "DurationOfPitch",
]

TARGET_COLUMN = "ProdTaken"


def main() -> None:
    assert os.path.exists(DATA_PATH), f"Dataset not found at {DATA_PATH}"

    df = pd.read_csv(DATA_PATH)

    missing = [c for c in EXPECTED_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"Missing expected columns: {missing}")

    print(f"Rows: {df.shape[0]}, Columns: {df.shape[1]}")
    print("Columns:", list(df.columns))
    print()
    print("Target distribution (ProdTaken):")
    print(df[TARGET_COLUMN].value_counts())
    print()
    print("Missing values per column:")
    print(df.isna().sum())
    print()
    print("Dataset registered successfully.")


if __name__ == "__main__":
    main()
