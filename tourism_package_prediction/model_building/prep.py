"""Clean and split the raw tourism dataset into train/test CSV files."""
import os

import pandas as pd
from sklearn.model_selection import train_test_split

DATA_PATH = os.path.normpath(
    os.path.join(os.path.dirname(__file__), "..", "data", "tourism.csv")
)

DROP_COLUMNS = ["CustomerID"]
TARGET_COLUMN = "ProdTaken"

TEST_SIZE = 0.2
RANDOM_STATE = 42


def _clean(df: pd.DataFrame) -> pd.DataFrame:
    """Drop identifier columns and impute missing values."""
    for col in DROP_COLUMNS:
        if col in df.columns:
            df = df.drop(columns=[col])

    numeric_cols = df.select_dtypes(include=["int64", "float64"]).columns
    categorical_cols = df.select_dtypes(include=["object"]).columns

    for col in numeric_cols:
        if df[col].isna().any():
            df[col] = df[col].fillna(df[col].median())

    for col in categorical_cols:
        if df[col].isna().any():
            df[col] = df[col].fillna(df[col].mode().iloc[0])

    return df


def main() -> None:
    assert os.path.exists(DATA_PATH), f"Dataset not found at {DATA_PATH}"

    df = pd.read_csv(DATA_PATH)
    before = df.shape
    df = _clean(df)
    print(f"Loaded shape: {before}  Cleaned shape: {df.shape}")
    print(f"Remaining missing values: {int(df.isna().sum().sum())}")

    assert TARGET_COLUMN in df.columns, f"{TARGET_COLUMN} not in dataset"

    y = df[TARGET_COLUMN]
    X = df.drop(columns=[TARGET_COLUMN])

    Xtrain, Xtest, ytrain, ytest = train_test_split(
        X, y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y,
    )

    Xtrain.to_csv("Xtrain.csv", index=False)
    Xtest.to_csv("Xtest.csv", index=False)
    ytrain.to_csv("ytrain.csv", index=False)
    ytest.to_csv("ytest.csv", index=False)

    print(f"Train shape: {Xtrain.shape}, Test shape: {Xtest.shape}")
    print("Saved Xtrain.csv, Xtest.csv, ytrain.csv, ytest.csv")


if __name__ == "__main__":
    main()
