import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split


def load_data(file_path: str) -> pd.DataFrame:
    """Loads raw dataset from a CSV file."""
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Dataset not found at: {file_path}")
    df = pd.read_csv(file_path)
    if df.empty:
        raise ValueError("The loaded dataset is empty.")
    return df


def handle_missing_values(df: pd.DataFrame) -> pd.DataFrame:
    """Imputes missing values without mutating the original DataFrame."""
    df_clean = df.copy()

    # Replace '?' or invalid strings with NaN if present
    df_clean.replace("?", np.nan, inplace=True)

    # Fill missing values: median for numeric, mode for categorical
    for col in df_clean.columns:
        if df_clean[col].dtype in ["float64", "int64"]:
            df_clean[col] = df_clean[col].fillna(df_clean[col].median())
        else:
            mode_val = df_clean[col].mode()
            if not mode_val.empty:
                df_clean[col] = df_clean[col].fillna(mode_val[0])

    return df_clean


def preprocess_data(
    df: pd.DataFrame, target_col: str = "num"
) -> tuple[pd.DataFrame, pd.Series]:
    """Preprocesses dataset: missing value handling, target binarization, and encoding."""
    if target_col not in df.columns:
        raise KeyError(f"Target column '{target_col}' not found in DataFrame.")

    # Ensure original dataframe remains unmodified
    df_processed = handle_missing_values(df)

    # Convert target 'num' (0 = No disease, >0 = Disease) into binary (0 or 1)
    df_processed[target_col] = (df_processed[target_col] > 0).astype(int)

    # Separate features and target
    X = df_processed.drop(columns=[target_col])
    y = df_processed[target_col]

    # One-hot encode categorical features if string types exist
    X = pd.get_dummies(X, drop_first=True)

    return X, y


def split_data(
    X: pd.DataFrame, y: pd.Series, test_size: float = 0.2, random_state: int = 42
):
    """Splits feature matrix and target vector into train and test sets."""
    return train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )


if __name__ == "__main__":
    import yaml

    with open("configs/config.yaml", "r") as f:
        config = yaml.safe_load(f)

    raw_df = load_data(config["data"]["raw_path"])
    X, y = preprocess_data(raw_df, target_col=config["data"]["target_column"])
    X_train, X_test, y_train, y_test = split_data(
        X,
        y,
        test_size=config["data"]["test_size"],
        random_state=config["data"]["random_state"],
    )

    print(
        f"Preprocessing completed successfully! Train shape: {X_train.shape},"
        f" Test shape: {X_test.shape}"
    )
