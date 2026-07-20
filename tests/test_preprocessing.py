import numpy as np
import pandas as pd
import pytest

from src.data_preprocessing import (
    handle_missing_values,
    load_data,
    preprocess_data,
)


def test_load_data_file_not_found():
    """Test that load_data raises FileNotFoundError for invalid path."""
    with pytest.raises(FileNotFoundError):
        load_data("data/non_existent_file.csv")


def test_handle_missing_values_imputation():
    """Test that missing values and '?' strings are properly imputed."""
    df_with_missing = pd.DataFrame(
        {"numeric": [1.0, 2.0, np.nan, 4.0],
            "categorical": ["a", "?", "a", "b"]}
    )
    cleaned = handle_missing_values(df_with_missing)
    assert cleaned["numeric"].isnull().sum() == 0
    assert "?" not in cleaned["categorical"].values


def test_preprocess_data_unmodified_original():
    """Test that preprocess_data does not mutate the original input DataFrame."""
    original_df = pd.DataFrame(
        {"age": [50, 60], "sex": [1, 0], "num": [0, 2]}
    )
    df_copy = original_df.copy()
    preprocess_data(original_df, target_col="num")

    pd.testing.assert_frame_equal(original_df, df_copy)


def test_preprocess_data_missing_target_keyerror():
    """Test that missing target column raises a KeyError."""
    df = pd.DataFrame({"age": [50, 60], "sex": [1, 0]})
    with pytest.raises(KeyError):
        preprocess_data(df, target_col="invalid_target")


def test_preprocess_data_target_binarization():
    """Test that non-zero target values are binarized to 0 and 1."""
    df = pd.DataFrame({"age": [50, 60, 70], "num": [0, 1, 3]})
    X, y = preprocess_data(df, target_col="num")
    assert set(y.unique()).issubset({0, 1})


def test_preprocess_data_one_hot_encoding():
    """Test that string categorical columns are encoded properly."""
    df = pd.DataFrame(
        {"cp": ["typical", "atypical", "typical"], "num": [0, 1, 0]}
    )
    X, y = preprocess_data(df, target_col="num")
    assert "cp_typical" in X.columns or "cp_atypical" in X.columns
