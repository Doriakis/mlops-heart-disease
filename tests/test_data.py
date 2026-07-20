import os
import pandas as pd
import pytest

from src.data_preprocessing import load_data, preprocess_data


@pytest.fixture
def raw_dataset():
    path = "data/heart_disease.csv"
    if not os.path.exists(path):
        pytest.skip("Raw dataset not found at data/heart_disease.csv")
    return load_data(path)


def test_expected_columns_exist(raw_dataset):
    """Verify essential features exist in dataset."""
    required_cols = ["age", "sex", "cp", "trestbps", "chol", "num"]
    for col in required_cols:
        assert col in raw_dataset.columns


def test_target_variable_range(raw_dataset):
    """Verify processed target variable is strictly binary (0 or 1)."""
    _, y = preprocess_data(raw_dataset, target_col="num")
    assert set(y.unique()).issubset({0, 1})


def test_numeric_feature_ranges(raw_dataset):
    """Verify numeric features fall within expected physical boundaries."""
    assert raw_dataset["age"].min() > 0 and raw_dataset["age"].max() < 120
    assert raw_dataset["trestbps"].min() > 0
    assert raw_dataset["chol"].min() > 0
