import numpy as np
import pytest
from sklearn.ensemble import RandomForestClassifier

from src.data_preprocessing import load_data, preprocess_data, split_data
from src.evaluate import evaluate_model


@pytest.fixture
def trained_pipeline():
    df = load_data("data/heart_disease.csv")
    X, y = preprocess_data(df, target_col="num")
    X_train, X_test, y_train, y_test = split_data(X, y)

    model = RandomForestClassifier(n_estimators=50, random_state=42)
    model.fit(X_train, y_train)
    return model, X_test, y_test


def test_model_prediction_shape_and_type(trained_pipeline):
    """Verify model produces array predictions matching test set length."""
    model, X_test, _ = trained_pipeline
    predictions = model.predict(X_test)

    assert isinstance(predictions, np.ndarray)
    assert len(predictions) == len(X_test)


def test_model_minimum_performance_threshold(trained_pipeline):
    """Verify baseline model achieves at least 0.75 F1 score."""
    model, X_test, y_test = trained_pipeline
    metrics = evaluate_model(model, X_test, y_test)

    assert metrics["f1_score"] >= 0.75
    assert metrics["accuracy"] >= 0.75
