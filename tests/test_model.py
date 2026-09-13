"""
test_model.py
-------------
Basic tests for the churn prediction pipeline.
Run with: pytest tests/
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from inference import predict_one  # noqa: E402
from train import build_pipeline, load_data, train  # noqa: E402


def test_data_loads():
    df = load_data()
    assert len(df) > 0
    assert "Churn" in df.columns
    assert "tenure" in df.columns


def test_pipeline_builds():
    pipeline = build_pipeline()
    assert pipeline is not None


def test_model_trains_and_scores_reasonably():
    df = load_data()
    model, metrics = train(df)
    assert 0.0 <= metrics["accuracy"] <= 1.0
    assert 0.0 <= metrics["precision"] <= 1.0
    assert 0.0 <= metrics["recall"] <= 1.0
    # sanity check: should beat a coin flip on this synthetic data
    assert metrics["accuracy"] > 0.5


def test_predict_one_long_tenure_two_year_contract_predicts_no_churn():
    result = predict_one({"tenure": 72, "MonthlyCharges": 42.1, "Contract": "Two year"})
    assert result == {"Churn Prediction": "No"}


def test_predict_one_short_tenure_month_to_month_predicts_churn():
    result = predict_one({"tenure": 1, "MonthlyCharges": 100.0, "Contract": "Month-to-month"})
    assert result == {"Churn Prediction": "Yes"}


def test_predict_one_returns_expected_shape():
    result = predict_one({"tenure": 30, "MonthlyCharges": 60.0, "Contract": "One year"})
    assert "Churn Prediction" in result
    assert result["Churn Prediction"] in ("Yes", "No")
