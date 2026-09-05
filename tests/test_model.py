"""
Tests for model loading, prediction, and SHAP explanations.
"""

import sys
from pathlib import Path

import numpy as np
import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

MODELS_DIR = PROJECT_ROOT / "models"


def _model_artifacts_exist() -> bool:
    """Check whether trained model artifacts are available."""
    return all(
        (MODELS_DIR / f).exists()
        for f in ["risk_model.pkl", "scaler.pkl", "feature_names.pkl"]
    )


@pytest.mark.skipif(not _model_artifacts_exist(), reason="Model not trained yet")
class TestRiskPredictor:
    """Tests for the RiskPredictor class."""

    @pytest.fixture(autouse=True)
    def setup(self):
        from src.predict import RiskPredictor

        self.predictor = RiskPredictor(
            model_path=str(MODELS_DIR / "risk_model.pkl"),
            scaler_path=str(MODELS_DIR / "scaler.pkl"),
            feature_names_path=str(MODELS_DIR / "feature_names.pkl"),
        )

    def _make_features(self) -> np.ndarray:
        """Create a sample feature vector using the FeatureExtractor."""
        from src.features import FeatureExtractor

        extractor = FeatureExtractor(
            feature_names=self.predictor.feature_names,
        )
        raw = {
            "amount": 45000.0,
            "payment_method": "UPI",
            "delivery_distance_km": 350.0,
            "customer_tenure_days": 2,
            "order_item_category": "Electronics",
            "payment_failure_history": 3,
            "is_foreign_ip": True,
            "hours_since_last_order": 0.5,
            "transaction_velocity_5min": 8,
            "device_age_days": 1,
            "is_new_device": 1,
            "attempted_payment_method_changes": 0,
        }
        return extractor.transform(raw)

    def test_predict_returns_float(self):
        """predict() should return a float between 0 and 1."""
        features = self._make_features()
        score = self.predictor.predict(features)
        assert isinstance(score, float)
        assert 0.0 <= score <= 1.0

    def test_predict_with_explanation_returns_tuple(self):
        """predict_with_explanation() should return (float, list[str])."""
        features = self._make_features()
        score, explanations = self.predictor.predict_with_explanation(features)
        assert isinstance(score, float)
        assert 0.0 <= score <= 1.0
        assert isinstance(explanations, list)
        assert len(explanations) <= 2
        for exp in explanations:
            assert isinstance(exp, str)

    def test_explanation_mentions_risk(self):
        """Explanation strings should contain 'risk' or relevant info."""
        features = self._make_features()
        _, explanations = self.predictor.predict_with_explanation(features)
        for exp in explanations:
            assert "risk" in exp.lower() or "increased" in exp.lower() or "%" in exp

    def test_feature_names_loaded(self):
        """Feature names should be a non-empty list."""
        assert isinstance(self.predictor.feature_names, list)
        assert len(self.predictor.feature_names) > 0

    def test_different_inputs_different_scores(self):
        """Different inputs should (generally) produce different scores."""
        from src.features import FeatureExtractor

        extractor = FeatureExtractor(feature_names=self.predictor.feature_names)

        high_risk = {
            "amount": 90000.0,
            "payment_method": "Card",
            "delivery_distance_km": 700.0,
            "customer_tenure_days": 1,
            "order_item_category": "Electronics",
            "payment_failure_history": 7,
            "is_foreign_ip": True,
            "hours_since_last_order": 0.1,
            "transaction_velocity_5min": 15,
            "device_age_days": 0,
            "is_new_device": 1,
            "attempted_payment_method_changes": 3,
        }
        low_risk = {
            "amount": 200.0,
            "payment_method": "UPI",
            "delivery_distance_km": 5.0,
            "customer_tenure_days": 1500,
            "order_item_category": "Grocery",
            "payment_failure_history": 0,
            "is_foreign_ip": False,
            "hours_since_last_order": 200.0,
            "transaction_velocity_5min": 0,
            "device_age_days": 1000,
            "is_new_device": 0,
            "attempted_payment_method_changes": 0,
        }

        score_high = self.predictor.predict(extractor.transform(high_risk))
        score_low = self.predictor.predict(extractor.transform(low_risk))
        assert score_high > score_low
