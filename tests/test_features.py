"""
Tests for the FeatureExtractor class.
"""

import sys
from pathlib import Path

import numpy as np
import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.features import (  # noqa: E402
    BOOLEAN_COLS,
    CATEGORICAL_COLS,
    DERIVED_FEATURE_NAMES,
    NUMERICAL_COLS,
    FeatureExtractor,
)


@pytest.fixture
def extractor():
    """Return a fresh FeatureExtractor."""
    ext = FeatureExtractor()
    ext.build_feature_names()
    return ext


@pytest.fixture
def sample_raw():
    """A sample raw transaction dict."""
    return {
        "amount": 25000.0,
        "payment_method": "UPI",
        "delivery_distance_km": 100.0,
        "customer_tenure_days": 365,
        "order_item_category": "Apparel",
        "payment_failure_history": 1,
        "is_foreign_ip": False,
        "hours_since_last_order": 24.0,
        "transaction_velocity_5min": 2,
        "device_age_days": 200,
        "is_new_device": 0,
        "attempted_payment_method_changes": 0,
    }


class TestFeatureExtractor:
    """Tests for FeatureExtractor transformations."""

    def test_build_feature_names(self, extractor):
        """Feature names should include numerical, boolean, one-hot, and derived."""
        names = extractor.get_feature_names()
        assert len(names) > 0

        # Check numerical columns are present
        for col in NUMERICAL_COLS:
            assert col in names

        # Check boolean columns
        for col in BOOLEAN_COLS:
            assert col in names

        # Check one-hot columns
        for col, categories in CATEGORICAL_COLS.items():
            for cat in categories:
                assert f"{col}_{cat}" in names

        # Check derived features
        for feat in DERIVED_FEATURE_NAMES:
            assert feat in names

    def test_transform_output_shape(self, extractor, sample_raw):
        """transform() should return shape (1, n_features)."""
        result = extractor.transform(sample_raw)
        assert isinstance(result, np.ndarray)
        assert result.ndim == 2
        assert result.shape[0] == 1
        assert result.shape[1] == len(extractor.get_feature_names())

    def test_one_hot_encoding_upi(self, extractor, sample_raw):
        """UPI should have payment_method_UPI = 1, others = 0."""
        result = extractor.transform(sample_raw)
        names = extractor.get_feature_names()

        upi_idx = names.index("payment_method_UPI")
        card_idx = names.index("payment_method_Card")

        assert result[0, upi_idx] == 1.0
        assert result[0, card_idx] == 0.0

    def test_one_hot_encoding_category(self, extractor, sample_raw):
        """Apparel should have order_item_category_Apparel = 1."""
        result = extractor.transform(sample_raw)
        names = extractor.get_feature_names()

        apparel_idx = names.index("order_item_category_Apparel")
        electronics_idx = names.index("order_item_category_Electronics")

        assert result[0, apparel_idx] == 1.0
        assert result[0, electronics_idx] == 0.0

    def test_derived_amount_per_tenure(self, extractor, sample_raw):
        """amount_per_tenure should be amount / (tenure + 1)."""
        result = extractor.transform(sample_raw)
        names = extractor.get_feature_names()
        apt_idx = names.index("amount_per_tenure")
        expected = sample_raw["amount"] / (sample_raw["customer_tenure_days"] + 1)
        np.testing.assert_almost_equal(result[0, apt_idx], expected, decimal=2)

    def test_derived_is_high_velocity_false(self, extractor, sample_raw):
        """velocity=2 should give is_high_velocity=0."""
        result = extractor.transform(sample_raw)
        names = extractor.get_feature_names()
        hv_idx = names.index("is_high_velocity")
        assert result[0, hv_idx] == 0.0

    def test_derived_is_high_velocity_true(self, extractor, sample_raw):
        """velocity=10 should give is_high_velocity=1."""
        sample_raw["transaction_velocity_5min"] = 10
        result = extractor.transform(sample_raw)
        names = extractor.get_feature_names()
        hv_idx = names.index("is_high_velocity")
        assert result[0, hv_idx] == 1.0

    def test_is_new_device_derived(self, extractor):
        """If is_new_device is absent, it should be derived from device_age_days."""
        raw = {
            "amount": 1000.0,
            "payment_method": "Card",
            "delivery_distance_km": 10.0,
            "customer_tenure_days": 100,
            "order_item_category": "Grocery",
            "payment_failure_history": 0,
            "is_foreign_ip": False,
            "hours_since_last_order": 24.0,
            "transaction_velocity_5min": 1,
            "device_age_days": 3,  # < 7 → is_new_device should be True
            "attempted_payment_method_changes": 0,
        }
        # Don't include is_new_device — it should be derived
        result = extractor.transform(raw)
        names = extractor.get_feature_names()
        new_dev_idx = names.index("is_new_device")
        assert result[0, new_dev_idx] == 1.0

    def test_missing_value_handled(self, extractor, sample_raw):
        """Missing numerical value should be imputed (no crash)."""
        sample_raw["amount"] = None
        # Should not raise
        result = extractor.transform(sample_raw)
        assert result.shape[1] == len(extractor.get_feature_names())

    def test_all_features_finite(self, extractor, sample_raw):
        """All output features should be finite numbers."""
        result = extractor.transform(sample_raw)
        assert np.all(np.isfinite(result))
