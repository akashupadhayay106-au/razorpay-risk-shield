"""
RiskPredictor — wraps a trained XGBoost model + SHAP TreeExplainer.

The SHAP TreeExplainer is cached at the *class* level so it is built once per
process (TreeExplainer is far faster than KernelExplainer, keeping single-row
latency comfortably under 150 ms).
"""

from __future__ import annotations

import os
from typing import List, Optional, Tuple

import joblib
import numpy as np
import shap

from src.features import FeatureExtractor
from src.utils import PROJECT_ROOT


class RiskPredictor:
    """Loads model artifacts and exposes fraud-probability + SHAP explanations."""

    # Class-level cache: created lazily on first instantiation.
    _explainer_cache = {}

    def __init__(
        self,
        model_path: str = "models/risk_model.pkl",
        scaler_path: str = "models/scaler.pkl",
        feature_names_path: str = "models/feature_names.pkl",
        extractor_path: Optional[str] = None,
    ) -> None:
        self.model = joblib.load(model_path)
        self.scaler = joblib.load(scaler_path)
        self.feature_names: List[str] = list(joblib.load(feature_names_path))

        # Rebuild the FeatureExtractor used at inference time.
        if extractor_path and os.path.exists(extractor_path):
            extractor = joblib.load(extractor_path)
            extractor.feature_names = self.feature_names
        else:
            extractor = FeatureExtractor(feature_names=self.feature_names, scaler=None)
        extractor.scaler = self.scaler
        self.extractor = extractor

        # A fitted XGBoost tree model exposes predict_proba; TreeExplainer uses
        # the tree structure directly for instant, exact SHAP values.
        if "explainer" not in RiskPredictor._explainer_cache:
            RiskPredictor._explainer_cache["explainer"] = shap.TreeExplainer(self.model)
        self.explainer = RiskPredictor._explainer_cache["explainer"]

    def _to_vector(self, features) -> np.ndarray:
        """Normalise the feature input into a (1, n_features) scaled array."""
        if isinstance(features, np.ndarray):
            arr = np.asarray(features, dtype=float)
            if arr.ndim == 1:
                arr = arr.reshape(1, -1)
            if self.scaler is not None:
                arr = self.scaler.transform(arr)
            return arr
        return self.extractor.transform(features)

    def predict(self, features) -> float:
        """Return the fraud probability for a single transaction."""
        vector = self._to_vector(features)
        prob = float(self.model.predict_proba(vector)[0, 1])
        return prob

    def predict_with_explanation(self, features) -> Tuple[float, List[str]]:
        """
        Return (fraud_probability, top-2 risk-increasing explanations).

        Explanations are SHAP contributions expressed as a percentage of the
        total absolute contribution across all features.
        """
        vector = self._to_vector(features)
        prob = float(self.model.predict_proba(vector)[0, 1])

        shap_values = self.explainer.shap_values(vector)

        # Normalise SHAP output shapes across XGBoost versions.
        sv = shap_values
        if isinstance(sv, list):
            sv = sv[-1]
        if len(np.asarray(sv).shape) == 3:
            sv = np.asarray(sv)[0, :, 1]
        else:
            sv = np.asarray(sv)[0]

        contributions = dict(zip(self.feature_names, sv))
        total_abs = float(np.sum(np.abs(sv))) or 1e-9

        # Sort by contribution (largest risk-increase first).
        ordered = sorted(contributions.items(), key=lambda kv: kv[1], reverse=True)

        explanations: List[str] = []
        for name, val in ordered:
            if val > 0:
                pct = (val / total_abs) * 100.0
                explanations.append(f"{name} increased risk by {pct:.1f}%")
            if len(explanations) == 2:
                break
        return prob, explanations


def default_predictor() -> RiskPredictor:
    """Build a RiskPredictor from the default model directory paths."""
    models_dir = PROJECT_ROOT / "models"
    return RiskPredictor(
        model_path=str(models_dir / "risk_model.pkl"),
        scaler_path=str(models_dir / "scaler.pkl"),
        feature_names_path=str(models_dir / "feature_names.pkl"),
        extractor_path=str(models_dir / "extractor.pkl"),
    )