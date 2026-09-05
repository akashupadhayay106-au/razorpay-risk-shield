"""
Feature engineering for the Razorpay Risk Shield fraud-detection pipeline.

The FeatureExtractor converts raw transaction dicts / DataFrames into a
fixed-length, model-ready feature vector.  It:

  * one-hot encodes `payment_method` and `order_item_category`
  * derives `amount_per_tenure`, `is_high_velocity` and `is_new_device`
  * median-imputes missing numerical values
  * (optionally) scales the matrix with a `StandardScaler` that is fitted
    during training and re-loaded at inference time.
"""

from __future__ import annotations

from typing import List, Optional

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

# ---------------------------------------------------------------------------
# Canonical column groups (shared by training, prediction and tests)
# ---------------------------------------------------------------------------

NUMERICAL_COLS: List[str] = [
    "amount",
    "delivery_distance_km",
    "customer_tenure_days",
    "payment_failure_history",
    "hours_since_last_order",
    "transaction_velocity_5min",
    "device_age_days",
    "attempted_payment_method_changes",
]

BOOLEAN_COLS: List[str] = ["is_foreign_ip"]

CATEGORICAL_COLS: dict = {
    "payment_method": ["UPI", "Card", "NetBanking", "Wallet"],
    "order_item_category": [
        "Electronics",
        "Apparel",
        "Grocery",
        "Furniture",
        "Digital_Goods",
    ],
}

DERIVED_FEATURE_NAMES: List[str] = [
    "amount_per_tenure",
    "is_high_velocity",
    "is_new_device",
]

ALL_FEATURE_NAMES: List[str] = (
    NUMERICAL_COLS
    + BOOLEAN_COLS
    + [f"{col}_{cat}" for col, cats in CATEGORICAL_COLS.items() for cat in cats]
    + DERIVED_FEATURE_NAMES
)

NEW_DEVICE_THRESHOLD_DAYS = 7
HIGH_VELOCITY_THRESHOLD = 5


class FeatureExtractor:
    """
    Converts raw transactions into a scaled numerical feature vector.

    Parameters
    ----------
    feature_names : list[str], optional
        Ordered feature list.  When omitted, the canonical layout is built.
    scaler : StandardScaler, optional
        A pre-fitted scaler.  When provided, `transform()` outputs scaled
        features; otherwise raw (only median-imputed) features are returned.
    """

    def __init__(
        self,
        feature_names: Optional[List[str]] = None,
        scaler: Optional[StandardScaler] = None,
    ) -> None:
        self.scaler = scaler
        self.medians: dict = {}
        if feature_names is not None:
            self.feature_names: List[str] = list(feature_names)
        else:
            self.build_feature_names()

    # ----------------------------------------------------------------------
    # Feature naming
    # ----------------------------------------------------------------------

    def build_feature_names(self) -> List[str]:
        """Build the canonical, ordered list of feature names."""
        self.feature_names = list(ALL_FEATURE_NAMES)
        return list(self.feature_names)

    def get_feature_names(self) -> List[str]:
        """Return the ordered list of feature names."""
        return list(self.feature_names)

    # ----------------------------------------------------------------------
    # Fitting
    # ----------------------------------------------------------------------

    def fit_medians(self, df: pd.DataFrame) -> "FeatureExtractor":
        """Compute per-column medians for later missing-value imputation."""
        for col in NUMERICAL_COLS:
            if col not in df.columns:
                self.medians[col] = 0.0
                continue
            vals = pd.to_numeric(df[col], errors="coerce").dropna()
            median = float(vals.median()) if len(vals) else 0.0
            if not np.isfinite(median):
                median = 0.0
            self.medians[col] = median
        return self

    def fit(self, df: pd.DataFrame) -> "FeatureExtractor":
        """Median-impute + scale on a training DataFrame.

        Fits the median imputer and, when no scaler was injected, fits a new
        StandardScaler over the generated feature matrix.
        """
        self.fit_medians(df)
        matrix = self.transform_dataframe(df, scale=False)
        if self.scaler is None:
            self.scaler = StandardScaler()
            self.scaler.fit(matrix)
        return self

    # ----------------------------------------------------------------------
    # Row resolution
    # ----------------------------------------------------------------------

    def _resolve_row(self, raw: dict) -> dict:
        """Resolve raw fields (imputation + boolean coercion + derived features)."""
        r: dict = {}

        for col in NUMERICAL_COLS:
            v = raw.get(col)
            try:
                if v is None or (
                    isinstance(v, float) and (np.isnan(v) or np.isinf(v))
                ):
                    v = self.medians.get(col, 0.0)
                else:
                    v = float(v)
            except (TypeError, ValueError):
                v = self.medians.get(col, 0.0)
            r[col] = v

        fip = raw.get("is_foreign_ip", False)
        r["is_foreign_ip"] = (
            1.0
            if fip is True or str(fip).strip().lower() in ("1", "true", "yes")
            else 0.0
        )

        amount = r.get("amount", 0.0)
        tenure = r.get("customer_tenure_days", 0.0)
        r["amount_per_tenure"] = amount / (tenure + 1.0)

        velocity = r.get("transaction_velocity_5min", 0.0)
        r["is_high_velocity"] = 1.0 if velocity > HIGH_VELOCITY_THRESHOLD else 0.0

        if "is_new_device" in raw and raw.get("is_new_device") is not None:
            nd = raw["is_new_device"]
            r["is_new_device"] = (
                1.0 if nd is True or str(nd).strip().lower() in ("1", "true", "yes") else 0.0
            )
        else:
            r["is_new_device"] = (
                1.0 if r.get("device_age_days", 999) < NEW_DEVICE_THRESHOLD_DAYS else 0.0
            )

        for col in CATEGORICAL_COLS:
            value = raw.get(col, "")
            for c in CATEGORICAL_COLS[col]:
                r[f"{col}_{c}"] = 1.0 if value == c else 0.0

        return r

    # ----------------------------------------------------------------------
    # Transform
    # ----------------------------------------------------------------------

    def transform_dataframe(self, df: pd.DataFrame, scale: bool = True) -> np.ndarray:
        """Generate a feature matrix for an entire DataFrame."""
        out = np.zeros((len(df), len(self.feature_names)), dtype=float)
        for i, raw in enumerate(df.to_dict("records")):
            resolved = self._resolve_row(raw)
            out[i] = [float(resolved.get(name, 0.0)) for name in self.feature_names]
        if scale and self.scaler is not None:
            out = self.scaler.transform(out)
        return out

    def transform(self, raw_data) -> np.ndarray:
        """
        Convert a raw transaction (dict), a DataFrame, or an already-built
        feature vector into a (n, n_features) array.

        Missing values are median-imputed.  If a fitted scaler is available,
        the resulting array is standardized.
        """
        if isinstance(raw_data, pd.DataFrame):
            return self.transform_dataframe(raw_data)

        if isinstance(raw_data, np.ndarray):
            arr = np.asarray(raw_data, dtype=float)
            if arr.ndim == 1:
                arr = arr.reshape(1, -1)
            if self.scaler is not None:
                arr = self.scaler.transform(arr)
            return arr

        if isinstance(raw_data, dict):
            resolved = self._resolve_row(raw_data)
            vec = np.array(
                [float(resolved.get(name, 0.0)) for name in self.feature_names],
                dtype=float,
            ).reshape(1, -1)
            if self.scaler is not None:
                vec = self.scaler.transform(vec)
            return vec

        raise TypeError(f"Unsupported raw_data type: {type(raw_data)!r}")