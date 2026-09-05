"""
Streamlit monitoring dashboard for Razorpay Risk Shield.

Displays:
  * live model metrics cards (Precision, Recall, F1, FPR, Cost Saved)
  * an interactive confusion-matrix + cost heatmap with a threshold slider
    (0.3 - 0.9) that recalculates in real time
  * the last 50 transactions from the SQLite audit trail
  * a risk-score histogram comparing approved vs. declined transactions
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import seaborn as sns  # noqa: E402
import streamlit as st  # noqa: E402
from sklearn.metrics import confusion_matrix  # noqa: E402
from sklearn.model_selection import train_test_split  # noqa: E402
from sqlalchemy import create_engine, text  # noqa: E402

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.features import FeatureExtractor  # noqa: E402

MODELS_DIR = PROJECT_ROOT / "models"
METRICS_PATH = MODELS_DIR / "model_metrics.json"
AUDIT_DB_PATH = PROJECT_ROOT / "audit.db"
DATA_PATH = PROJECT_ROOT / "data" / "transactions.csv"

RANDOM_STATE = 42
TEST_SIZE = 0.15

st.set_page_config(page_title="RiskShield AI - Dashboard", page_icon="🛡️", layout="wide")
st.title("🛡️ RiskShield AI — Fraud Monitoring Dashboard")
st.markdown("---")


# ---------------------------------------------------------------------------
# Loaders
# ---------------------------------------------------------------------------
@st.cache_data(ttl=30)
def load_metrics() -> dict:
    if METRICS_PATH.exists():
        with open(METRICS_PATH) as f:
            return json.load(f)
    return {}


@st.cache_data(ttl=10)
def load_audit_log(limit: int = 50) -> pd.DataFrame:
    if not AUDIT_DB_PATH.exists():
        return pd.DataFrame()
    engine = create_engine(f"sqlite:///{AUDIT_DB_PATH}")
    query = text("SELECT * FROM audit_trail ORDER BY id DESC LIMIT :lim")
    try:
        with engine.connect() as conn:
            return pd.read_sql(query, conn, params={"lim": limit})
    except Exception:
        return pd.DataFrame()


@st.cache_data(ttl=60)
def load_test_predictions():
    """Recreate the held-out test predictions (mandatory threshold analysis)."""
    model_path = MODELS_DIR / "risk_model.pkl"
    scaler_path = MODELS_DIR / "scaler.pkl"
    feature_names_path = MODELS_DIR / "feature_names.pkl"

    if not all(p.exists() for p in [model_path, scaler_path, feature_names_path, DATA_PATH]):
        return None, None

    import joblib

    model = joblib.load(model_path)
    scaler = joblib.load(scaler_path)
    feature_names = list(joblib.load(feature_names_path))

    df = pd.read_csv(DATA_PATH)
    y = df["is_fraud"].values
    X = df.drop(columns=["is_fraud"])

    # Same stratified split used during training (seed + size).
    _, X_test, _, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, stratify=y, random_state=RANDOM_STATE
    )

    extractor = FeatureExtractor(feature_names=feature_names, scaler=scaler)
    X_test_matrix = extractor.transform(X_test)
    y_proba = model.predict_proba(X_test_matrix)[:, 1]
    return y_test, y_proba


# ---------------------------------------------------------------------------
# Section 1: Metric cards
# ---------------------------------------------------------------------------
st.header("📊 Model Performance")
metrics = load_metrics()

if metrics:
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Precision", f"{metrics.get('precision', 0):.3f}")
    c2.metric("Recall", f"{metrics.get('recall', 0):.3f}")
    c3.metric("F1 Score", f"{metrics.get('f1_score', 0):.3f}")
    c4.metric("False Positive Rate", f"{metrics.get('fpr', 0):.3f}")
    c5.metric("Total Cost Saved", f"₹{metrics.get('total_cost_saved', 0):,.0f}")
    st.caption(
        f"ROC-AUC {metrics.get('roc_auc', 0):.3f} • "
        f"Baseline cost ₹{metrics.get('total_cost_baseline', 0):,.0f} → "
        f"ML cost ₹{metrics.get('total_cost_ml', 0):,.0f}"
    )
else:
    st.warning("No metrics found — train the model first (`python src/train_model.py`).")

st.markdown("---")

# ---------------------------------------------------------------------------
# Section 2: Interactive threshold + confusion matrix / cost
# ---------------------------------------------------------------------------
st.header("🎚️ Threshold Tuning & Cost Analysis")

y_test, y_proba = load_test_predictions()

if y_test is not None and y_proba is not None:
    default_threshold = float(metrics.get("threshold", 0.5)) if metrics else 0.5
    threshold = st.slider(
        "Decision Threshold",
        min_value=0.30,
        max_value=0.90,
        value=default_threshold,
        step=0.01,
    )

    fp_cost = int(metrics.get("fp_cost", 500))
    fn_cost = int(metrics.get("fn_cost", 2500))

    y_pred = (y_proba >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
    total_cost = fp * fp_cost + fn * fn_cost
    baseline = int(metrics.get("total_cost_baseline", 0))
    savings = baseline - total_cost

    col_cm, col_info = st.columns([2, 1])

    with col_cm:
        cm = np.array([[tn, fp], [fn, tp]])
        fig, ax = plt.subplots(figsize=(6, 4.5))
        sns.heatmap(
            cm,
            annot=True,
            fmt="d",
            cmap="RdYlGn_r",
            cbar=False,
            xticklabels=["Not Fraud", "Fraud"],
            yticklabels=["Not Fraud", "Fraud"],
            ax=ax,
        )
        ax.set_xlabel("Predicted")
        ax.set_ylabel("Actual")
        ax.set_title(f"Confusion Matrix (threshold = {threshold:.2f})")
        st.pyplot(fig)
        plt.close(fig)

    with col_info:
        st.subheader("Cost Breakdown")
        st.write(f"**True Negatives:** {tn}")
        st.write(f"**False Positives:** {fp} → cost ₹{fp * fp_cost:,.0f}")
        st.write(f"**False Negatives:** {fn} → cost ₹{fn * fn_cost:,.0f}")
        st.write(f"**True Positives:** {tp}")
        st.metric("Total ML Cost", f"₹{total_cost:,.0f}")
        st.metric("Cost Saved vs. Baseline", f"₹{savings:,.0f}")

    # Precision / recall at the selected threshold.
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    st.info(
        f"At threshold {threshold:.2f} → Precision **{precision:.3f}**, "
        f"Recall **{recall:.3f}**, F1 **{f1:.3f}**."
    )
else:
    st.info("Train the model to see threshold analysis.")

st.markdown("---")

# ---------------------------------------------------------------------------
# Section 3: Recent transaction audit log
# ---------------------------------------------------------------------------
st.header("📋 Transaction Audit Log (Last 50)")

audit_df = load_audit_log(limit=50)

if not audit_df.empty:
    display_cols = [
        "timestamp",
        "transaction_id",
        "merchant_id",
        "risk_score",
        "action",
        "explanation",
        "latency_ms",
        "fallback_used",
    ]
    available = [c for c in display_cols if c in audit_df.columns]
    st.dataframe(audit_df[available], width="stretch")
else:
    st.info("No transactions logged yet. Send a request to the API first.")

st.markdown("---")

# ---------------------------------------------------------------------------
# Section 4: Risk-score distribution histogram
# ---------------------------------------------------------------------------
st.header("📈 Risk Score Distribution")

if y_test is not None and y_proba is not None:
    legit = y_proba[y_test == 0]
    fraud = y_proba[y_test == 1]

    fig2, ax2 = plt.subplots(figsize=(10, 5))
    ax2.hist(legit, bins=50, alpha=0.6, color="#2ecc71", label=f"Legit (n={len(legit)})", density=True)
    ax2.hist(fraud, bins=50, alpha=0.6, color="#e74c3c", label=f"Fraud (n={len(fraud)})", density=True)
    ax2.set_xlabel("Risk Score")
    ax2.set_ylabel("Density")
    ax2.set_title("Risk Score Distribution (Test Set)")
    ax2.legend()
    st.pyplot(fig2)
    plt.close(fig2)
elif not audit_df.empty and "risk_score" in audit_df.columns:
    fig2, ax2 = plt.subplots(figsize=(10, 5))
    ax2.hist(audit_df["risk_score"], bins=30, color="#3498db", alpha=0.7, edgecolor="white")
    ax2.set_xlabel("Risk Score")
    ax2.set_ylabel("Count")
    ax2.set_title("Risk Score Distribution (API Transactions)")
    st.pyplot(fig2)
    plt.close(fig2)
else:
    st.info("No data available for risk distribution.")

st.markdown("---")
st.caption("RiskShield AI v1.0 • Cost-aware fraud detection with SHAP explainability")