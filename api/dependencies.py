"""
Dependency layer for the RiskShield API.

Loads model artifacts once at startup, sets up the SQLite audit trail and
keeps an in-memory per-merchant transaction counter used for cold-start
routing of brand-new merchants.
"""

from __future__ import annotations

import json
import logging
import os
import sys
from collections import defaultdict
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv
from sqlalchemy import Column, Float, Integer, String, Text, create_engine
from sqlalchemy.orm import Session, declarative_base, sessionmaker

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
load_dotenv(PROJECT_ROOT / ".env")

from src.features import FeatureExtractor  # noqa: E402
from src.predict import RiskPredictor  # noqa: E402
from src.utils import (  # noqa: E402
    get_feature_names_path,
    get_metrics_path,
    get_model_path,
    get_scaler_path,
)

logger = logging.getLogger("risk_shield.api.deps")

# ---------------------------------------------------------------------------
# Singleton state
# ---------------------------------------------------------------------------
_predictor: Optional[RiskPredictor] = None
_metrics: Optional[dict] = None

# Cold-start tracking: merchant_id -> number of transactions seen this process.
merchant_tx_count: defaultdict = defaultdict(int)


# ---------------------------------------------------------------------------
# SQLite audit trail
# ---------------------------------------------------------------------------
Base = declarative_base()


class AuditRecord(Base):
    """One row per risk assessment."""

    __tablename__ = "audit_trail"

    id = Column(Integer, primary_key=True, autoincrement=True)
    timestamp = Column(String, nullable=False)
    transaction_id = Column(String, nullable=False, index=True)
    merchant_id = Column(Integer, nullable=False)
    risk_score = Column(Float, nullable=False)
    action = Column(String, nullable=False)
    explanation = Column(Text, nullable=True)
    latency_ms = Column(Float, nullable=False)
    fallback_used = Column(Integer, nullable=False, default=0)


AUDIT_DB_PATH = PROJECT_ROOT / "audit.db"
engine = create_engine(f"sqlite:///{AUDIT_DB_PATH}", echo=False)
SessionLocal = sessionmaker(bind=engine)

Base.metadata.create_all(engine)


def get_db() -> Session:
    """Yield a SQLAlchemy session bound to the audit database."""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------
def load_model_artifacts() -> None:
    """Load (or reload) the model, scaler, feature names and metrics."""
    global _predictor, _metrics

    model_path = get_model_path()
    scaler_path = get_scaler_path()
    feature_names_path = get_feature_names_path()
    metrics_path = get_metrics_path()

    if not model_path.exists():
        logger.warning(
            "Model not found at %s — predictions will use rule fallback.", model_path
        )
        return

    extractor_path = PROJECT_ROOT / "models" / "extractor.pkl"

    _predictor = RiskPredictor(
        model_path=str(model_path),
        scaler_path=str(scaler_path),
        feature_names_path=str(feature_names_path),
        extractor_path=str(extractor_path) if extractor_path.exists() else None,
    )

    if metrics_path.exists():
        with open(metrics_path) as f:
            _metrics = json.load(f)
        logger.info("Model metrics loaded.")

    logger.info("All model artifacts loaded successfully.")


def get_predictor() -> Optional[RiskPredictor]:
    """Return the loaded predictor (None if the model is missing)."""
    return _predictor


def get_extractor() -> FeatureExtractor:
    """Return a feature extractor (bare, for fallback/scoring)."""
    if _predictor is not None:
        return _predictor.extractor
    return FeatureExtractor()


def get_metrics() -> Optional[dict]:
    """Return the cached model-metrics dict."""
    return _metrics


def get_costs() -> tuple[int, int]:
    """Return (FP_COST, FN_COST) read from the environment with sensible defaults."""
    fp_cost = int(os.getenv("FP_COST", "500"))
    fn_cost = int(os.getenv("FN_COST", "2500"))
    return fp_cost, fn_cost