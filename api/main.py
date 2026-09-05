"""
FastAPI application for the Razorpay Risk Shield fraud-detection API.

Endpoints
---------
POST /assess_risk   Score a transaction (ML + SHAP, with cold-start and
                    circuit-breaker fallbacks).
GET  /health        Health check.
GET  /metrics       Trained model performance metrics.
"""

from __future__ import annotations

import json
import logging
import time
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Tuple

from fastapi import Depends, FastAPI, HTTPException
from sqlalchemy.orm import Session

import sys

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from api.dependencies import (  # noqa: E402
    AuditRecord,
    get_costs,
    get_db,
    get_metrics,
    get_predictor,
    load_model_artifacts,
    merchant_tx_count,
)
from api.models import HealthResponse, RiskResponse, TransactionRequest  # noqa: E402
from src.utils import ensure_dirs, log_fallback_alert  # noqa: E402

logger = logging.getLogger("risk_shield.api")
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
PREDICTION_TIMEOUT_MS = 300
COLD_START_TX_THRESHOLD = 5
DECLINE_THRESHOLD = 0.75
REVIEW_THRESHOLD = 0.50

FP_COST, FN_COST = get_costs()


# ---------------------------------------------------------------------------
# Lifespan: load artifacts once at startup
# ---------------------------------------------------------------------------
@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Starting Risk Shield API...")
    ensure_dirs()
    load_model_artifacts()
    yield
    logger.info("Shutting down Risk Shield API.")


app = FastAPI(title="Razorpay Risk Shield", version="1.0.0", lifespan=lifespan)


# ---------------------------------------------------------------------------
# Rule helpers (cold-start + circuit-breaker fallback)
# ---------------------------------------------------------------------------
def rule_based_decision(data: Dict[str, Any]) -> Tuple[float, str, List[str]]:
    """Score with simple explainable rules.

    Cold-start rule: approve iff amount < Rs20,000 AND not a foreign IP,
    otherwise MANUAL_REVIEW. Returns (risk_score, action, explanations).
    """
    amount = data.get("amount", 0.0)
    foreign = bool(data.get("is_foreign_ip", False))

    if amount < 20_000 and not foreign:
        return 0.05, "APPROVE", ["Low risk: amount under Rs20k and domestic IP"]

    risk = 0.55
    explanations: List[str] = []
    if amount >= 20_000:
        risk = min(risk + 0.15, 0.8)
        explanations.append("High transaction amount")
    if foreign:
        risk = min(risk + 0.15, 0.8)
        explanations.append("Foreign IP detected")
    return float(risk), "MANUAL_REVIEW", explanations or ["Manual review required"]


# ---------------------------------------------------------------------------
# POST /assess_risk
# ---------------------------------------------------------------------------
@app.post("/assess_risk", response_model=RiskResponse)
def assess_risk(
    txn: TransactionRequest,
    db: Session = Depends(get_db),
) -> RiskResponse:
    start = time.perf_counter()
    now = datetime.now(timezone.utc).isoformat()
    raw = txn.model_dump()

    # 1. Track merchant transaction count (cold-start monitoring).
    merchant_tx_count[txn.merchant_id] += 1

    risk_score: float = 0.0
    action: str = "APPROVE"
    explanation: List[str] = []
    fallback_used: bool = False

    # 2. Cold start: merchants with < 5 transactions use rules.
    if merchant_tx_count[txn.merchant_id] < COLD_START_TX_THRESHOLD:
        risk_score, action, explanation = rule_based_decision(raw)
        fallback_used = True
    else:
        predictor = get_predictor()

        if predictor is None:
            # Model not loaded (e.g. not trained yet) -> graceful fallback.
            risk_score, action, explanation = rule_based_decision(raw)
            fallback_used = True
            log_fallback_alert(txn.transaction_id, "Model not loaded", action, risk_score)
        else:
            # 3-5. ML prediction + SHAP with a circuit breaker.
            pred_start = time.perf_counter()
            try:
                prob, explanation = predictor.predict_with_explanation(raw)
                pred_elapsed_ms = (time.perf_counter() - pred_start) * 1000.0
                risk_score = prob

                if pred_elapsed_ms > PREDICTION_TIMEOUT_MS:
                    # Circuit breaker: prediction too slow -> fallback.
                    logger.warning(
                        "Prediction took %.1f ms (> %d ms) - falling back to rules.",
                        pred_elapsed_ms,
                        PREDICTION_TIMEOUT_MS,
                    )
                    risk_score, action, explanation = rule_based_decision(raw)
                    fallback_used = True
                    log_fallback_alert(
                        txn.transaction_id,
                        f"Prediction timeout ({pred_elapsed_ms:.0f} ms)",
                        action,
                        risk_score,
                    )
                else:
                    # 6. Decision thresholds.
                    if prob > DECLINE_THRESHOLD:
                        action = "DECLINE"
                    elif prob > REVIEW_THRESHOLD:
                        action = "MANUAL_REVIEW"
                    else:
                        action = "APPROVE"
            except Exception as exc:  # noqa: BLE001
                logger.exception("Model prediction failed: %s", exc)
                risk_score, action, explanation = rule_based_decision(raw)
                fallback_used = True
                log_fallback_alert(txn.transaction_id, str(exc), action, risk_score)

    latency_ms = (time.perf_counter() - start) * 1000.0

    # 8. Persist audit record.
    _audit_log(db, now, txn, risk_score, action, explanation, latency_ms, fallback_used)

    # 9. Return response.
    return RiskResponse(
        transaction_id=txn.transaction_id,
        risk_score=round(float(risk_score), 4),
        action=action,
        explanation=explanation,
        latency_ms=round(latency_ms, 2),
        fallback_used=fallback_used,
        timestamp=now,
    )


# ---------------------------------------------------------------------------
# GET /health
# ---------------------------------------------------------------------------
@app.get("/health", response_model=HealthResponse)
def health_check() -> HealthResponse:
    return HealthResponse(
        status="ok",
        model_loaded=get_predictor() is not None,
        timestamp=datetime.now(timezone.utc).isoformat(),
    )


# ---------------------------------------------------------------------------
# GET /metrics
# ---------------------------------------------------------------------------
@app.get("/metrics")
def model_metrics() -> Dict[str, Any]:
    metrics = get_metrics()
    if metrics is None:
        raise HTTPException(status_code=503, detail="Metrics not available - train the model first.")
    return metrics


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _audit_log(
    db: Session,
    timestamp: str,
    txn: TransactionRequest,
    risk_score: float,
    action: str,
    explanation: List[str],
    latency_ms: float,
    fallback_used: bool,
) -> None:
    try:
        record = AuditRecord(
            timestamp=timestamp,
            transaction_id=txn.transaction_id,
            merchant_id=txn.merchant_id,
            risk_score=round(float(risk_score), 4),
            action=action,
            explanation=json.dumps(explanation),
            latency_ms=round(latency_ms, 2),
            fallback_used=int(fallback_used),
        )
        db.add(record)
        db.commit()
    except Exception as exc:  # noqa: BLE001
        logger.error("Failed to write audit record: %s", exc)
        db.rollback()