"""
Utility helpers for the Razorpay Risk Shield project.
"""

import json
import logging
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional

PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODELS_DIR = PROJECT_ROOT / "models"
LOGS_DIR = PROJECT_ROOT / "logs"
DATA_DIR = PROJECT_ROOT / "data"


def ensure_dirs() -> None:
    """Create required runtime directories."""
    MODELS_DIR.mkdir(exist_ok=True)
    LOGS_DIR.mkdir(exist_ok=True)


def setup_logging(name: str = "risk_shield", level: str = "INFO") -> logging.Logger:
    """Return a configured logger."""
    logger = logging.getLogger(name)
    if logger.handlers:
        return logger
    logger.setLevel(getattr(logging, level.upper(), logging.INFO))
    fmt = logging.Formatter("%(asctime)s [%(levelname)s] %(name)s: %(message)s")
    ch = logging.StreamHandler()
    ch.setFormatter(fmt)
    logger.addHandler(ch)
    return logger


VALID_PAYMENT_METHODS = {"UPI", "Card", "NetBanking", "Wallet"}
VALID_CATEGORIES = {"Electronics", "Apparel", "Grocery", "Furniture", "Digital_Goods"}


def validate_transaction(data: Dict[str, Any]) -> Optional[str]:
    """Return None if valid, else an error message."""
    required = [
        "transaction_id", "merchant_id", "amount", "payment_method",
        "delivery_distance_km", "customer_tenure_days", "order_item_category",
        "payment_failure_history", "is_foreign_ip", "hours_since_last_order",
        "transaction_velocity_5min", "device_age_days",
        "attempted_payment_method_changes",
    ]
    for f in required:
        if f not in data:
            return f"Missing required field: {f}"
    if data["payment_method"] not in VALID_PAYMENT_METHODS:
        return f"Invalid payment_method: {data['payment_method']}"
    if data["order_item_category"] not in VALID_CATEGORIES:
        return f"Invalid order_item_category: {data['order_item_category']}"
    if data["amount"] <= 0:
        return "amount must be positive"
    return None


def log_fallback_alert(transaction_id: str, reason: str, action: str, risk_score: float) -> None:
    """Append a fallback alert to logs/fallback_alerts.json."""
    ensure_dirs()
    alert_file = LOGS_DIR / "fallback_alerts.json"
    alert = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "transaction_id": transaction_id,
        "reason": reason,
        "action": action,
        "risk_score": risk_score,
    }
    alerts = []
    if alert_file.exists():
        try:
            with open(alert_file, "r") as f:
                alerts = json.load(f)
        except (json.JSONDecodeError, IOError):
            alerts = []
    alerts.append(alert)
    with open(alert_file, "w") as f:
        json.dump(alerts, f, indent=2)


def get_model_path() -> Path:
    return PROJECT_ROOT / os.getenv("MODEL_PATH", "models/risk_model.pkl")

def get_scaler_path() -> Path:
    return PROJECT_ROOT / os.getenv("SCALER_PATH", "models/scaler.pkl")

def get_feature_names_path() -> Path:
    return PROJECT_ROOT / os.getenv("FEATURE_NAMES_PATH", "models/feature_names.pkl")

def get_metrics_path() -> Path:
    return MODELS_DIR / "model_metrics.json"
