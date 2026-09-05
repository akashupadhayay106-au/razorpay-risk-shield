"""
Pydantic request / response models for the RiskShield API.
"""

from __future__ import annotations

from typing import List, Literal, Optional

from pydantic import BaseModel, Field


class TransactionRequest(BaseModel):
    """Incoming transaction to be assessed for fraud risk."""

    transaction_id: str = Field(..., description="Unique transaction identifier")
    merchant_id: int = Field(..., ge=1, description="Merchant identifier")
    amount: float = Field(..., gt=0, description="Transaction amount in INR")
    payment_method: Literal["UPI", "Card", "NetBanking", "Wallet"] = Field(
        ..., description="Payment method"
    )
    delivery_distance_km: float = Field(0.0, ge=0, description="Delivery distance in km")
    customer_tenure_days: int = Field(0, ge=0, description="Days since customer signup")
    order_item_category: Literal[
        "Electronics", "Apparel", "Grocery", "Furniture", "Digital_Goods"
    ] = Field(..., description="Product category")
    payment_failure_history: int = Field(0, ge=0, description="Past payment failures")
    is_foreign_ip: bool = Field(False, description="Is the request from a foreign IP?")
    hours_since_last_order: float = Field(0.0, ge=0, description="Hours since last order")
    transaction_velocity_5min: int = Field(0, ge=0, description="Transactions in last 5 min")
    device_age_days: int = Field(0, ge=0, description="Device age in days")
    attempted_payment_method_changes: int = Field(
        0, ge=0, description="Payment method change attempts"
    )

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "transaction_id": "TXN_12345",
                    "merchant_id": 42,
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
                    "attempted_payment_method_changes": 0,
                }
            ]
        }
    }


class RiskResponse(BaseModel):
    """Risk assessment result returned to the caller."""

    transaction_id: str
    risk_score: float = Field(..., ge=0, le=1)
    action: str = Field(..., description="APPROVE | MANUAL_REVIEW | DECLINE")
    explanation: List[str] = Field(default_factory=list)
    latency_ms: float
    fallback_used: bool = False
    timestamp: str


class HealthResponse(BaseModel):
    """Health-check payload."""

    status: str = "ok"
    model_loaded: bool = True
    timestamp: str


class MetricsResponse(BaseModel):
    """Model performance metrics."""

    precision: float
    recall: float
    f1_score: float
    fpr: float
    roc_auc: float
    total_cost_ml: Optional[float] = None
    total_cost_baseline: Optional[float] = None
    total_cost_saved: Optional[float] = None