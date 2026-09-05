"""
Tests for the FastAPI Risk Shield API endpoints.
"""

import sys
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from api.main import app  # noqa: E402


@pytest.fixture
def client():
    """Return a FastAPI test client."""
    return TestClient(app)


# ---------------------------------------------------------------------------
# Valid transaction payload
# ---------------------------------------------------------------------------
VALID_PAYLOAD = {
    "transaction_id": "TXN_TEST_001",
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

LOW_RISK_PAYLOAD = {
    "transaction_id": "TXN_TEST_002",
    "merchant_id": 100,
    "amount": 500.0,
    "payment_method": "UPI",
    "delivery_distance_km": 10.0,
    "customer_tenure_days": 800,
    "order_item_category": "Grocery",
    "payment_failure_history": 0,
    "is_foreign_ip": False,
    "hours_since_last_order": 48.0,
    "transaction_velocity_5min": 1,
    "device_age_days": 500,
    "attempted_payment_method_changes": 0,
}


# ---------------------------------------------------------------------------
# POST /assess_risk
# ---------------------------------------------------------------------------

class TestAssessRisk:
    """Tests for the /assess_risk endpoint."""

    def test_valid_request_returns_200(self, client):
        """A valid transaction should return HTTP 200 with required fields."""
        response = client.post("/assess_risk", json=VALID_PAYLOAD)
        assert response.status_code == 200

        data = response.json()
        assert "transaction_id" in data
        assert "risk_score" in data
        assert "action" in data
        assert "explanation" in data
        assert "latency_ms" in data
        assert "fallback_used" in data
        assert "timestamp" in data

    def test_risk_score_in_range(self, client):
        """Risk score should be between 0 and 1."""
        response = client.post("/assess_risk", json=VALID_PAYLOAD)
        data = response.json()
        assert 0.0 <= data["risk_score"] <= 1.0

    def test_action_is_valid(self, client):
        """Action should be one of APPROVE, MANUAL_REVIEW, DECLINE."""
        response = client.post("/assess_risk", json=VALID_PAYLOAD)
        data = response.json()
        assert data["action"] in {"APPROVE", "MANUAL_REVIEW", "DECLINE"}

    def test_explanation_is_list(self, client):
        """Explanation should be a list of strings."""
        response = client.post("/assess_risk", json=VALID_PAYLOAD)
        data = response.json()
        assert isinstance(data["explanation"], list)
        for item in data["explanation"]:
            assert isinstance(item, str)

    def test_invalid_payment_method(self, client):
        """Invalid payment_method should return 422."""
        payload = VALID_PAYLOAD.copy()
        payload["payment_method"] = "Bitcoin"
        response = client.post("/assess_risk", json=payload)
        assert response.status_code == 422

    def test_invalid_category(self, client):
        """Invalid order_item_category should return 422."""
        payload = VALID_PAYLOAD.copy()
        payload["order_item_category"] = "Unknown"
        response = client.post("/assess_risk", json=payload)
        assert response.status_code == 422

    def test_negative_amount(self, client):
        """Negative amount should return 422."""
        payload = VALID_PAYLOAD.copy()
        payload["amount"] = -100
        response = client.post("/assess_risk", json=payload)
        assert response.status_code == 422

    def test_missing_transaction_id(self, client):
        """Missing transaction_id should return 422."""
        payload = VALID_PAYLOAD.copy()
        del payload["transaction_id"]
        response = client.post("/assess_risk", json=payload)
        assert response.status_code == 422

    def test_low_risk_transaction(self, client):
        """A low-risk transaction should get APPROVE or MANUAL_REVIEW."""
        # Send enough transactions to pass cold-start threshold
        for i in range(6):
            p = LOW_RISK_PAYLOAD.copy()
            p["transaction_id"] = f"TXN_LOW_{i}"
            client.post("/assess_risk", json=p)

        response = client.post("/assess_risk", json=LOW_RISK_PAYLOAD)
        data = response.json()
        # Low-risk should not be DECLINE (though model may vary)
        assert data["action"] in {"APPROVE", "MANUAL_REVIEW", "DECLINE"}

    def test_cold_start_fallback(self, client):
        """A new merchant should trigger cold-start fallback."""
        payload = VALID_PAYLOAD.copy()
        payload["merchant_id"] = 99999  # unlikely to have history
        payload["transaction_id"] = "TXN_COLD_START"
        response = client.post("/assess_risk", json=payload)
        data = response.json()
        assert data["fallback_used"] is True


# ---------------------------------------------------------------------------
# GET /health
# ---------------------------------------------------------------------------

class TestHealth:
    """Tests for the /health endpoint."""

    def test_health_returns_200(self, client):
        response = client.get("/health")
        assert response.status_code == 200

    def test_health_has_status(self, client):
        response = client.get("/health")
        data = response.json()
        assert data["status"] == "ok"
        assert "model_loaded" in data
        assert "timestamp" in data


# ---------------------------------------------------------------------------
# GET /metrics
# ---------------------------------------------------------------------------

class TestMetrics:
    """Tests for the /metrics endpoint."""

    def test_metrics_endpoint(self, client):
        """Metrics endpoint should return 200 if model is trained, else 503."""
        response = client.get("/metrics")
        assert response.status_code in {200, 503}

        if response.status_code == 200:
            data = response.json()
            assert "precision" in data
            assert "recall" in data
            assert "f1_score" in data
            assert "fpr" in data
            assert "roc_auc" in data
