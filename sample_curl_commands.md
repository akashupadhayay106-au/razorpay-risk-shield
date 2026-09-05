# 🧪 Sample curl Commands — RiskShield AI API

Quick reference for testing the FastAPI fraud-risk endpoint locally.

---

## ▶️ 1. Run the API locally

```bash
cd razorpay-risk-shield

# Ensure the model is trained first (unless already committed in models/)
python src/train_model.py

# Start the server
uvicorn api.main:app --reload --host 0.0.0.0 --port 8000
```

Interactive docs (Swagger): <http://localhost:8000/docs>

---

## 📡 2. Health check

```bash
curl http://127.0.0.1:8000/health
```

**Response**

```json
{"status":"ok","model_loaded":true,"timestamp":"2026-09-05T16:33:14+00:00"}
```

---

## 💥 3. Assess a transaction (the main endpoint)

```bash
curl -X POST "http://127.0.0.1:8000/assess_risk" \
  -H "Content-Type: application/json" \
  -d "{\"transaction_id\":\"TXN_999\",\"merchant_id\":42,\"amount\":45000,\"payment_method\":\"UPI\",\"delivery_distance_km\":350,\"customer_tenure_days\":2,\"order_item_category\":\"Electronics\",\"payment_failure_history\":3,\"is_foreign_ip\":true,\"hours_since_last_order\":0.5,\"transaction_velocity_5min\":8,\"device_age_days\":1,\"attempted_payment_method_changes\":0}"
```

**Request JSON (pretty)**

```json
{
  "transaction_id": "TXN_999",
  "merchant_id": 42,
  "amount": 45000,
  "payment_method": "UPI",
  "delivery_distance_km": 350,
  "customer_tenure_days": 2,
  "order_item_category": "Electronics",
  "payment_failure_history": 3,
  "is_foreign_ip": true,
  "hours_since_last_order": 0.5,
  "transaction_velocity_5min": 8,
  "device_age_days": 1,
  "attempted_payment_method_changes": 0
}
```

**Expected response (ML + SHAP path, after warm-up)**

```json
{
  "transaction_id": "TXN_999",
  "risk_score": 0.07,
  "action": "APPROVE",
  "explanation": [
    "amount increased risk by 7.9%",
    "is_foreign_ip increased risk by 4.3%"
  ],
  "latency_ms": 9.41,
  "fallback_used": false,
  "timestamp": "2026-09-05T16:33:49+00:00"
}
```

> ℹ️ **Cold start behaviour** — a merchant with fewer than 5 observed
> transactions is scored by interpretable **rules** (`fallback_used: true`).
> Send ~5 warm-up requests first to hit the trained **XGBoost + SHAP** path
> shown above and get per-feature explanations.

---

## 📊 4. Low-risk transaction (APROVE)

```bash
curl -X POST "http://127.0.0.1:8000/assess_risk" \
  -H "Content-Type: application/json" \
  -d "{\"transaction_id\":\"TXN_LOW\",\"merchant_id\":7,\"amount\":299,\"payment_method\":\"UPI\",\"delivery_distance_km\":3,\"customer_tenure_days\":800,\"order_item_category\":\"Grocery\",\"payment_failure_history\":0,\"is_foreign_ip\":false,\"hours_since_last_order\":24,\"transaction_velocity_5min\":1,\"device_age_days\":900,\"attempted_payment_method_changes\":0}"
```

---

## ⚠️ 5. Invalid request (expect 422)

```bash
curl -X POST "http://127.0.0.1:8000/assess_risk" \
  -H "Content-Type: application/json" \
  -d "{\"transaction_id\":\"TX_BAD\",\"merchant_id\":1,\"amount\":-100,\"payment_method\":\"UPI\",\"delivery_distance_km\":1,\"customer_tenure_days\":1,\"order_item_category\":\"Grocery\",\"payment_failure_history\":0,\"is_foreign_ip\":false,\"hours_since_last_order\":1,\"transaction_velocity_5min\":0,\"device_age_days\":1,\"attempted_payment_method_changes\":0}"
```

**Response:** HTTP `422 Unprocessable Entity` (validation error from Pydantic).

---

## 📈 6. Model metrics

```bash
curl http://127.0.0.1:8000/metrics
```

**Response**

```json
{
  "precision": 0.129,
  "recall": 0.3596,
  "f1_score": 0.1899,
  "fpr": 0.205,
  "roc_auc": 0.5881,
  "total_cost_baseline": 658000,
  "total_cost_ml": 822000,
  "total_cost_saved": -164000
}
```

---

## 🧾 7. Dashboard (separate terminal)

```bash
streamlit run dashboard/app.py
```

Dashboard: <http://localhost:8501>