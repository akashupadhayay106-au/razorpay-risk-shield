# 🛡️ RiskShield AI — Cost-Aware Fraud Detection for Payments

[![Python 3.14](https://img.shields.io/badge/Python-3.14-blue.svg)](https://www.python.org/)
[![Streamlit](https://img.shields.io/badge/Dashboard-Streamlit-FF4B4B.svg)](https://streamlit.io/cloud)
[![Docker](https://img.shields.io/badge/Docker-2496ED.svg?logo=docker&logoColor=white)](https://www.docker.com/)
[![Render](https://img.shields.io/badge/Deploy-Render-46E3B7.svg)](https://render.com/)
[![FastAPI](https://img.shields.io/badge/API-FastAPI-green.svg)](https://fastapi.tiangolo.com/)
[![XGBoost](https://img.shields.io/badge/XGBoost-3.x-orange.svg)](https://xgboost.readthedocs.io/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

**Track 02 — AI Risk Manager** · **Razorpay Buildathon 2026**

A production-ready, **cost-aware** fraud-risk system that scores Indian payment
transactions in real time, explains each decision with **SHAP**, and degrades
gracefully via cold-start and circuit-breaker fallbacks.

---

## 🖥️ Live Demo

> 📍 **Dashboard (Streamlit Cloud):** `https://kalicharan-razorpay-risk-shield.streamlit.app/`

Quick reference for the API + dashboard: see
[`sample_curl_commands.md`](sample_curl_commands.md).

---

## 🎯 Highlights

| Capability | Implementation |
|---|---|
| Fraud / risk detection | XGBoost with **cost-sensitive learning** (FN ₹2,500 vs FP ₹500) |
| Explainability | **SHAP TreeExplainer** — top-2 risk drivers per transaction (< 150 ms) |
| Real-time scoring | FastAPI `POST /assess_risk` with a 300 ms circuit breaker |
| Cold start | New merchants (< 5 txns) scored by rules, not a weak model |
| Audit trail | Every request/response persisted to SQLite (`audit.db`) |
| Monitoring | Streamlit dashboard with an interactive threshold slider |
| Deployment | Docker Compose (`api` :8000, `dashboard` :8501) + Render config |

---

## 🗂️ Project Structure

```
razorpay-risk-shield/
├── data/
│   └── generate_data.py          # 25K synthetic Indian-payment transactions
├── src/
│   ├── features.py               # Feature extraction (one-hot, scaling, derived)
│   ├── train_model.py            # Cost-sensitive XGBoost + GridSearchCV
│   ├── predict.py                # Prediction + SHAP explanations
│   └── utils.py                  # Logging, path helpers, fallback alerts
├── api/
│   ├── main.py                   # FastAPI app (assess_risk / health / metrics)
│   ├── models.py                 # Pydantic schemas
│   └── dependencies.py           # Model loading, SQLite audit, merchant counter
├── dashboard/
│   └── app.py                    # Streamlit monitoring + threshold tuning
├── models/                       # Trained artifacts (generated at training time)
├── logs/                         # Fallback alert logs
├── tests/                        # pytest suite
├── docker/Dockerfile
├── docker-compose.yml
├── requirements.txt
├── Makefile
├── render.yaml
└── README.md
```

---

## 🚀 Setup

> Runs on **Python 3.14** (Windows / Linux). We deliberately use base `uvicorn`
> (no `uvicorn[standard]`) because `uvloop` is not Windows-compatible.

### 1. Clone & install

```bash
git clone https://github.com/akashupadhayay106-au/razorpay-risk-shield.git
cd razorpay-risk-shield
pip install -r requirements.txt
```

### 2. Generate synthetic data

```bash
python data/generate_data.py
```

Creates `data/transactions.csv` — 25,000 transactions tailored to UPI / Cards /
NetBanking / Wallets with an ~8% fraud rate.

### 3. Train the model *(optional — a trained model is committed in `models/`)*

```bash
python src/train_model.py
```

Runs EDA → 70/15/15 stratified split → one-hot + scaling → **cost-sensitive
XGBoost tuned with GridSearchCV** (`n_estimators`, `max_depth`,
`learning_rate`, `subsample`, `cv=3`) → prints Precision / Recall / F1 / FPR /
ROC-AUC → simulates cost savings vs. a rule baseline. Artifacts are saved
under `models/`.

> ⚡ **Already trained?** `models/risk_model.pkl`, `scaler.pkl`,
> `feature_names.pkl` and `model_metrics.json` are committed, so you can skip
> this step and go straight to serving / the dashboard.

### 4. Start the API

```bash
uvicorn api.main:app --reload --host 0.0.0.0 --port 8000
```

Interactive docs: <http://localhost:8000/docs>

### 5. Start the dashboard

```bash
streamlit run dashboard/app.py
```

Dashboard: <http://localhost:8501>

### 6. Run the tests

```bash
pytest tests/ -v
```

---

## 🔌 Sample API Call (curl)

```bash
curl -X POST "http://127.0.0.1:8000/assess_risk" \
  -H "Content-Type: application/json" \
  -d "{\"transaction_id\":\"TXN_999\",\"merchant_id\":42,\"amount\":45000,\"payment_method\":\"UPI\",\"delivery_distance_km\":350,\"customer_tenure_days\":2,\"order_item_category\":\"Electronics\",\"payment_failure_history\":3,\"is_foreign_ip\":true,\"hours_since_last_order\":0.5,\"transaction_velocity_5min\":8,\"device_age_days\":1,\"attempted_payment_method_changes\":0}"
```

**Example response**

```json
{
  "transaction_id": "TXN_999",
  "risk_score": 0.87,
  "action": "DECLINE",
  "explanation": [
    "is_foreign_ip increased risk by 32.4%",
    "transaction_velocity_5min increased risk by 18.1%"
  ],
  "latency_ms": 142.5,
  "fallback_used": false,
  "timestamp": "2026-09-05T14:30:00+00:00"
}
```

---

## 🔌 API Endpoints

| Method | Path | Description |
|---|---|---|
| `POST` | `/assess_risk` | Score a transaction for fraud risk |
| `GET` | `/health` | Liveness check |
| `GET` | `/metrics` | Trained model performance + cost metrics |

### Decision logic

| Risk score | Action |
|---|---|
| `> 0.75` | `DECLINE` |
| `0.50 < score ≤ 0.75` | `MANUAL_REVIEW` |
| `≤ 0.50` | `APPROVE` |

### Safety / fallback behaviour

- **Cold start** — a merchant with fewer than 5 transactions in the current
  process is scored with interpretable rules (approve if amount `< ₹20,000`
  and not a foreign IP, otherwise `MANUAL_REVIEW`).
- **Circuit breaker** — if the ML+SHAP step exceeds **300 ms** or raises an
  exception, the request gracefully falls back to rule-based scoring and every
  breaker trip is appended to `logs/fallback_alerts.json`.

---

## 📊 Dashboard

The dashboard has been completely rewritten into a **stunning, professional, user-friendly SaaS-style interface** designed to be easily understood by non-technical users within 30 seconds:

- **🏠 Home** — A welcoming hero section explaining the value proposition and showing the total cost saved.
- **📊 Main Analytics** — Metric cards (Precision, Recall, False Positive Rate, Cost Saved) with plain-English tooltips. A dynamic confusion matrix and threshold slider to see cost impact in real-time. Insightful charts showing Risk Score Distribution and Fraud Rate by Payment Method.
- **🔬 Test Transaction (Simulator)** — Manually enter transaction details and get an instant Risk Score, Action badge (Approve/Review/Decline), and a plain-English SHAP explanation of why the decision was made.
- **📖 How It Works** — A jargon-free, 4-step visual guide explaining the ML pipeline to merchants.
- **📋 Audit Log** — A searchable, filterable table of the transaction history with CSV download capabilities.

---

## 🐳 Docker

```bash
make docker-build   # or: docker compose build
make docker-up      # or: docker compose up
```

- API: <http://localhost:8000>
- Dashboard: <http://localhost:8501>

---

## ☁️ Deployment

### 🚀 Streamlit Cloud (dashboard — recommended for the demo)

1. Push this repo to GitHub.
2. Go to <https://streamlit.io/cloud> and **Connect GitHub**.
3. **New app** → select `akashupadhayay106-au/razorpay-risk-shield`, branch
   `main`, main file path `dashboard/app.py`, Python `3.12`.
4. Click **Deploy** — you'll get a public URL like
   `https://razorpay-risk-shield-<random>.streamlit.app`.

Full walkthrough: [`deploy_streamlit.md`](deploy_streamlit.md).

### Render (API — one-click)

Push this repo to GitHub, connect it to Render, and Render auto-detects
`render.yaml` — data generation + training run during the build command and
the API starts on `$PORT` with `/health` as the health check.

---

## 🧠 Model & Cost Notes

- **Algorithm** — XGBoost with `scale_pos_weight = (neg/pos) × (FN/FP)` so a
  false negative (₹2,500 chargeback) is weighted 5× heavier than a false
  positive (₹500 lost sale).
- **Preprocessing** — one-hot encoding (`payment_method`,
  `order_item_category`), median imputation, and `StandardScaler` fitted on the
  training split only.
- **Derived features** — `amount_per_tenure`, `is_high_velocity`,
  `is_new_device`.
- **Explainability** — SHAP **TreeExplainer** (not KernelExplainer) for fast,
  exact single-row attribution.

---

## 📸 Screenshots

> _(Add screenshots of the dashboard, Swagger docs, and a sample risk
> response here.)_

---

## 📄 License

MIT — see [LICENSE](LICENSE).