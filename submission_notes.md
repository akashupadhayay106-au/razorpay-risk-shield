# 📝 Submission Notes — RiskShield AI

- **Track:** 02 — AI Risk Manager
- **Project:** RiskShield AI
- **Objectives:** Cost-aware fraud detection with SHAP explainability — score
  Indian payment transactions in real time, explain each decision, and minimise
  total business cost (false negatives ₹2,500 vs false positives ₹500).

---

## 🏗️ Pipeline Implemented

1. **Data generation** — `data/generate_data.py` creates 25,000 synthetic
   transactions tailored to the Indian payment ecosystem (UPI/Card/NetBanking/
   Wallet, log-normal amounts ₹100–₹1,00,000, ~8% fraud rate).
2. **Training** — `src/train_model.py`: 70/15/15 stratified split, one-hot +
   `StandardScaler`, cost-sensitive XGBoost tuned with GridSearchCV.
3. **Serving** — `api/main.py`: FastAPI `/assess_risk` with cold-start routing,
   a 300 ms circuit breaker, SQLite audit trail, and SHAP explanations.
4. **Monitoring** — `dashboard/app.py`: Streamlit dashboard with an interactive
   threshold slider and live cost recalculation.

---

## 🧗 Challenges Faced

1. **Python 3.14 + Windows compatibility**
   - Removed `uvicorn[standard]` (pulls in `uvloop`, which crashes on Windows)
     and use base `uvicorn`; upgraded xgboost / shap / scikit-learn to versions
     with Python 3.14 wheels.
2. **XGBoost deprecation**
   - Removed the deprecated `use_label_encoder` parameter and explicitly set
     `eval_metric='logloss'` for xgboost ≥ 3.0.
3. **SHAP latency**
   - Switched to `shap.TreeExplainer` (exact, tree-structure-based) instead of
     the slow `KernelExplainer`, keeping single-row latency under 150 ms.
4. **Cold start for new merchants**
   - Merchants with fewer than 5 observed transactions are scored with simple,
     interpretable rules instead of an under-trained model.
5. **Circuit breaker / graceful degradation**
   - If the ML+SHAP step exceeds 300 ms or raises an exception, the request
     falls back to rule-based scoring and logs the event to
     `logs/fallback_alerts.json`.

---

## 🔗 Resources

- **GitHub Repo:** https://github.com/akashupadhayay106-au/razorpay-risk-shield
- **Demo runbook:** see `README.md` (setup → data → train → API → dashboard).