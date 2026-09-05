# 🚀 Deploy the RiskShield AI Dashboard to Streamlit Cloud

This guide walks you through deploying the **live monitoring dashboard**
(`dashboard/app.py`) to Streamlit Cloud in under 5 minutes.

> 💡 **Why the dashboard, not the API?** Streamlit Cloud runs Python 3.10–3.12
> and is ideal for a self-contained data-app. The dashboard loads the **trained
> model artifacts** (`models/`) and the **test dataset** (`data/transactions.csv`)
> that are committed in this repo, so it works standalone — no training required.

---

## ✅ Prerequisites

1. This repo is pushed to GitHub at:
   `https://github.com/akashupadhayay106-au/razorpay-risk-shield`
2. It contains the committed model + data files (already included):
   - `models/risk_model.pkl`
   - `models/scaler.pkl`
   - `models/feature_names.pkl`
   - `models/model_metrics.json`
   - `data/transactions.csv`

---

## 🧭 Step-by-Step

1. **Go to Streamlit Cloud**
   → open <https://streamlit.io/cloud>.

2. **Sign in** with your GitHub account
   (click **"Continue with GitHub"** and authorize the app).

3. **Create a new app**
   → click the **"New app"** / **"Deploy an app"** button.

4. **Connect your GitHub account** if prompted.

5. **Fill in the deployment form:**

   | Field | Value |
   |---|---|
   | Repository | `akashupadhayay106-au/razorpay-risk-shield` |
   | Branch | `main` |
   | Main file path | `dashboard/app.py` |
   | Python version | `3.12` (matched by the `.python-version` file) |

6. **Click "Deploy"**
   🎉 Streamlit will provision a cloud environment, `pip install -r
   requirements.txt`, and start the app.

7. **Public URL**
   After a minute or two you'll get a URL like:
   `https://razorpay-risk-shield-<random>.streamlit.app`

> 🔁 Every push to `main` auto-redeploys. No manual steps needed.

---

## 📊 What you'll see (live)

- 💳 **Metric cards** — Precision, Recall, F1, FPR, **Total Cost Saved (₹)**
- 🎚️ **Threshold slider (0.30–0.90)** — drag it and watch the Confusion Matrix
  and Cost Saved recalculate in **real time**
- 📋 **Transaction audit log** — last 50 requests from `audit.db` if present
- 📈 **Risk-score histogram** — approved vs. declined score distributions

---

## ⚙️ Python Version Note

- **Streamlit Cloud** supports **Python 3.10 – 3.12** (not 3.14 yet).
- This repo pins **`3.12`** via `.python-version` for Streamlit Cloud.
- The **Docker image** (`docker/Dockerfile`) stays on `python:3.14-slim` for
  local/Docker deployments — the two runtimes are intentionally independent.

---

## 🛠 Troubleshooting

| Symptom | Fix |
|---|---|
| Import error for `src.features` | Deploy from the **repo root** (Streamlit does this automatically). |
| "No metrics found" banner | Ensure `models/model_metrics.json` is committed (it is). |
| Model fails to load | Confirm `risk_model.pkl` / `scaler.pkl` / `feature_names.pkl` are present in `models/`. |
| Very slow first load | Normal — cold start installs deps (~1–2 min). Refresh after it finishes. |
| Shows fake/placeholder table | The audit log reads `audit.db`, which is empty on a fresh cloud deploy — the dashboard gracefully shows a "no transactions yet" message. The threshold/confusion-matrix section works from committed test data regardless. |

---

## 🏁 Done

Once deployed, grab your `.streamlit.app` URL and add it to the **Live Demo**
section of the `README.md` for your buildathon submission. 🏆