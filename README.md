# 🛡️ RiskShield AI

## AI That Optimizes the Cost of Being Wrong

RiskShield AI is a cost-sensitive transaction decision engine that balances fraud risk, customer friction and business loss to recommend the best payment action.

## 🚀 LIVE DEMO

### [Open RiskShield AI →](https://kalicharan-razorpay-risk-shield.streamlit.app/)

`https://kalicharan-razorpay-risk-shield.streamlit.app/`

---

## What Problem Are We Solving?

Every payment decision has two risks:

1. **FRAUD GETS THROUGH** → Financial loss
2. **GOOD CUSTOMER GETS BLOCKED** → Revenue + customer trust loss

Traditional fraud detection systems often optimize solely for accuracy ("Is this fraud?").

**RiskShield optimizes the cost of being wrong.** 

The best fraud decision is not necessarily the decision with the highest accuracy. It is the decision that results in the lowest expected business cost.

---

## How It Works

RiskShield combines Machine Learning with deterministic business rules to evaluate every transaction:

**TRANSACTION**  
↓  
**AI RISK SCORE** (XGBoost)  
↓  
**EXPLANATION** (SHAP)  
↓  
**COST ENGINE** (Business Logic)  
↓  
**DECISION POLICY**  
↓  
**ACTION** (APPROVE / REVIEW / DECLINE)

---

## 🎬 90-Second Judge Demo

Want to see cost-sensitive AI in action? 

1. Open the [Live Demo](https://kalicharan-razorpay-risk-shield.streamlit.app/)
2. Go to the **Transaction Lab** page.
3. Click the **HIGH RISK** preset.
4. Read the **AI Explanation** to understand exactly why it was flagged.
5. Change the **Transaction Amount** and observe the Risk Score update in milliseconds.
6. Open the **Cost Optimizer** page.
7. Adjust the Risk Threshold and see the **Expected Business Cost** curve change.
8. Compare your current policy against the AI's optimized lowest-cost decision.

---

## Key Features

- **Transaction Lab**: Simulate any transaction and watch the AI evaluate the risk and explain its reasoning in plain English using SHAP values.
- **Cost Optimizer**: Input your business assumptions (Fraud Loss Cost, FP Cost, Review Cost) and let the engine find the mathematically optimal decision threshold.
- **Explainable AI**: No black boxes. Every decision is backed by feature-level contribution metrics.
- **Judge Mode**: A built-in guided tour of the product's core value proposition.
- **Trust & Safety (Failure Recovery)**: The API is protected by circuit breakers, cold-start rules for new merchants, and median-imputation for missing data. It degrades gracefully to manual review if the model fails.

---

## Why AI? And Why Not AI Everywhere?

**Why AI here?**  
Machine learning identifies complex, non-linear transaction patterns that are incredibly difficult to encode using fixed heuristic rules (e.g., velocity spikes coupled with specific foreign IP subnets).

**Why not AI everywhere?**  
Business costs, thresholds, safety controls, and audit policies remain explicit, deterministic, and fully controllable by the merchant. The AI predicts the probability; the business logic calculates the cost.

---

## Architecture & Tech Stack

- **Dashboard**: Streamlit, Plotly
- **Backend API**: FastAPI, Uvicorn
- **Machine Learning**: XGBoost (Classifier), SHAP (TreeExplainer)
- **Data Pipeline**: Scikit-learn (StandardScaler, Median Imputation)
- **Deployment**: Streamlit Cloud (Frontend), Render (Backend), Docker

---

## Local Setup

```bash
# Clone the repository
git clone https://github.com/akashupadhayay106-au/razorpay-risk-shield.git
cd razorpay-risk-shield

# Install dependencies
pip install -r requirements.txt

# Run the backend API
make api-dev

# In a separate terminal, run the dashboard
make dashboard
```

---

## 🚀 LIVE DEMO

### [Open RiskShield AI →](https://kalicharan-razorpay-risk-shield.streamlit.app/)