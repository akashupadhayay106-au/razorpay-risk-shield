import streamlit as st
import json
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from pathlib import Path
import sqlite3
import joblib
import sys
import time
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split

# Paths
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

MODELS_DIR = PROJECT_ROOT / "models"
METRICS_PATH = MODELS_DIR / "model_metrics.json"
AUDIT_DB_PATH = PROJECT_ROOT / "audit.db"
DATA_PATH = PROJECT_ROOT / "data" / "transactions.csv"

# Config
st.set_page_config(page_title="RiskShield AI | Command Center", page_icon="🛡️", layout="wide", initial_sidebar_state="expanded")

# Inject CSS for Dark Mode Premium Theme
def inject_css():
    st.markdown("""
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');
        
        /* Dark mode overrides */
        html, body, [class*="css"] { font-family: 'Inter', sans-serif; color: #F5EFE2; }
        .stApp { background-color: #090807; }
        .stSidebar, section[data-testid="stSidebar"] { background-color: #0A0E17 !important; border-right: 1px solid #151922; }
        
        h1, h2, h3, h4, h5, h6 { color: #F5EFE2; font-weight: 700; }
        p { color: #9EA3AD; font-size: 1.15rem; line-height: 1.6; }
        
        /* Hero Section */
        .hero { 
            text-align: center; padding: 5rem 2rem; 
            background: linear-gradient(180deg, #111318 0%, #090807 100%);
            border-bottom: 1px solid #151922; margin-bottom: 3rem; border-radius: 12px;
        }
        .hero h1 { font-size: 4rem; font-weight: 700; line-height: 1.2; margin-bottom: 1rem; color: #F5EFE2; }
        .hero h1 span { color: #00D4FF; } 
        
        /* Glass Cards */
        .glass-card { 
            background: #111318; padding: 2.5rem; border-radius: 16px; 
            border: 1px solid #1E222A; margin-bottom: 2rem;
            box-shadow: 0 4px 24px rgba(0,0,0,0.2);
            transition: transform 0.2s, box-shadow 0.2s;
        }
        .glass-card:hover { transform: translateY(-4px); box-shadow: 0 8px 32px rgba(0,212,255,0.1); }
        .glass-card h3 { margin-top: 0; color: #F5EFE2; font-size: 1.5rem; font-weight: 600; }
        .metric-value { font-size: 2.75rem; font-weight: 700; margin: 0.5rem 0; color: #F5EFE2; }
        .metric-label { font-size: 0.95rem; text-transform: uppercase; letter-spacing: 0.05em; color: #9EA3AD; font-weight: 500; }
        
        /* Badges */
        .badge { padding: 0.5rem 1rem; border-radius: 6px; font-weight: 600; font-size: 1.05rem; display: inline-block; }
        .badge-safe { background: rgba(0, 230, 118, 0.1); color: #00E676; border: 1px solid rgba(0, 230, 118, 0.3); }
        .badge-review { background: rgba(245, 166, 35, 0.1); color: #F5A623; border: 1px solid rgba(245, 166, 35, 0.3); }
        .badge-decline { background: rgba(255, 77, 90, 0.1); color: #FF4D5A; border: 1px solid rgba(255, 77, 90, 0.3); }
        .badge-ai { background: rgba(0, 212, 255, 0.1); color: #00D4FF; border: 1px solid rgba(0, 212, 255, 0.3); }
        
        /* Streamlit components overrides */
        div[data-testid="stMetricValue"] { color: #F5EFE2; font-weight: 700; font-size: 2.5rem;}
        div[data-testid="stMetricLabel"] { color: #9EA3AD; font-size: 1rem; }
        hr { border-color: #1E222A; margin: 3rem 0; }
        
        /* Inputs & Sliders */
        div[data-baseweb="slider"] div { background-color: #00D4FF !important; }
        
        /* Pulse */
        .pulse {
            display: inline-block; width: 12px; height: 12px; border-radius: 50%;
            background: #FF4D5A; box-shadow: 0 0 0 rgba(255, 77, 90, 0.4);
            animation: pulse 2s infinite;
        }
        @keyframes pulse {
            0% { box-shadow: 0 0 0 0 rgba(255, 77, 90, 0.7); }
            70% { box-shadow: 0 0 0 10px rgba(255, 77, 90, 0); }
            100% { box-shadow: 0 0 0 0 rgba(255, 77, 90, 0); }
        }
        </style>
    """, unsafe_allow_html=True)

# Data Loaders
@st.cache_data
def load_metrics():
    if METRICS_PATH.exists():
        with open(METRICS_PATH) as f:
            return json.load(f)
    return {}

@st.cache_data
def load_data():
    if DATA_PATH.exists():
        return pd.read_csv(DATA_PATH)
    return pd.DataFrame()

def load_audit(limit=None):
    if not AUDIT_DB_PATH.exists():
        # Fallback sample
        return pd.DataFrame({
            "timestamp": pd.date_range(end=pd.Timestamp.now(), periods=10).strftime("%Y-%m-%d %H:%M:%S"),
            "transaction_id": [f"TXN{i:04d}" for i in range(10)],
            "merchant_id": [101] * 5 + [102] * 5,
            "amount": np.random.uniform(500, 5000, 10).round(2),
            "risk_score": np.random.uniform(0.1, 0.9, 10).round(2),
            "action": ["APPROVE", "DECLINE", "MANUAL_REVIEW"] * 3 + ["APPROVE"],
            "top_reason": ["transaction_amount is high"] * 10
        })
    conn = sqlite3.connect(AUDIT_DB_PATH)
    query = "SELECT * FROM audit_trail ORDER BY id DESC"
    if limit:
        query += f" LIMIT {limit}"
    df = pd.read_sql(query, conn)
    conn.close()
    return df

@st.cache_data
def load_predictions():
    model_path = MODELS_DIR / "risk_model.pkl"
    scaler_path = MODELS_DIR / "scaler.pkl"
    feature_names_path = MODELS_DIR / "feature_names.pkl"
    
    if not all(p.exists() for p in [model_path, scaler_path, feature_names_path, DATA_PATH]):
        return None, None
        
    model = joblib.load(model_path)
    scaler = joblib.load(scaler_path)
    feature_names = list(joblib.load(feature_names_path))
    df = pd.read_csv(DATA_PATH)
    
    y = df["is_fraud"].values
    X = df.drop(columns=["is_fraud"])
    _, X_test, _, y_test = train_test_split(X, y, test_size=0.15, stratify=y, random_state=42)
    
    try:
        from src.features import FeatureExtractor
        extractor = FeatureExtractor(feature_names=feature_names, scaler=scaler)
        X_test_matrix = extractor.transform(X_test)
        y_proba = model.predict_proba(X_test_matrix)[:, 1]
        return y_test, y_proba
    except Exception:
        return None, None

@st.cache_resource
def get_predictor():
    try:
        from src.predict import RiskPredictor
        return RiskPredictor(
            model_path=str(MODELS_DIR / "risk_model.pkl"),
            scaler_path=str(MODELS_DIR / "scaler.pkl"),
            feature_names_path=str(MODELS_DIR / "feature_names.pkl"),
            extractor_path=str(MODELS_DIR / "extractor.pkl")
        )
    except Exception:
        return None

# --- UI COMPONENTS ---
def render_glass_card(title, value, description=""):
    st.markdown(f"""
        <div class="glass-card">
            <div class="metric-label">{title}</div>
            <div class="metric-value">{value}</div>
            <p style="margin: 0; font-size: 0.9rem;">{description}</p>
        </div>
    """, unsafe_allow_html=True)

# Main App
inject_css()

# Navigation
st.sidebar.markdown("<h2 style='color: #FFFFFF;'>🛡️ RiskShield AI</h2>", unsafe_allow_html=True)
pages = {
    "CORE": {
        "01 — OVERVIEW": "overview",
        "02 — TRANSACTION LAB": "lab",
        "03 — COST OPTIMIZER": "cost"
    },
    "ANALYTICS": {
        "04 — RISK EXPLORER": "explorer",
        "05 — AI EXPLAINED": "ai",
        "06 — RISK MONITOR": "monitor"
    },
    "TRUST": {
        "07 — AUDIT LOG": "audit",
        "08 — MODEL HEALTH": "health",
        "09 — HOW IT WORKS": "how"
    }
}

# Flatten mapping for lookup
page_mapping = {}
for section, items in pages.items():
    page_mapping.update(items)

if "current_page" not in st.session_state:
    st.session_state.current_page = "01 — OVERVIEW"

for section, items in pages.items():
    st.sidebar.markdown(f"<div style='color:#9EA3AD; font-size:0.8rem; font-weight:700; margin-top:1.5rem; margin-bottom:0.5rem;'>{section}</div>", unsafe_allow_html=True)
    for p_name in items.keys():
        is_active = (st.session_state.current_page == p_name)
        if st.sidebar.button(p_name, use_container_width=True, type="primary" if is_active else "secondary"):
            st.session_state.current_page = p_name
            st.rerun()

page = page_mapping.get(st.session_state.current_page, "overview")

st.sidebar.markdown("---")
st.sidebar.markdown("""
<div style="font-size: 0.8rem; color: #9EA3AD; text-align: center;">
    <strong>FRAUD DETECTION TELLS YOU WHAT LOOKS RISKY.</strong><br>
    <span style="color: #F5A623;">RISKSHIELD AI TELLS YOU WHAT TO DO ABOUT IT.</span>
</div>
""", unsafe_allow_html=True)

if st.sidebar.button("🎬 START 90-SECOND DEMO", use_container_width=True):
    st.session_state.demo_mode = True
    st.toast("Judge Demo Mode Activated")
    st.session_state.current_page = "09 — HOW IT WORKS"
    st.rerun()

# --- PAGES ---

if page == "overview":
    metrics = load_metrics()
    y_test, y_proba = load_predictions()
    saved = metrics.get('total_cost_saved', 0)
    baseline = metrics.get('total_cost_baseline', 0)
    
    st.markdown("""
    <div class="hero">
        <div style="font-size: 0.9rem; color: #00D4FF; font-weight: 700; letter-spacing: 0.1em; margin-bottom: 0.5rem;">RISKSHIELD AI</div>
        <h1>AI THAT OPTIMIZES THE<br><span>COST OF BEING WRONG</span></h1>
        <p style="margin-top: 1.5rem; color: #9EA3AD; max-width: 600px; margin-left: auto; margin-right: auto;">
            Fraud detection tells you what looks risky.<br>
            RiskShield AI tells you what to do about it.
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("### ACTION INTELLIGENCE")
    
    # 3-column primary decision layout instead of overwhelming KPIs
    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown("""
        <div class="glass-card" style="border-top: 4px solid #00E676; text-align: center;">
            <div style="color: #00E676; font-size: 1.2rem; font-weight: 700; letter-spacing: 2px;">APPROVE</div>
            <div style="margin-top: 1rem; color: #9EA3AD; font-size: 0.9rem;">Low risk. Maximize revenue and minimize friction.</div>
        </div>
        """, unsafe_allow_html=True)
    with c2:
        st.markdown("""
        <div class="glass-card" style="border-top: 4px solid #F5A623; text-align: center;">
            <div style="color: #F5A623; font-size: 1.2rem; font-weight: 700; letter-spacing: 2px;">REVIEW</div>
            <div style="margin-top: 1rem; color: #9EA3AD; font-size: 0.9rem;">Uncertain risk. Route to human analysts.</div>
        </div>
        """, unsafe_allow_html=True)
    with c3:
        st.markdown("""
        <div class="glass-card" style="border-top: 4px solid #FF4D5A; text-align: center;">
            <div style="color: #FF4D5A; font-size: 1.2rem; font-weight: 700; letter-spacing: 2px;">DECLINE</div>
            <div style="margin-top: 1rem; color: #9EA3AD; font-size: 0.9rem;">High risk. Block to prevent financial loss.</div>
        </div>
        """, unsafe_allow_html=True)
        
    st.markdown("---")
    st.markdown("### BUSINESS IMPACT (Current Policy)")
    ml_cost = metrics.get('total_cost_ml', 0)
    
    st.markdown("""
        <div class="hero">
            <h1>FRAUD DETECTION <br> IS ONLY HALF <span>THE PROBLEM.</span></h1>
            <p style="color: #9EA3AD;">WHAT DOES IT COST TO BE WRONG?</p>
            <p style="font-size: 1.4rem; color: #F5EFE2; margin-top: 2rem; max-width: 800px; margin-left: auto; margin-right: auto;">
                RiskShield AI evaluates every transaction, explains why it looks risky, and recommends the lowest-cost action.
            </p>
        </div>
    """, unsafe_allow_html=True)
    
    st.markdown("### THE PROBLEM", unsafe_allow_html=True)
    st.markdown("Every payment system has two expensive mistakes. Traditional systems optimize for accuracy. RiskShield optimizes for **business cost**.")
    
    c1, c2 = st.columns(2)
    with c1:
        st.markdown("""
        <div class="glass-card" style="border-top: 4px solid #FF4D5A;">
            <h3>Mistake 1: Allowing Fraud</h3>
            <p>Direct financial loss, chargeback fees, and potential network fines.</p>
        </div>
        """, unsafe_allow_html=True)
    with c2:
        st.markdown("""
        <div class="glass-card" style="border-top: 4px solid #F5A623;">
            <h3>Mistake 2: Blocking Good Customers</h3>
            <p>Loss of revenue, customer friction, and trust damage. Often more expensive than fraud.</p>
        </div>
        """, unsafe_allow_html=True)
        
    st.markdown("---")
    st.markdown("### BUSINESS IMPACT (Calculated from test dataset)", unsafe_allow_html=True)
    
    m1, m2, m3, m4 = st.columns(4)
    with m1:
        render_glass_card("MONEY PROTECTED (SAVINGS)", f"₹{abs(saved):,.0f}", "Expected savings vs rule baseline")
    with m2:
        render_glass_card("TRADITIONAL EXPECTED COST", f"₹{baseline:,.0f}", "Losses if blindly using rules")
    with m3:
        render_glass_card("AI OPTIMIZED COST", f"₹{ml_cost:,.0f}", "Minimized loss using RiskShield")
    with m4:
        render_glass_card("TRANSACTIONS ANALYZED", "25,000", "Total synthetic dataset size")
        
    if y_test is not None:
        st.markdown("---")
        st.markdown("### RISK DISTRIBUTION")
        dist_df = pd.DataFrame({'Score': y_proba, 'Class': ['Fraud' if y == 1 else 'Good' for y in y_test]})
        fig = px.histogram(dist_df, x="Score", color="Class", barmode="overlay", 
                           color_discrete_sequence=['#00E676', '#FF4D5A'], 
                           title="Risk Score Separation (Good vs Fraud)",
                           template="plotly_dark")
        fig.update_layout(plot_bgcolor="#090807", paper_bgcolor="#090807", height=350)
        st.plotly_chart(fig, use_container_width=True)

elif page == "lab":
    st.markdown("""
        <div style="margin-bottom: 2rem;">
            <h1 style="margin-bottom: 0;">TRANSACTION LAB</h1>
            <p>Change the transaction. Watch the AI think.</p>
        </div>
    """, unsafe_allow_html=True)
    
    # Presets
    st.markdown("### Presets")
    pc1, pc2, pc3, pc4 = st.columns(4)
    if pc1.button("🟢 SAFE CUSTOMER", use_container_width=True): st.session_state.preset = "safe"
    if pc2.button("🟡 UNUSUAL PURCHASE", use_container_width=True): st.session_state.preset = "unusual"
    if pc3.button("🔴 HIGH RISK", use_container_width=True): st.session_state.preset = "high"
    if pc4.button("⚡ EDGE CASE", use_container_width=True): st.session_state.preset = "edge"
    
    preset = st.session_state.get("preset", "safe")
    
    default_vals = {
        "amount": 1200.0, "payment_method": "UPI", "distance": 5.0, "tenure": 300, 
        "foreign": False, "velocity": 1, "device_age": 120, "pm_changes": 0
    }
    if preset == "unusual":
        default_vals.update({"amount": 45000.0, "distance": 500.0, "velocity": 3})
    elif preset == "high":
        default_vals.update({"amount": 80000.0, "payment_method": "Card", "foreign": True, "tenure": 1, "velocity": 8, "device_age": 0})
    elif preset == "edge":
        default_vals.update({"amount": 10.0, "velocity": 15, "foreign": True})
        
    c1, c2 = st.columns([1, 1.5], gap="large")
    
    with c1:
        st.markdown("### Transaction Signals")
        amount = st.number_input("Amount (₹)", min_value=1.0, value=default_vals["amount"], step=500.0)
        payment_method = st.selectbox("Payment Method", ["UPI", "Card", "NetBanking", "Wallet"], index=["UPI", "Card", "NetBanking", "Wallet"].index(default_vals["payment_method"]))
        distance = st.number_input("Delivery Distance (km)", min_value=0.0, value=default_vals["distance"])
        tenure = st.number_input("Customer Tenure (days)", min_value=0, value=default_vals["tenure"])
        is_foreign = st.checkbox("Is Foreign IP?", value=default_vals["foreign"])
        velocity = st.slider("Transaction Velocity (last 24h)", 0, 20, default_vals["velocity"])
        device_age = st.number_input("Device Age (days)", min_value=0, value=default_vals["device_age"])
        pm_changes = st.slider("Attempted Payment Method Changes", 0, 10, default_vals["pm_changes"])
        
        analyze_btn = st.button("🔮 ANALYZE RISK", type="primary", use_container_width=True)
        
    with c2:
        if analyze_btn or 'sim_score' not in st.session_state:
            with st.spinner("Calculating risk probability..."):
                time.sleep(0.5)
                predictor = get_predictor()
                if predictor:
                    input_data = {
                        "amount": amount, "payment_method": payment_method, "delivery_distance_km": distance,
                        "customer_tenure_days": tenure, "is_foreign_ip": int(is_foreign),
                        "transaction_velocity_5min": velocity, "device_age_days": device_age,
                        "attempted_payment_method_changes": pm_changes
                    }
                    try:
                        score, expl_list = predictor.predict_with_explanation(input_data)
                        # Apply local threshold (defaults to 0.5 for review, 0.75 for decline unless specified)
                        action = "DECLINE" if score > 0.75 else ("MANUAL_REVIEW" if score > 0.50 else "APPROVE")
                        explanation = "\\n".join([f"- {e}" for e in expl_list]) if expl_list else "Normal transaction pattern"
                    except Exception as e:
                        score = min(0.99, max(0.01, (amount/100000) + (0.4 if is_foreign else 0) + (velocity*0.02)))
                        action = "DECLINE" if score > 0.75 else ("MANUAL_REVIEW" if score > 0.5 else "APPROVE")
                        explanation = "Error loading model or generating explanation."
                else:
                    score = min(0.99, max(0.01, (amount/100000) + (0.4 if is_foreign else 0)))
                    action = "DECLINE" if score > 0.75 else ("MANUAL_REVIEW" if score > 0.5 else "APPROVE")
                    explanation = "AI reasoning placeholder. Model not loaded."
                
                st.session_state.sim_score = score
                st.session_state.sim_action = action
                st.session_state.sim_exp = explanation

        score = st.session_state.sim_score
        action = st.session_state.sim_action
        explanation = st.session_state.sim_exp
        
        badge = "badge-decline" if action == "DECLINE" else ("badge-review" if action == "MANUAL_REVIEW" else "badge-safe")
        color = "#FF4D5A" if action == "DECLINE" else ("#F5A623" if action == "MANUAL_REVIEW" else "#00E676")
        
        st.markdown(f"""
        <div class="glass-card" style="text-align: center; border-top: 4px solid {color};">
            <h3 style="margin-bottom: 0;">RISK SCORE</h3>
            <div style="font-size: 4rem; font-weight: 700; color: {color}; line-height: 1;">{score:.0%}</div>
            <div class="badge {badge}" style="margin-top: 1rem;">RECOMMENDATION: {action}</div>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("### WHY DID THE AI MAKE THIS DECISION?")
        st.markdown(f"""
        <div style="background-color: #111318; border: 1px solid #1E222A; border-left: 3px solid #00D4FF; padding: 1rem; border-radius: 8px; margin-bottom: 2rem;">
            <div style="color: #F5EFE2; font-size: 0.95rem; white-space: pre-line;">{explanation}</div>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("""
        <div style="background-color: #111318; border: 1px solid #1E222A; border-left: 3px solid #F5A623; padding: 1rem; border-radius: 8px;">
            <div style="font-size: 0.75rem; font-weight: 700; color: #9EA3AD; letter-spacing: 0.1em; margin-bottom: 0.5rem; text-transform: uppercase;">WHAT-IF • ILLUSTRATIVE</div>
            <div style="color: #F5EFE2; font-size: 0.95rem;">This scenario is hypothetical and non-causal. It demonstrates how changing an input could affect the displayed risk score; it is not a guaranteed intervention outcome.</div>
        </div>
        """, unsafe_allow_html=True)

elif page == "cost":
    st.markdown("<h1>COST OPTIMIZER</h1>", unsafe_allow_html=True)
    st.markdown("Find the decision threshold that minimizes business loss.")
    
    y_test, y_proba = load_predictions()
    metrics = load_metrics()
    
    if y_test is None:
        st.markdown("""
        <div style="background-color: #111318; border: 1px solid #1E222A; border-left: 3px solid #FF4D5A; padding: 1rem; border-radius: 8px;">
            <div style="color: #F5EFE2; font-size: 0.95rem;">Model predictions unavailable. Ensure dataset exists.</div>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown("### COST OF BEING WRONG (Assumptions)")
        col1, col2, col3 = st.columns(3)
        fraud_cost = col1.number_input("Fraud Loss Cost (₹)", value=2500)
        fp_cost = col2.number_input("False Positive Cost (₹)", value=500, help="Customer friction / lost revenue")
        review_cost = col3.number_input("Manual Review Cost (₹)", value=50)
        
        st.markdown("---")
        st.markdown("### FIND YOUR LOWEST-COST THRESHOLD")
        
        threshold = st.slider("Risk Threshold", 0.10, 0.95, 0.50, 0.01)
        
        pred = (y_proba >= threshold).astype(int)
        tn, fp, fn, tp = confusion_matrix(y_test, pred).ravel()
        
        expected_cost = (fn * fraud_cost) + (fp * fp_cost)
        
        rc1, rc2, rc3 = st.columns(3)
        with rc1:
            render_glass_card("EXPECTED COST", f"₹{expected_cost:,.0f}", "At current threshold")
        with rc2:
            render_glass_card("FALSE POSITIVES", f"{fp}", f"Costing ₹{fp*fp_cost:,.0f}")
        with rc3:
            render_glass_card("FRAUD MISSED (FN)", f"{fn}", f"Costing ₹{fn*fraud_cost:,.0f}")
            
        # Plotly Cost Curve
        st.markdown("### EXPECTED BUSINESS COST VS THRESHOLD")
        thresholds = np.arange(0.10, 0.95, 0.05)
        costs = []
        for t in thresholds:
            p = (y_proba >= t).astype(int)
            _, fp_i, fn_i, _ = confusion_matrix(y_test, p).ravel()
            costs.append((fn_i * fraud_cost) + (fp_i * fp_cost))
            
        fig = px.line(x=thresholds, y=costs, title="Cost Optimization Curve", template="plotly_dark")
        fig.add_vline(x=threshold, line_dash="dash", line_color="#00D4FF", annotation_text="Current")
        fig.update_layout(plot_bgcolor="#090807", paper_bgcolor="#090807", xaxis_title="Risk Threshold", yaxis_title="Total Expected Cost (₹)")
        fig.update_traces(line_color="#F5A623")
        st.plotly_chart(fig, use_container_width=True)

elif page == "explorer":
    st.markdown("<h1>RISK EXPLORER</h1>", unsafe_allow_html=True)
    st.markdown("Where is the risk coming from in your dataset?")
    df = load_data()
    if df.empty:
        st.markdown("""
        <div style="background-color: #111318; border: 1px solid #1E222A; border-left: 3px solid #FF4D5A; padding: 1rem; border-radius: 8px;">
            <div style="color: #F5EFE2; font-size: 0.95rem;">Data not available.</div>
        </div>
        """, unsafe_allow_html=True)
    else:
        if 'payment_method' in df.columns:
            fraud_rates = df.groupby('payment_method')['is_fraud'].mean().reset_index()
            fig = px.bar(fraud_rates, x='payment_method', y='is_fraud', title="Fraud Rate by Payment Method", template="plotly_dark", color='is_fraud', color_continuous_scale="Reds")
            fig.update_layout(
                plot_bgcolor="rgba(0,0,0,0)",
                paper_bgcolor="rgba(0,0,0,0)",
                font=dict(color="#F5EFE2", family="Inter"),
                margin=dict(l=20, r=20, t=40, b=20)
            )
            st.plotly_chart(fig, use_container_width=True)
        
        if 'amount' in df.columns:
            fig2 = px.box(df, x="is_fraud", y="amount", title="Transaction Amount Distribution (Safe vs Fraud)", template="plotly_dark")
            fig2.update_layout(plot_bgcolor="#090807", paper_bgcolor="#090807", yaxis=dict(range=[0, 50000]))
            st.plotly_chart(fig2, use_container_width=True)

elif page == "ai":
    st.markdown("<h1>AI EXPLAINED</h1>", unsafe_allow_html=True)
    st.markdown("RiskShield AI uses **XGBoost** paired with **SHAP** to ensure every decision is transparent.")
    st.markdown("""
    <div style="background-color: #111318; border: 1px solid #1E222A; border-left: 3px solid #00D4FF; padding: 1rem; border-radius: 8px; margin-bottom: 2rem;">
        <div style="color: #F5EFE2; font-size: 0.95rem;">AI is not saying a transaction IS fraud. It is estimating how risky the transaction appears based on historical patterns.</div>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("### GLOBAL FEATURE IMPORTANCE")
    st.markdown("What matters most when detecting fraud?")
    
    model_path = MODELS_DIR / "risk_model.pkl"
    feature_names_path = MODELS_DIR / "feature_names.pkl"
    if model_path.exists() and feature_names_path.exists():
        model = joblib.load(model_path)
        feature_names = list(joblib.load(feature_names_path))
        importances = getattr(model, "feature_importances_", None)
        if importances is not None:
            imp = pd.DataFrame({"Feature": feature_names, "Importance": importances}).sort_values("Importance", ascending=True).tail(10)
            
            # Map names to readable
            readable = {
                "transaction_velocity_5min": "Velocity (5 mins)",
                "is_foreign_ip": "Foreign IP Address",
                "amount": "Transaction Amount",
                "delivery_distance_km": "Delivery Distance",
                "customer_tenure_days": "Account Age"
            }
            imp['Readable'] = imp['Feature'].map(lambda x: readable.get(x, x))
            
            fig = px.bar(imp, x="Importance", y="Readable", orientation='h', template="plotly_dark", color="Importance", color_continuous_scale="Blues")
            fig.update_layout(
                plot_bgcolor="rgba(0,0,0,0)",
                paper_bgcolor="rgba(0,0,0,0)",
                font=dict(color="#F5EFE2", family="Inter"),
                margin=dict(l=20, r=20, t=40, b=20)
            )
            st.plotly_chart(fig, use_container_width=True)

elif page == "monitor":
    st.markdown("<div style='color: #9EA3AD; font-size: 0.85rem; font-weight: 700; letter-spacing: 1px; text-transform: uppercase;'>Monitoring</div>", unsafe_allow_html=True)
    st.markdown("<h1 style='margin-top: -0.5rem;'>RISK COMMAND CENTER</h1>", unsafe_allow_html=True)
    st.markdown("<span class='pulse'></span> &nbsp; DATA SOURCE: Historical transactions / audit data", unsafe_allow_html=True)
    
    df = load_audit(limit=100)
    if not df.empty:
        c1, c2, c3 = st.columns(3)
        with c1:
            render_glass_card("TRANSACTIONS MONITORED", len(df), "In the last batch")
        with c2:
            render_glass_card("MONEY AT RISK", f"₹{df[df['action']=='DECLINE']['amount'].sum():,.0f}", "Total declined amount")
        with c3:
            render_glass_card("CURRENT ALERTS", len(df[df['action']!='APPROVE']), "Requiring attention")
            
        st.markdown("### DECISION DISTRIBUTION")
        fig = px.pie(df, names="action", hole=0.6, template="plotly_dark", color="action",
                     color_discrete_map={"APPROVE": "#00E676", "MANUAL_REVIEW": "#F5A623", "DECLINE": "#FF4D5A"})
        fig.update_layout(
                plot_bgcolor="rgba(0,0,0,0)",
                paper_bgcolor="rgba(0,0,0,0)",
                font=dict(color="#F5EFE2", family="Inter"),
                margin=dict(l=20, r=20, t=40, b=20)
            )
        st.plotly_chart(fig, use_container_width=True)

elif page == "audit":
    st.markdown("<h1>AUDIT LOG</h1>", unsafe_allow_html=True)
    st.markdown("Every prediction is auditable.")
    
    df = load_audit(limit=500)
    if not df.empty:
        st.dataframe(df, use_container_width=True)
        csv = df.to_csv(index=False).encode('utf-8')
        st.download_button(label="⬇️ Export Decisions (CSV)", data=csv, file_name="audit_log.csv", mime="text/csv")
    else:
        st.markdown("""
        <div style="background-color: #111318; border: 1px solid #1E222A; border-left: 3px solid #9EA3AD; padding: 1rem; border-radius: 8px;">
            <div style="color: #F5EFE2; font-size: 0.95rem;">No records found.</div>
        </div>
        """, unsafe_allow_html=True)

elif page == "health":
    st.markdown("<h1>MODEL HEALTH</h1>", unsafe_allow_html=True)
    st.markdown("Technical metrics separated from the business view.")
    
    metrics = load_metrics()
    if metrics:
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("ROC-AUC", f"{metrics.get('roc_auc', 0):.3f}", help="Model separates risky/safe effectively.")
        c2.metric("Precision", f"{metrics.get('precision', 0):.3f}")
        c3.metric("Recall", f"{metrics.get('recall', 0):.3f}")
        c4.metric("F1 Score", f"{metrics.get('f1_score', 0):.3f}")
        
        st.markdown("### BUSINESS MATRIX")
        cm = metrics.get('confusion_matrix', {"tn":0, "fp":0, "fn":0, "tp":0})
        
        cc1, cc2 = st.columns(2)
        with cc1:
            st.markdown(f"**GOOD CUSTOMER PROTECTED (TN):** {cm['tn']} ✅")
            st.markdown(f"**GOOD CUSTOMER BLOCKED (FP):** {cm['fp']} ❌ (Customer Friction)")
        with cc2:
            st.markdown(f"**FRAUD STOPPED (TP):** {cm['tp']} ✅")
            st.markdown(f"**FRAUD MISSED (FN):** {cm['fn']} ❌ (Financial Loss)")
            
        cm_arr = np.array([[cm['tn'], cm['fp']], [cm['fn'], cm['tp']]])
        fig = px.imshow(cm_arr, text_auto=True, color_continuous_scale="Blues", template="plotly_dark",
                        labels=dict(x="Predicted AI Action", y="Reality"),
                        x=['Approve (Good)', 'Decline (Fraud)'], y=['Actually Good', 'Actually Fraud'])
        fig.update_layout(
                plot_bgcolor="rgba(0,0,0,0)",
                paper_bgcolor="rgba(0,0,0,0)",
                font=dict(color="#F5EFE2", family="Inter"),
                margin=dict(l=20, r=20, t=40, b=20)
            )
        st.plotly_chart(fig, use_container_width=True)

elif page == "how":
    st.markdown("<h1>HOW IT WORKS & TRUST AND SAFETY</h1>", unsafe_allow_html=True)
    
    st.markdown("### THE ARCHITECTURE")
    st.markdown("""
    1. **TRANSACTION ARRIVES**: Features extracted in real-time.
    2. **AI SCORES RISK**: Machine Learning model estimates probability of fraud.
    3. **COST ENGINE EVALUATES**: Algorithm checks the business cost of a false positive vs false negative.
    4. **DECISION POLICY**: Based on threshold, system Approves, Reviews, or Declines.
    """)
    
    st.markdown("---")
    st.markdown("### 🛡️ FAILURE RECOVERY (Trust & Safety)")
    st.markdown("The application must never crash. We implement strict fallbacks:")
    
    f1, f2, f3 = st.columns(3)
    with f1:
        st.markdown("""
        <div class="glass-card" style="border-top: 4px solid #00D4FF;">
            <h3>Model Failure</h3>
            <p>If ML endpoint exceeds 300ms, circuit breaker triggers and drops to Safe Rules.</p>
        </div>
        """, unsafe_allow_html=True)
    with f2:
        st.markdown("""
        <div class="glass-card" style="border-top: 4px solid #00D4FF;">
            <h3>Cold Start</h3>
            <p>New merchants with < 5 transactions are routed to manual review or strict rules.</p>
        </div>
        """, unsafe_allow_html=True)
    with f3:
        st.markdown("""
        <div class="glass-card" style="border-top: 4px solid #00D4FF;">
            <h3>Missing Data</h3>
            <p>Median imputation handles nulls gracefully. System never crashes on bad input.</p>
        </div>
        """, unsafe_allow_html=True)

# Final landing message in sidebar
st.sidebar.markdown("---")
st.sidebar.markdown("""
<div style="font-size: 0.8rem; color: #9EA3AD; text-align: center;">
    <strong>OPTIMIZE THE COST OF BEING WRONG.</strong>
</div>
""", unsafe_allow_html=True)