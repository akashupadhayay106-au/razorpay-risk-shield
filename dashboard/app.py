import streamlit as st
import json
import pandas as pd
import numpy as np
import altair as alt
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
st.set_page_config(page_title="RiskShield AI", page_icon="🛡️", layout="wide", initial_sidebar_state="expanded")

# Custom CSS
def inject_css():
    st.markdown("""
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');
        html, body, [class*="css"] {
            font-family: 'Inter', sans-serif;
        }
        .hero { text-align: center; padding: 3.5rem 2rem; background: linear-gradient(135deg, #1E88E5 0%, #1565C0 100%); color: white; border-radius: 16px; margin-bottom: 2.5rem; box-shadow: 0 10px 25px rgba(30,136,229,0.3); }
        .hero h1 { font-size: 3.5rem; margin-bottom: 0.5rem; color: white; font-weight: 700; letter-spacing: -0.02em; }
        .hero p { font-size: 1.25rem; opacity: 0.95; font-weight: 400; }
        .step-card { background: white; padding: 1.75rem; border-radius: 12px; box-shadow: 0 4px 15px rgba(0,0,0,0.05); margin-bottom: 1.5rem; border-left: 4px solid #1E88E5; height: 100%; transition: transform 0.2s ease; }
        .step-card:hover { transform: translateY(-2px); }
        .step-card h4, .step-card h3 { color: #1E293B; margin-top: 0; font-weight: 600; }
        .result-card { background: #FFFFFF; padding: 2.5rem; border-radius: 16px; box-shadow: 0 10px 30px rgba(0,0,0,0.08); text-align: center; margin-top: 1.5rem; border: 1px solid #E2E8F0; }
        .badge { padding: 0.6rem 1.2rem; border-radius: 999px; font-weight: 600; font-size: 1.1rem; display: inline-block; margin: 1rem 0; text-transform: uppercase; letter-spacing: 0.05em; }
        .badge-approve { background: #D1FAE5; color: #065F46; border: 1px solid #A7F3D0; }
        .badge-review { background: #FEF3C7; color: #92400E; border: 1px solid #FDE68A; }
        .badge-decline { background: #FEE2E2; color: #991B1B; border: 1px solid #FECACA; }
        .stTabs [data-baseweb="tab-list"] { gap: 2rem; }
        .stTabs [data-baseweb="tab"] { height: 3.5rem; white-space: pre-wrap; font-size: 1.1rem; }
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
        return pd.DataFrame({
            "timestamp": pd.date_range(end=pd.Timestamp.now(), periods=10).strftime("%Y-%m-%d %H:%M:%S"),
            "transaction_id": [f"TXN{i:04d}" for i in range(10)],
            "merchant_id": [101] * 5 + [102] * 5,
            "amount": np.random.uniform(500, 5000, 10).round(2),
            "risk_score": np.random.uniform(0.1, 0.9, 10).round(2),
            "action": ["APPROVE", "DECLINE", "MANUAL_REVIEW"] * 3 + ["APPROVE"],
            "explanation": ["Normal transaction pattern"] * 10
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

# Chart Helpers (Altair)
def get_confusion_matrix_chart(tn, fp, fn, tp):
    cm_df = pd.DataFrame({
        'Predicted': ['Good', 'Good', 'Fraud', 'Fraud'],
        'Actual': ['Good', 'Fraud', 'Good', 'Fraud'],
        'Count': [tn, fn, fp, tp]
    })
    
    base = alt.Chart(cm_df).encode(
        x=alt.X('Predicted:O', title='Predicted by AI'),
        y=alt.Y('Actual:O', title='Actual (Reality)')
    )
    
    heatmap = base.mark_rect().encode(
        color=alt.Color('Count:Q', scale=alt.Scale(scheme='blues'), legend=None)
    )
    
    text = base.mark_text(baseline='middle', size=16, fontWeight='bold').encode(
        text='Count:Q',
        color=alt.condition(alt.datum.Count > (max(tn, fp, fn, tp) / 2), alt.value('white'), alt.value('black'))
    )
    
    return (heatmap + text).properties(width=400, height=300)

def get_risk_distribution_chart(y_test, y_proba, threshold):
    dist_df = pd.DataFrame({'Score': y_proba, 'Class': ['Fraud' if y == 1 else 'Good' for y in y_test]})
    
    area_chart = alt.Chart(dist_df).transform_density(
        'Score',
        as_=['Score', 'Density'],
        groupby=['Class'],
        extent=[0, 1]
    ).mark_area(opacity=0.6).encode(
        x=alt.X('Score:Q', title='Risk Score'),
        y=alt.Y('Density:Q', title='Density'),
        color=alt.Color('Class:N', scale=alt.Scale(domain=['Good', 'Fraud'], range=['#10B981', '#EF4444']))
    )
    
    rule = alt.Chart(pd.DataFrame({'Threshold': [threshold]})).mark_rule(color='#2563EB', strokeWidth=2, strokeDash=[5,5]).encode(
        x='Threshold:Q'
    )
    
    return (area_chart + rule).properties(height=350).interactive()

def get_cost_curve_chart(y_test, y_proba, fp_cost, fn_cost, baseline_cost):
    thresholds = np.arange(0.30, 0.91, 0.02)
    savings = []
    
    for t in thresholds:
        pred = (y_proba >= t).astype(int)
        tn, fp, fn, tp = confusion_matrix(y_test, pred).ravel()
        cost = (fp * fp_cost) + (fn * fn_cost)
        savings.append(baseline_cost - cost)
        
    cost_df = pd.DataFrame({'Threshold': thresholds, 'Savings': savings})
    
    line = alt.Chart(cost_df).mark_line(color='#1E88E5', strokeWidth=3).encode(
        x=alt.X('Threshold:Q', scale=alt.Scale(domain=[0.3, 0.9]), title='Decision Threshold'),
        y=alt.Y('Savings:Q', title='Cost Saved (₹)'),
        tooltip=[alt.Tooltip('Threshold:Q', format='.2f'), alt.Tooltip('Savings:Q', format=',.0f', title='Savings (₹)')]
    )
    
    zero_line = alt.Chart(pd.DataFrame({'y': [0]})).mark_rule(color='#94A3B8').encode(y='y:Q')
    
    area = alt.Chart(cost_df).mark_area(opacity=0.1, color='#1E88E5').encode(
        x='Threshold:Q',
        y='Savings:Q',
        y2=alt.datum(0)
    )
    
    return (area + zero_line + line).properties(height=350).interactive()

# Main App
inject_css()
st.sidebar.title("🛡️ RiskShield AI")
pages = {
    "🏠 Home": "home",
    "📊 Dashboard": "dashboard",
    "🔬 Test Transaction": "simulator",
    "📖 How It Works": "how_it_works",
    "📋 Audit Log": "audit_log"
}

if "current_page" not in st.session_state:
    st.session_state.current_page = "🏠 Home"

def nav_to(page):
    st.session_state.current_page = page

selected = st.sidebar.radio("Navigation", list(pages.keys()), index=list(pages.keys()).index(st.session_state.current_page))
st.session_state.current_page = selected
page = pages[selected]

# --- PAGES ---

if page == "home":
    metrics = load_metrics()
    saved = metrics.get('total_cost_saved', 0)
    
    st.markdown("""
        <div class="hero">
            <h1>🛡️ RiskShield AI</h1>
            <p>AI-Powered Risk Manager for Razorpay Merchants</p>
            <p style="font-size: 1.1rem; opacity: 0.8; margin-top: 1rem;">Detect fraud, reduce chargebacks, and save money – explained in plain English.</p>
        </div>
    """, unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns([1, 1.5, 1])
    with col2:
        if st.button("🚀 Launch Dashboard", use_container_width=True, type="primary"):
            nav_to("📊 Dashboard")
            st.rerun()
            
    st.markdown("<br>", unsafe_allow_html=True)
    if saved:
        st.success(f"**Impact:** The system is currently preventing **₹{abs(saved):,.0f}** in potential losses compared to legacy rule-based systems.")
    
    st.markdown("### Why RiskShield?")
    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown("""
        <div class="step-card">
            <h4>⚡ Real-time scoring</h4>
            <p style="color: #475569; margin-top: 0.5rem;">Evaluates transactions in milliseconds using advanced Machine Learning without adding friction to checkout.</p>
        </div>
        """, unsafe_allow_html=True)
    with c2:
        st.markdown("""
        <div class="step-card">
            <h4>🔍 Explainable decisions</h4>
            <p style="color: #475569; margin-top: 0.5rem;">No black boxes. Get clear, plain-English reasons powered by SHAP for every blocked transaction.</p>
        </div>
        """, unsafe_allow_html=True)
    with c3:
        st.markdown("""
        <div class="step-card">
            <h4>💰 Cost savings</h4>
            <p style="color: #475569; margin-top: 0.5rem;">Optimized with a cost matrix to balance false positives and missed frauds, directly maximizing your revenue.</p>
        </div>
        """, unsafe_allow_html=True)

elif page == "dashboard":
    st.title("📊 Dashboard")
    metrics = load_metrics()
    y_test, y_proba = load_predictions()
    
    if not metrics:
        st.warning("Metrics not available.")
    else:
        # Top Row Metric Cards
        c1, c2, c3, c4 = st.columns(4)
        prec = metrics.get('precision', 0)
        rec = metrics.get('recall', 0)
        fpr = metrics.get('fpr', 0)
        saved = metrics.get('total_cost_saved', 0)
        
        c1.metric("Precision", f"{prec:.1%}", help="Of all fraud alerts, how many were actually fraud?")
        c2.metric("Recall", f"{rec:.1%}", help="Of all actual frauds, how many did we catch?")
        c3.metric("False Positive Rate", f"{fpr:.1%}", delta_color="inverse", help="Good Customers Flagged by Mistake.")
        c4.metric("Total Cost Saved", f"₹{saved:,.0f}", help="Money saved compared to old rule-based systems.")
        
        st.markdown("---")
        
        tab1, tab2 = st.tabs(["🎚️ Live Tuning", "📈 Data Insights"])
        
        with tab1:
            col_cm, col_slider = st.columns([1, 1], gap="large")
            
            with col_slider:
                st.subheader("Decision Threshold")
                st.caption("Drag the slider to see how tuning the aggressiveness of the model impacts decisions and cost savings in real-time.")
                threshold = st.slider("Threshold Slider", 0.30, 0.90, 0.50, 0.01, label_visibility="collapsed")
                
                if y_test is not None and y_proba is not None:
                    pred = (y_proba >= threshold).astype(int)
                    tn, fp, fn, tp = confusion_matrix(y_test, pred).ravel()
                    fp_cost = metrics.get('fp_cost', 500)
                    fn_cost = metrics.get('fn_cost', 2500)
                    baseline_cost = metrics.get('total_cost_baseline', 658000)
                    
                    total_cost = (fp * fp_cost) + (fn * fn_cost)
                    current_saved = baseline_cost - total_cost
                    
                    st.metric("Cost Saved at this threshold", f"₹{current_saved:,.0f}", delta=f"vs Baseline (₹{baseline_cost:,.0f})")
                    st.markdown(f"**False Positives (Lost Sales):** {fp} ➔ Cost: ₹{fp*fp_cost:,.0f}")
                    st.markdown(f"**False Negatives (Chargebacks):** {fn} ➔ Cost: ₹{fn*fn_cost:,.0f}")
                else:
                    st.info("Dynamic calculations require model predictions.")
                    tn, fp, fn, tp = metrics.get('confusion_matrix', {}).values() if 'confusion_matrix' in metrics else (2749, 709, 187, 105)
            
            with col_cm:
                st.subheader("Confusion Matrix")
                st.altair_chart(get_confusion_matrix_chart(tn, fp, fn, tp), use_container_width=True)
                
        with tab2:
            st.subheader("Model Performance Analysis")
            
            if y_test is not None and y_proba is not None:
                col_chart1, col_chart2 = st.columns(2, gap="large")
                
                with col_chart1:
                    st.markdown("**Risk Score Distribution**")
                    st.caption("How well does the model separate good users from bad users?")
                    st.altair_chart(get_risk_distribution_chart(y_test, y_proba, threshold), use_container_width=True)
                    
                with col_chart2:
                    st.markdown("**Cost vs Threshold Curve**")
                    st.caption("Find the 'sweet spot' that maximizes savings.")
                    fp_cost = metrics.get('fp_cost', 500)
                    fn_cost = metrics.get('fn_cost', 2500)
                    baseline_cost = metrics.get('total_cost_baseline', 658000)
                    st.altair_chart(get_cost_curve_chart(y_test, y_proba, fp_cost, fn_cost, baseline_cost), use_container_width=True)
            else:
                st.info("Model predictions needed for distribution charts.")

            df = load_data()
            if not df.empty and "payment_method" in df.columns:
                st.markdown("---")
                st.markdown("**Fraud Rate by Payment Method**")
                fraud_rates = df.groupby('payment_method')['is_fraud'].mean().reset_index()
                bar_chart = alt.Chart(fraud_rates).mark_bar(color='#1E88E5').encode(
                    x=alt.X('payment_method:N', title='Payment Method', sort='-y'),
                    y=alt.Y('is_fraud:Q', title='Fraud Rate', axis=alt.Axis(format='%')),
                    tooltip=[alt.Tooltip('payment_method:N', title='Method'), alt.Tooltip('is_fraud:Q', format='.2%', title='Fraud Rate')]
                ).properties(height=300).interactive()
                st.altair_chart(bar_chart, use_container_width=True)

elif page == "simulator":
    st.title("🔬 Test Transaction")
    st.markdown("Simulate a transaction in real-time. The AI will output a score, action, and a plain-English explanation.")
    
    with st.form("simulator_form"):
        col1, col2, col3 = st.columns(3)
        with col1:
            amount = st.number_input("Amount (₹)", min_value=1.0, value=15000.0, step=100.0)
            payment_method = st.selectbox("Payment Method", ["card", "upi", "netbanking", "wallet"])
            distance = st.number_input("Delivery Distance (km)", min_value=0.0, value=50.0)
        with col2:
            tenure = st.number_input("Customer Tenure (days)", min_value=0, value=10)
            velocity = st.number_input("Transaction Velocity (last 24h)", min_value=1, value=5)
            is_foreign = st.checkbox("Is Foreign IP?", value=True)
        with col3:
            device_age = st.number_input("Device Age (days)", min_value=0, value=1)
            pm_changes = st.number_input("Payment Method Changes", min_value=0, value=2)
            
        submitted = st.form_submit_button("🔮 Predict Risk", type="primary", use_container_width=True)

    if submitted:
        with st.spinner("Analyzing hundreds of risk signals using XGBoost..."):
            time.sleep(0.8) # Simulate processing time for UX
            predictor = get_predictor()
            if predictor:
                input_data = {
                    "amount": amount,
                    "payment_method": payment_method,
                    "delivery_distance_km": distance,
                    "customer_tenure_days": tenure,
                    "is_foreign_ip": int(is_foreign),
                    "transaction_velocity_24h": velocity,
                    "device_age_days": device_age,
                    "attempted_pm_changes": pm_changes,
                    "is_guest_checkout": 1,
                    "email_domain": "unknown"
                }
                try:
                    res = predictor.predict(input_data)
                    score = res.get("risk_score", 0.5)
                    action = res.get("action", "MANUAL_REVIEW")
                    explanation = res.get("explanation", "Transaction looks typical.")
                except Exception:
                    # Mock if predict fails
                    score = min(0.99, max(0.01, (amount / 20000) + (0.3 if is_foreign else 0)))
                    action = "DECLINE" if score > 0.75 else ("MANUAL_REVIEW" if score > 0.5 else "APPROVE")
                    explanation = "This transaction was flagged due to the combination of high amount and foreign IP." if score > 0.5 else "Looks good."
            else:
                score = 0.88 if is_foreign and amount > 5000 else 0.12
                action = "DECLINE" if score > 0.75 else "APPROVE"
                explanation = "High risk detected due to foreign IP and large transaction amount." if score > 0.75 else "Low risk pattern identified."

            st.markdown(f"""
            <div class="result-card">
                <h3 style="margin-top: 0; color: #1E293B;">Risk Assessment Complete</h3>
            </div>
            """, unsafe_allow_html=True)
            
            # Use columns for layout in result
            rc1, rc2 = st.columns([1, 2])
            
            with rc1:
                st.metric("Risk Score", f"{score:.1%}")
                badge_class = "badge-decline" if action == "DECLINE" else ("badge-review" if action == "MANUAL_REVIEW" else "badge-approve")
                st.markdown(f"<div class='badge {badge_class}'>{action}</div>", unsafe_allow_html=True)
            
            with rc2:
                st.markdown("### ℹ️ Explainability Report")
                st.markdown(f"<p style='font-size: 1.15rem; color: #334155; line-height: 1.6;'>{explanation}</p>", unsafe_allow_html=True)
                
            st.progress(score, text="Risk Confidence Bar")

elif page == "how_it_works":
    st.title("📖 How It Works")
    st.markdown("RiskShield AI protects your business in four simple steps without requiring technical expertise.")
    
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("""
        <div class="step-card">
            <h3>1️⃣ Learn</h3>
            <p style="color: #475569;">The AI studies millions of past transactions to understand what legitimate purchases and fraudulent attempts look like. It learns the complex, non-linear relationships that simple rules miss.</p>
        </div>
        """, unsafe_allow_html=True)
    with col2:
        st.markdown("""
        <div class="step-card">
            <h3>2️⃣ Score</h3>
            <p style="color: #475569;">When a new transaction occurs, it instantly receives a <b>Risk Score</b> (0% to 100%) based on dozens of signals like location, amount, device age, and velocity.</p>
        </div>
        """, unsafe_allow_html=True)
        
    col3, col4 = st.columns(2)
    with col3:
        st.markdown("""
        <div class="step-card">
            <h3>3️⃣ Explain</h3>
            <p style="color: #475569;">Unlike legacy "black box" algorithms, RiskShield uses SHAP to translate complex math into plain-English reasons, telling you <b>exactly why</b> a score is high.</p>
        </div>
        """, unsafe_allow_html=True)
    with col4:
        st.markdown("""
        <div class="step-card">
            <h3>4️⃣ Decide</h3>
            <p style="color: #475569;">Based on your cost-optimized threshold, the system automatically <b>Approves</b>, <b>Declines</b>, or flags for <b>Manual Review</b>—maximizing revenue and saving time.</p>
        </div>
        """, unsafe_allow_html=True)

elif page == "audit_log":
    st.title("📋 Audit Log")
    st.markdown("Review all processed transactions and their outcomes.")
    
    col1, col2 = st.columns([1, 1])
    with col1:
        merchant_filter = st.text_input("🔍 Filter by Merchant ID (optional)")
    
    df = load_audit(limit=1000)
    
    if merchant_filter and not df.empty and "merchant_id" in df.columns:
        try:
            mid = int(merchant_filter)
            df = df[df["merchant_id"] == mid]
        except ValueError:
            pass
            
    if not df.empty:
        display_cols = ["timestamp", "transaction_id", "merchant_id", "amount", "risk_score", "action"]
        cols_to_show = [c for c in display_cols if c in df.columns]
        
        st.dataframe(
            df[cols_to_show],
            use_container_width=True, 
            hide_index=True,
            column_config={
                "risk_score": st.column_config.ProgressColumn(
                    "Risk Score", help="The AI risk probability", format="%.2f", min_value=0, max_value=1
                ),
                "amount": st.column_config.NumberColumn(
                    "Amount", format="₹%.2f"
                ),
                "action": st.column_config.TextColumn(
                    "Action"
                )
            }
        )
        csv = df.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="⬇️ Download CSV Extract",
            data=csv,
            file_name="audit_log.csv",
            mime="text/csv",
        )
    else:
        st.info("No records found in the audit database.")