import streamlit as st
import numpy as np
import pandas as pd
import xgboost as xgb
import shap
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

# Configures the web page style
st.set_page_config(page_title="MSME Credit Health Card", page_icon="🏦", layout="wide")

# =====================================================================
# 1. CORE BACKEND DATA & MACHINE LEARNING MACHINE (CACHED)
# =====================================================================
@st.cache_resource
def load_and_train_credit_engine():
    """Generates the underlying data and builds a verified scoring system."""
    np.random.seed(42)
    n_samples = 1200
    
    # Raw alternative network data collection mapping
    data = {
        'aa_avg_daily_balance_inr': np.random.exponential(scale=150000, size=n_samples) + 20000,
        'aa_inflow_outflow_ratio': np.random.normal(loc=1.05, scale=0.15, size=n_samples),
        'aa_fund_insufficient_bounces_3m': np.random.poisson(lam=0.8, size=n_samples),
        'gst_monthly_turnover_inr': np.random.exponential(scale=500000, size=n_samples) + 50000,
        'gst_buyer_concentration_ratio': np.random.beta(a=2, b=5, size=n_samples), 
        'gst_filing_delay_days_avg': np.random.poisson(lam=3, size=n_samples),
        'upi_tx_volume_monthly': np.random.randint(50, 2000, size=n_samples),
        'upi_ticket_size_avg_inr': np.random.normal(loc=350, scale=120, size=n_samples),
        'epfo_employee_count': np.random.randint(2, 50, size=n_samples),
        'epfo_payment_punctuality_score': np.random.uniform(0.5, 1.0, size=n_samples)
    }
    df = pd.DataFrame(data)
    
    # Calculate non-linear business defaults
    risk_score = (
        (df['aa_fund_insufficient_bounces_3m'] * 0.4) + 
        (df['gst_buyer_concentration_ratio'] * 2.0) +
        (df['gst_filing_delay_days_avg'] * 0.15) -
        (df['aa_inflow_outflow_ratio'] * 1.5) -
        (df['epfo_payment_punctuality_score'] * 1.0)
    )
    threshold = np.percentile(risk_score, 88)
    df['is_default'] = (risk_score >= threshold).astype(int)
    
    X = df.drop(columns=['is_default'])
    y = df['is_default']
    
    # Train robust classifier
    model = xgb.XGBClassifier(
        n_estimators=150, max_depth=5, learning_rate=0.05,
        scale_pos_weight=7, random_state=42, eval_metric='logloss'
    )
    model.fit(X, y)
    explainer = shap.TreeExplainer(model)
    return model, explainer, X.columns.tolist()

# Run background setup safely
model, explainer, feature_names = load_and_train_credit_engine()

# Layman terminology translator dictionary for explainable charts
layman_translation = {
    'aa_avg_daily_balance_inr': 'Average Bank Balance',
    'aa_inflow_outflow_ratio': 'Money In vs Money Out Ratio',
    'aa_fund_insufficient_bounces_3m': 'Cheque Bounces (Low Funds)',
    'gst_monthly_turnover_inr': 'Monthly Sales Turnover',
    'gst_buyer_concentration_ratio': 'Risk of Depending on Single Buyer',
    'gst_filing_delay_days_avg': 'Tax Filing Delay Days',
    'upi_tx_volume_monthly': 'Monthly UPI Sales Volume',
    'upi_ticket_size_avg_inr': 'Average Customer Bill Size',
    'epfo_employee_count': 'Employee Headcount Size',
    'epfo_payment_punctuality_score': 'Staff Salary Fund Punctuality'
}

# =====================================================================
# 2. APPRAISAL LAYOUT DESIGN (STREAMLIT FRONTEND)
# =====================================================================
st.title("🏦 AI-Driven MSME Financial Health Passport")
st.markdown("Designed for **Track 03: Financial Inclusion & Digital Lending**. This dashboard translates alternate business metrics (GST, UPI, Bank Records) into an instant credit decision tool that anyone can understand.")
st.markdown("---")

# Split screen into interactive controls on left, health card results on right
col_sidebar, col_card = st.columns([1, 1.2])

with col_sidebar:
    st.subheader("📡 Step 1: Input Business Metrics")
    st.caption("Change these values to simulate pulling real data via Account Aggregators or Tax Portals.")
    
    # Section A
    with st.expander("💼 Bank Account Framework (Account Aggregator)", expanded=True):
        input_balance = st.number_input("Average Daily Balance kept in Bank (INR)", min_value=0, value=145000, step=5000)
        input_ratio = st.slider("Money Inflow vs Outflow Ratio (Target above 1.0x)", 0.5, 2.0, 1.20, 0.05)
        input_bounces = st.number_input("Cheque Bounces due to low funds (Last 3 Months)", min_value=0, max_value=12, value=0)
        
    # Section B
    with st.expander("📜 Tax & Sales Records (GST Portal)", expanded=True):
        input_turnover = st.number_input("Average Monthly Sales/Turnover (INR)", min_value=0, value=520000, step=10000)
        input_conc = st.slider("Dependency Risk (High means depending on 1 buyer)", 0.0, 1.0, 0.20, 0.05)
        input_delay = st.number_input("Average Tax Filing Delay (Days)", min_value=0, max_value=30, value=1)
        
    # Section C
    with st.expander("📱 Everyday Digital Operations (UPI & Employee Data)", expanded=True):
        input_upi_vol = st.number_input("Total UPI/QR Code Sales Transactions per Month", min_value=0, value=650)
        input_upi_size = st.number_input("Average Bill Amount per Customer (INR)", min_value=10, value=420)
        input_epfo_staff = st.number_input("Active Registered Staff Count", min_value=0, value=8)
        input_epfo_score = st.slider("Staff Provident Fund Payment Timeliness (1.0 = Perfect)", 0.0, 1.0, 0.95, 0.05)

# Convert all frontend interface inputs into our backend numeric execution matrix array
profile_payload = pd.DataFrame([{
    'aa_avg_daily_balance_inr': float(input_balance),
    'aa_inflow_outflow_ratio': float(input_ratio),
    'aa_fund_insufficient_bounces_3m': int(input_bounces),
    'gst_monthly_turnover_inr': float(input_turnover),
    'gst_buyer_concentration_ratio': float(input_conc),
    'gst_filing_delay_days_avg': int(input_delay),
    'upi_tx_volume_monthly': int(input_upi_vol),
    'upi_ticket_size_avg_inr': float(input_upi_size),
    'epfo_employee_count': int(input_epfo_staff),
    'epfo_payment_punctuality_score': float(input_epfo_score)
}])

# =====================================================================
# 3. MATHEMATIC PROCESSING ENGINE WITH ROBUST ARRAYS
# =====================================================================
# 🔥 CRITICAL FIX: Slices row 0, column 1 explicitly to force a clean Python scalar
prob_default = float(model.predict_proba(profile_payload)[0, 1])
health_score = int(300 + (1 - prob_default) * 600)

# Determine final status conditions
if health_score >= 750:
    tier, color, alert_box = "EXCELLENT FINANCIAL HEALTH", "#2ecc71", st.success
    nudge = "Business profile is in perfect standing. Ready for instant, pre-approved loan disbursement with zero paperwork via ULI network protocols."
elif health_score >= 650:
    tier, color, alert_box = "GOOD STANDING", "#f1c40f", st.warning
    nudge = "Business profile is stable. Approved under standard interest configurations. Micro-advances can be extended immediately through OCEN lending networks."
else:
    tier, color, alert_box = "HIGH RISK (NEEDS INVESTIGATION)", "#e74c3c", st.error
    nudge = "Business profile shows signs of financial stress. To raise this score, focus on eliminating fund-based cheque bounces and filing taxes on time for the next 60 days."

# =====================================================================
# 4. RENDER FINANCIAL HEALTH CARD VIEW PANEL (RIGHT SIDE)
# =====================================================================
with col_card:
    st.subheader("💳 Step 2: Live Credit Card Passport Results")
    
    # Visual metrics block displays
    metric_1, metric_2 = st.columns(2)
    with metric_1:
        st.metric(label="Your Financial Health Score", value=f"{health_score} / 900")
    with metric_2:
        st.metric(label="Estimated Risk Level", value=f"{prob_default * 100:.2f}%")
        
    # Render interactive status color banner
    alert_box(f"🎯 **System Assessment Tier Status:** {tier}")
    
    st.markdown("---")
    st.subheader("⚖️ Why is my score this number? (Plain English Insights)")
    st.caption("Our system looks behind the black box to show you exactly what is impacting your score profile direction.")
    
    # Process Explainability engine data tracking 1D elements safely
    shap_output = explainer(profile_payload)
    
    # 🔥 CRITICAL FIX: Target index [0] to extract the raw 1D array out of SHAP's matrix payload
    feature_impacts = dict(zip(feature_names, shap_output.values[0]))
    
    # Isolate top vectors based on pure positive vs negative scalar thresholds
    top_risks = sorted([item for item in feature_impacts.items() if item[1] > 0.001], key=lambda x: x[1], reverse=True)[:2]
    top_strengths = sorted([item for item in feature_impacts.items() if item[1] < -0.001], key=lambda x: x[1])[:2]
    
    col_str, col_risk = st.columns(2)
    with col_str:
        st.markdown("<p style='color:#2ecc71; font-weight:bold; font-size:16px;'>🌟 Factors Helping Your Score</p>", unsafe_allow_html=True)
        if top_strengths:
            for feat, val in top_strengths:
                st.markdown(f"✅ **{layman_translation[feat]}** is helping protect your financial reputation.")
        else:
            st.write("No major positive indicators found.")
            
    with col_risk:
        st.markdown("<p style='color:#e74c3c; font-weight:bold; font-size:16px;'>⚠️ Factors Hurting Your Score</p>", unsafe_allow_html=True)
        if top_risks:
            for feat, val in top_risks:
                st.markdown(f"❌ **{layman_translation[feat]}** is pulling down your loan eligibility score ranking.")
        else:
            st.write("Excellent! No active risk flags are dragging down your score.")
            
    # Display Actionable Next Steps
    st.markdown("---")
    st.subheader("💡 Automated Next Steps for the Business")
    st.info(nudge)
