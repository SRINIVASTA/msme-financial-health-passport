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
# 1. CORE BACKEND DATA & MACHINE LEARNING MACHINE (ADAPTIVE ENGINE)
# =====================================================================
def train_custom_credit_engine(custom_df=None):
    """Trains or updates model parameters using custom uploaded data or baseline defaults."""
    if custom_df is not None:
        # Step A: Create a clean copy and drop any completely blank or missing formatting lines (Fixes 'nan' Bug)
        df = custom_df.copy()
        
        # Step B: Dynamically inject the target metric logic if the bank table lacks a supervised classification flag
        if 'is_default' not in df.columns:
            # Safe numeric conversion step to prevent evaluation calculation issues
            for col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
            df = df.dropna() # Clears out any corrupted or incomplete lines instantly
            
            risk_score = (
                (df['aa_fund_insufficient_bounces_3m'] * 0.4) + 
                (df['gst_buyer_concentration_ratio'] * 2.0) +
                (df['gst_filing_delay_days_avg'] * 0.15) -
                (df['aa_inflow_outflow_ratio'] * 1.5) -
                (df['epfo_payment_punctuality_score'] * 1.0) -
                ((df['epfo_employee_count'] / 50.0) * 0.5)
            )
            threshold = np.percentile(risk_score, 88)
            df['is_default'] = (risk_score >= threshold).astype(int)
        else:
            # If the column is already present, convert to numeric values and drop any hanging empty row endings
            df['is_default'] = pd.to_numeric(df['is_default'], errors='coerce')
            df = df.dropna(subset=['is_default'])
            df['is_default'] = df['is_default'].astype(int)
    else:
        # Fallback to structural synthetic generation baseline matrix parameters
        np.random.seed(42)
        n_samples = 1200
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
        risk_score = (
            (df['aa_fund_insufficient_bounces_3m'] * 0.4) + 
            (df['gst_buyer_concentration_ratio'] * 2.0) +
            (df['gst_filing_delay_days_avg'] * 0.15) -
            (df['aa_inflow_outflow_ratio'] * 1.5) -
            (df['epfo_payment_punctuality_score'] * 1.0) -
            ((df['epfo_employee_count'] / 50.0) * 0.5)
        )
        threshold = np.percentile(risk_score, 88)
        df['is_default'] = (risk_score >= threshold).astype(int)

    X = df.drop(columns=['is_default'], errors='ignore')
    target_features = [
        'aa_avg_daily_balance_inr', 'aa_inflow_outflow_ratio', 'aa_fund_insufficient_bounces_3m',
        'gst_monthly_turnover_inr', 'gst_buyer_concentration_ratio', 'gst_filing_delay_days_avg',
        'upi_tx_volume_monthly', 'upi_ticket_size_avg_inr', 'epfo_employee_count', 'epfo_payment_punctuality_score'
    ]
    X = X[target_features]
    y = df['is_default'].astype(int) # Enforces strict integer targets to prevent label inference issues
    
    # Train robust classifier parameters with monotonic tracking constraints
    constraints = (0, -1, 1, 0, 1, 1, 0, 0, -1, -1)
    model = xgb.XGBClassifier(
        n_estimators=150, max_depth=5, learning_rate=0.05,
        scale_pos_weight=7, random_state=42, eval_metric='logloss',
        monotone_constraints=constraints
    )
    model.fit(X, y)
    explainer = shap.TreeExplainer(model)
    return model, explainer, X.columns.tolist(), df

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

col_sidebar, col_card = st.columns([1, 1.2])

with col_sidebar:
    st.subheader("📥 Data Sync & Model Optimization Sandbox")
    st.caption("Optionally upload custom bank sheets to optimize model training rules, or download the simulation schema.")
    
    uploaded_bank_file = st.file_uploader(
        label="📤 Upload Bank Batch Update Data (CSV Format)",
        type=["csv"],
        help="Upload a dataset containing matching columns to overwrite baseline weights with customized telemetry."
    )
    
    is_using_custom_data = False
    if uploaded_bank_file is not None:
        try:
            user_imported_df = pd.read_csv(uploaded_bank_file)
            model, explainer, feature_names, active_dataset = train_custom_credit_engine(user_imported_df)
            is_using_custom_data = True
            st.success("🟢 Model parameters optimized successfully using uploaded database data!")
        except Exception as e:
            st.error(f"🔴 Processing Error: Check your format schema constraints. Error: {str(e)}")
            model, explainer, feature_names, active_dataset = train_custom_credit_engine(None)
    else:
        model, explainer, feature_names, active_dataset = train_custom_credit_engine(None)

    st.markdown("---")
    st.subheader("📡 Step 1: Input Business Metrics")
    st.caption("Change these values to simulate pulling real data via Account Aggregators or Tax Portals.")
    
    with st.expander("💼 Bank Account Framework (Account Aggregator)", expanded=True):
        input_balance = st.number_input("Average Daily Balance kept in Bank (INR)", min_value=0, value=145000, step=5000)
        input_ratio = st.slider("Money Inflow vs Outflow Ratio (Target above 1.0x)", 0.5, 2.0, 1.20, 0.05)
        input_bounces = st.number_input("Cheque Bounces due to low funds (Last 3 Months)", min_value=0, max_value=12, value=0)
        
    with st.expander("📜 Tax & Sales Records (GST Portal)", expanded=True):
        input_turnover = st.number_input("Average Monthly Sales/Turnover (INR)", min_value=0, value=520000, step=10000)
        input_conc = st.slider("Dependency Risk (High means depending on 1 buyer)", 0.0, 1.0, 0.20, 0.05)
        input_delay = st.number_input("Average Tax Filing Delay (Days)", min_value=0, max_value=30, value=1)
        
    with st.expander("📱 Everyday Digital Operations (UPI & Employee Data)", expanded=True):
        input_upi_vol = st.number_input("Total UPI/QR Code Sales Transactions per Month", min_value=0, value=650)
        input_upi_size = st.number_input("Average Bill Amount per Customer (INR)", min_value=10, value=420)
        input_epfo_staff = st.number_input("Active Registered Staff Count", min_value=0, value=8)
        input_epfo_score = st.slider("Staff Provident Fund Payment Timeliness (1.0 = Perfect)", 0.0, 1.0, 0.95, 0.05)

    st.markdown("---")
    
    @st.cache_data
    def generate_downloadable_csv():
        np.random.seed(42)
        n_samples = 1200
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
        temp_df = pd.DataFrame(data)
        return temp_df.to_csv(index=False).encode('utf-8')

    csv_bytes = generate_downloadable_csv()
    st.download_button(
        label="📥 Download Template Schema File",
        data=csv_bytes,
        file_name="msme_alternate_credit_data.csv",
        mime="text/csv",
        use_container_width=True
    )

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
with col_card:
    if is_using_custom_data:
        st.subheader("📋 Active Uploaded Bank Registry Database")
        st.caption("Below is the custom dataset provided by the user. The AI models have been updated on this specific format.")
        st.dataframe(active_dataset, use_container_width=True, height=220)
    else:
        st.subheader("💡 Active Registry Status")
        st.info("ℹ️ Running on baseline **Synthetic Databank Engine** (1,200 Rows). Upload a custom banking spreadsheet in the sidebar to change data environments.")
        
    st.markdown("---")
    st.subheader("📊 Step 2: Live Credit Card Passport Results")
    
    prob_output = model.predict_proba(profile_payload)
    default_probability = float(prob_output[0][1])
    non_default_probability = 1.0 - default_probability
    
    health_score = int(300 + (non_default_probability * 600))
    risk_level_pct = default_probability * 100
    
    if health_score >= 750:
        badge_status = "SYSTEM ASSESSMENT TIER STATUS: EXCELLENT FINANCIAL HEALTH"
        st.success(f"🟢 {badge_status}")
    elif health_score >= 650:
        badge_status = "SYSTEM ASSESSMENT TIER STATUS: MODERATE FINANCIAL RISK"
        st.warning(f"🟡 {badge_status}")
    else:
        badge_status = "SYSTEM ASSESSMENT TIER STATUS: HIGH RISK VULNERABILITY"
        st.error(f"🔴 {badge_status}")
        
    col_stat1, col_stat2 = st.columns(2)
    with col_stat1:
        st.metric(label="Your Financial Health Score", value=f"{health_score} / 900")
    with col_stat2:
        st.metric(label="Estimated Risk Level", value=f"{risk_level_pct:.2f}%")
        
    st.progress((health_score - 300) / 600)
    
    st.markdown("---")
    st.subheader("⚙️ Why is my score this number? (Plain English Insights)")
    st.caption("Our system looks behind the black box to show you exactly what is impacting your score profile direction.")
    
    shap_values = explainer(profile_payload)
    raw_impacts = shap_values.values[0] * -1
    
    chart_dataframe = pd.DataFrame({
        'Feature': [layman_translation[f] for f in feature_names],
        'Impact': raw_impacts
    })
    
    chart_dataframe = chart_dataframe.sort_values(by='Impact', ascending=True)
    chart_dataframe['Color'] = np.where(chart_dataframe['Impact'] >= 0, '#2ecc71', '#e74c3c')
    
    fig, ax = plt.subplots(figsize=(6, 3.5))
    ax.barh(chart_dataframe['Feature'], chart_dataframe['Impact'], color=chart_dataframe['Color'])
    ax.axvline(0, color='black', linewidth=0.8, linestyle='--')
    ax.set_xlabel('Impact Contribution Weight Score')
    plt.tight_layout()
    st.pyplot(fig)
    
    st.markdown("---")
    
    # DYNAMIC COLUMNS FIX: Adjusted filtering bounds to avoid sorting errors
    col_help, col_hurt = st.columns(2)
    with col_help:
        st.markdown("#### ☀️ Factors Helping Your Score")
        positive_factors = chart_dataframe[chart_dataframe['Impact'] > 0.005]
        if not positive_factors.empty:
            for _, row in positive_factors.iterrows():
                st.markdown(f"✅ {row['Feature']}")
        else:
            st.caption("No strong positive metric drivers tracking currently.")
            
    with col_hurt:
        st.markdown("#### ⚠️ Factors Hurting Your Score")
        negative_factors = chart_dataframe[chart_dataframe['Impact'] < -0.005]
        if not negative_factors.empty:
            for _, row in negative_factors.iterrows():
                st.markdown(f"❌ {row['Feature']}")
        else:
            st.caption("No critical risk items degrading score profile.")

    st.markdown("---")
    st.subheader("💡 Automated Next Steps for the Business")
    if health_score >= 750:
        st.info("Business profile is in perfect standing. Ready for instant, pre-approved loan disbursement with zero paperwork via ULI network protocols.")
    else:
        st.warning("Score profile requires optimization. We recommend increasing average bank balances and reducing tax filing delays before applying through OCEN aggregators.")
