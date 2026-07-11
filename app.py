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

    # Download button routine directly to sidebar
    st.markdown("---")
    st.subheader("📥 Download Project Dataset")
    st.caption("Click here to download the master 1,200 row synthetic credit database used to build this AI model.")

    @st.cache_data
    def generate_downloadable_csv():
        """Clean dataset generator block completely isolated from model tuple objects."""
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

    # Triggers file compiler safely
    csv_bytes = generate_downloadable_csv()

    st.download_button(
        label="📥 Download Master Synthetic Credit Data",
        data=csv_bytes,
        file_name="msme_alternate_credit_data.csv",
        mime="text/csv",
        use_container_width=True
    )

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
with col_card:
    st.subheader("📑 Step 2: Live Credit Card Passport Results")
    
    # Calculate raw default probability
    # model.predict_proba returns [prob_non_default, prob_default]
    prob_array = model.predict_proba(profile_payload)[0]
    default_prob = prob_array[1]
    non_default_prob = prob_array[0]
    
    # Scale non-default probability cleanly into the traditional financial range (300 to 900)
    computed_passport_score = int(300 + (non_default_prob * 600))
    
    # Apply baseline styling indicators based on risk categories
    if computed_passport_score >= 750:
        badge_status = "🟢 SYSTEM ASSESSMENT TIER STATUS: EXCELLENT FINANCIAL HEALTH"
    elif computed_passport_score >= 650:
        badge_status = "🟡 SYSTEM ASSESSMENT TIER STATUS: MODERATE FINANCIAL RISK"
    else:
        badge_status = "🔴 SYSTEM ASSESSMENT TIER STATUS: HIGH RISK VULNERABILITY"
        
    # Visual presentation layout parameters matching your screenshot
    st.write(f"### Your Financial Health Score: **{computed_passport_score} / 900**")
    st.write(f"Estimated Risk Level: **{default_prob * 100:.2f}%**")
    
    if computed_passport_score >= 750:
        st.success(badge_status)
    elif computed_passport_score >= 650:
        st.warning(badge_status)
    else:
        st.error(badge_status)
        
    st.progress((computed_passport_score - 300) / 600)
    
    st.markdown("---")
    st.subheader("⚙️ Why is my score this number? (Plain English Insights)")
    st.caption("Our system looks behind the black box to show you exactly what is impacting your score profile direction.")
    
    # Compute underlying SHAP explanation layers
    shap_values = explainer(profile_payload)
    
    # INVERSION FIX: Multiply SHAP values by -1 because we are predicting 
    # financial health (non-default) rather than default probability risk.
    raw_impacts = shap_values.values[0] * -1
    
    # Format and translate tracking labels
    chart_dataframe = pd.DataFrame({
        'Feature': [layman_translation[f] for f in feature_names],
        'Impact': raw_impacts
    })
    
    # Sort attributes to reveal dominant indicators
    chart_dataframe = chart_dataframe.sort_values(by='Impact', ascending=True)
    
    # Dynamic coloration mapping: Positive impacts are green (help), negative are red (hurt)
    chart_dataframe['Color'] = np.where(chart_dataframe['Impact'] >= 0, '#2ecc71', '#e74c3c')
    
    # Generate clean matplotlib horizontal bar graph
    fig, ax = plt.subplots(figsize=(6, 3.5))
    ax.barh(chart_dataframe['Feature'], chart_dataframe['Impact'], color=chart_dataframe['Color'])
    ax.axvline(0, color='black', linewidth=0.8, linestyle='--')
    ax.set_xlabel('Impact Contribution Weight Score')
    plt.tight_layout()
    st.pyplot(fig)
    
    st.markdown("---")
    
    # Split the sorted factors into helping vs hurting columns for your UI checkmarks
    col_help, col_hurt = st.columns(2)
    
    with col_help:
        st.markdown("### ☀️ Factors Helping Your Score")
        positive_factors = chart_dataframe[chart_dataframe['Impact'] > 0]
        if not positive_factors.empty:
            for _, row in positive_factors.iterrows():
                st.success(f"✅ {row['Feature']}")
        else:
            st.caption("No positive metric adjustments recorded.")
            
    with col_hurt:
        st.markdown("### ⚠️ Factors Hurting Your Score")
        negative_factors = chart_dataframe[chart_dataframe['Impact'] < 0]
        if not negative_factors.empty:
            for _, row in negative_factors.iterrows():
                st.error(f"❌ {row['Feature']}")
        else:
            st.caption("No critical risk points dragging down score.")

    st.markdown("---")
    st.subheader("💡 Automated Next Steps for the Business")
    if computed_passport_score >= 750:
        st.info("Business profile is in perfect standing. Ready for instant, pre-approved loan disbursement with zero paperwork via ULI network protocols.")
    else:
        st.warning("Score profile requires optimization. We recommend increasing average bank balances and reducing tax filing delays before applying through OCEN aggregators.")

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
    
    # Target index to extract the raw 1D array out of SHAP's matrix payload
    feature_impacts = dict(zip(feature_names, shap_output.values[0]))
    
    # Isolate top vectors based on pure positive vs negative scalar thresholds
    top_risks = sorted([item for item in feature_impacts.items() if item[1] > 0.001], key=lambda x: x[1], reverse=True)[:2]
    top_strengths = sorted([item for item in feature_impacts.items() if item[1] < -0.001], key=lambda x: x[1])[:2]
    
    # 🎨 --- NEW DYNAMIC MATPLOTLIB VISUALIZATION GENERATOR ---
    # This automatically compiles the exact mathematical drivers into a clean, horizontal visual bar chart
    plot_features = []
    plot_weights = []
    plot_colors = []
    
    # Add risks (pointing right or colored red to show negative impact to the borrower)
    for name, weight in reversed(top_risks):
        plot_features.append(layman_translation[name])
        plot_weights.append(weight)
        plot_colors.append('#e74c3c')  # Bold warning red
        
    # Add strengths (pointing left or colored green to show positive protective impact)
    for name, weight in top_strengths:
        plot_features.append(layman_translation[name])
        plot_weights.append(weight)
        plot_colors.append('#2ecc71')  # Emerald green
        
    if plot_features:
        fig, ax = plt.subplots(figsize=(7, 3))
        fig.patch.set_facecolor('#ffffff')
        ax.set_facecolor('#ffffff')
        
        # Eliminate background box borders for an elegant, modern web interface look
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['left'].set_color('#bdc3c7')
        ax.spines['bottom'].set_color('#bdc3c7')
        
        # Build horizontal bar layout chart
        bars = ax.barh(plot_features, plot_weights, color=plot_colors, height=0.4)
        ax.axvline(x=0, color='#7f8c8d', linestyle='--', linewidth=0.8) # Base equilibrium line
        ax.set_xlabel("Impact Contribution Weight Score", fontsize=9, color="#34495e")
        ax.tick_params(axis='both', which='major', labelsize=9, colors='#2c3e50')
        
        # Put clean numeric value anchors directly beside the bars
        for bar in bars:
            width = bar.get_width()
            ha_val = 'left' if width >= 0 else 'right'
            offset = 0.002 if width >= 0 else -0.002
            ax.text(width + offset, bar.get_y() + bar.get_height()/2, f'{width:+.3f}', 
                    va='center', ha=ha_val, fontsize=8, weight='bold', color='#34495e')
            
        # Pushes the live Matplotlib plot engine context directly into the Streamlit layout canvas
        st.pyplot(fig)
    # 🎨 --- VISUALIZATION GENERATOR ENDS HERE ---

    st.markdown("---")
    col_str, col_risk = st.columns(2)
    with col_str:
        st.markdown("🌟 **Factors Helping Your Score**")
        if top_strengths:
            for feat, val in top_strengths:
                st.markdown(f"✅ {layman_translation[feat]}")
        else:
            st.write("No major positive indicators found.")
            
    with col_risk:
        st.markdown("⚠️ **Factors Hurting Your Score**")
        if top_risks:
            for feat, val in top_risks:
                st.markdown(f"❌ {layman_translation[feat]}")
        else:
            st.write("Excellent! No active risk flags are dragging down your score.")
            
    # Display Actionable Next Steps
    st.markdown("---")
    st.subheader("💡 Automated Next Steps for the Business")
    st.info(nudge)
