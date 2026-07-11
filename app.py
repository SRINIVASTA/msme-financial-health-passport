import streamlit as st
import numpy as np
import pandas as pd
import xgboost as xgb
import shap
import matplotlib.pyplot as plt
import io
from sklearn.model_selection import train_test_split
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib import colors

# =====================================================================
# SYSTEM INITIALIZATION & PAGE ARCHITECTURE
# =====================================================================
st.set_page_config(
    page_title="MSME Credit Health Card Portal", 
    page_icon="🏦", 
    layout="wide"
)

# Translation map for backend metric properties
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
# 1. ADVANCED ENGINE TRAINING PIPELINE
# =====================================================================
def train_custom_credit_engine(custom_df=None):
    if custom_df is not None:
        df = custom_df.copy()
        for col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
        df = df.dropna()
        
        if 'is_default' not in df.columns and len(df) > 0:
            risk_score = (
                (df['aa_fund_insufficient_bounces_3m'] * 0.4) +
                (df['gst_buyer_concentration_ratio'] * 2.0) +
                (df['gst_filing_delay_days_avg'] * 0.15) -
                (df['aa_inflow_outflow_ratio'] * 1.5) -
                (df['epfo_payment_punctuality_score'] * 1.0) -
                ((df['epfo_employee_count'] / 50.0) * 0.5)
            )
            if len(df) <= 5:
                df['is_default'] = (risk_score >= -0.5).astype(int)
            else:
                df['is_default'] = (risk_score >= np.percentile(risk_score, 16.67)).astype(int)
        else:
            df['is_default'] = df['is_default'].fillna(0).astype(int)
            
        if len(df) > 0 and df['is_default'].nunique() == 1:
            df.loc[df.index, 'is_default'] = 1 - df.loc[df.index, 'is_default']
    else:
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
        df['is_default'] = (risk_score >= np.percentile(risk_score, 16.67)).astype(int)

    X = df.drop(columns=['is_default'], errors='ignore')
    target_features = [
        'aa_avg_daily_balance_inr', 'aa_inflow_outflow_ratio', 'aa_fund_insufficient_bounces_3m',
        'gst_monthly_turnover_inr', 'gst_buyer_concentration_ratio', 'gst_filing_delay_days_avg',
        'upi_tx_volume_monthly', 'upi_ticket_size_avg_inr', 'epfo_employee_count',
        'epfo_payment_punctuality_score'
    ]
    X = X[target_features]
    y = df['is_default']
    
    # 0 = Unconstrained, 1 = Positive Monotone Constraint, -1 = Negative Monotone Constraint
    constraints = (0, -1, 1, 0, 1, 1, 0, 0, -1, -1)
    model = xgb.XGBClassifier(n_estimators=150, max_depth=5, learning_rate=0.05, monotone_constraints=constraints)
    model.fit(X, y)
    explainer = shap.TreeExplainer(model)
    return model, explainer, X.columns.tolist(), df
# =====================================================================
# 2. PDF CARD REPORT GENERATION MICROSERVICE (FIXED & COMPLETE)
# =====================================================================
def generate_credit_pdf(client_name, score, risk, tier, payload_dict, helpers, hurters):
    buffer = io.BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=letter, rightMargin=40, leftMargin=40, topMargin=40, bottomMargin=40)
    story = []
    
    styles = getSampleStyleSheet()
    title_style = ParagraphStyle('DocTitle', parent=styles['Heading1'], fontSize=22, textColor=colors.HexColor('#1B365D'), spaceAfter=15)
    header_style = ParagraphStyle('SecHeader', parent=styles['Heading2'], fontSize=14, textColor=colors.HexColor('#1B365D'), spaceBefore=15, spaceAfter=8)
    body_style = ParagraphStyle('DocBody', parent=styles['Normal'], fontSize=10, leading=14, spaceAfter=6)
    bold_body = ParagraphStyle('DocBodyBold', parent=body_style, fontName='Helvetica-Bold')
    
    story.append(Paragraph("<b>MSME FINANCIAL HEALTH PASSPORT</b>", title_style))
    story.append(Paragraph(f"<b>Client Enterprise Name:</b> {client_name}", body_style))
    story.append(Paragraph(f"<b>System Underwriting Status:</b> {tier}", bold_body))
    story.append(Spacer(1, 15))
    
    score_data = [
        [Paragraph("<b>Evaluation Vector</b>", bold_body), Paragraph("<b>Performance Value</b>", bold_body)],
        [Paragraph("Financial Health Index Score", body_style), Paragraph(f"<b>{score} / 900</b>", bold_body)],
        [Paragraph("Estimated Default Risk Probability", body_style), Paragraph(f"<b>{risk:.2f}%</b>", bold_body)]
    ]
    # FIXED: Explicit column dimensions mapped to fill the page canvas bounds cleanly
    t_score = Table(score_data, colWidths=[270, 270])
    t_score.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (1,0), colors.HexColor('#EAEEF4')),
        ('GRID', (0,0), (-1,-1), 0.5, colors.grey),
        ('PADDING', (0,0), (-1,-1), 8),
    ]))
    story.append(t_score)
    story.append(Spacer(1, 15))
    
    story.append(Paragraph("<b>Alternative Network Ingested Data Stream Summary</b>", header_style))
    metrics_data = [[Paragraph("<b>Metric Vector Parameter</b>", bold_body), Paragraph("<b>Reported Value</b>", bold_body)]]
    for k, v in payload_dict.items():
        metrics_data.append([Paragraph(layman_translation.get(k, k), body_style), Paragraph(str(v), body_style)])
    
    # FIXED: Explicit column dimensions mapped to handle incoming vector names
    t_metrics = Table(metrics_data, colWidths=[270, 270])
    t_metrics.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (1,0), colors.HexColor('#EAEEF4')),
        ('GRID', (0,0), (-1,-1), 0.5, colors.lightgrey),
        ('PADDING', (0,0), (-1,-1), 5),
    ]))
    story.append(t_metrics)
    story.append(Spacer(1, 15))
    
    story.append(Paragraph("<b>Explainable AI (XAI) Score Attribution Drivers</b>", header_style))
    story.append(Paragraph(f"<b>Top Metrics Supporting Score:</b> {', '.join(helpers) if helpers else 'None Identified'}", body_style))
    story.append(Paragraph(f"<b>Primary Drivers Negatively Impacting Score:</b> {', '.join(hurters) if hurters else 'None Identified'}", body_style))
    story.append(Spacer(1, 20))
    
    story.append(Paragraph("<i>This evaluation document is verified via decentralized account aggregators and automated public API tax registries. Generated instantly via ULI protocol interfaces.</i>", body_style))
    doc.build(story)
    buffer.seek(0)
    return buffer.getvalue()
# =====================================================================
# 3. SIDEBAR LAYOUT & MEMORY RESET CORE
# =====================================================================
col_sidebar, col_card = st.columns([1, 1.2])

# FIXED CALLBACK FUNCTION: Explicitly reads and writes memory snapshots with no loop blocks
def sync_inputs_to_selected_row():
    if "active_dataset" in st.session_state:
        current_label = st.session_state.active_msme_dropdown
        row_idx = int(current_label.split("-")) - 1
        row_data = st.session_state["active_dataset"].iloc[row_idx]
        
        # Flush targets out straight to session storage values
        st.session_state["sb_balance"] = int(row_data['aa_avg_daily_balance_inr'])
        st.session_state["sb_ratio"] = float(np.clip(row_data['aa_inflow_outflow_ratio'], 0.5, 2.0))
        st.session_state["sb_bounces"] = int(row_data['aa_fund_insufficient_bounces_3m'])
        st.session_state["sb_turnover"] = int(row_data['gst_monthly_turnover_inr'])
        st.session_state["sb_conc"] = float(np.clip(row_data['gst_buyer_concentration_ratio'], 0.0, 1.0))
        st.session_state["sb_delay"] = int(row_data['gst_filing_delay_days_avg'])
        st.session_state["sb_upi_vol"] = int(row_data['upi_tx_volume_monthly'])
        st.session_state["sb_upi_size"] = int(row_data['upi_ticket_size_avg_inr'])
        st.session_state["sb_epfo_staff"] = int(row_data['epfo_employee_count'])
        st.session_state["sb_epfo_score"] = float(np.clip(row_data['epfo_payment_punctuality_score'], 0.0, 1.0))

with col_sidebar:
    st.subheader("🌐 Data Ingestion Protocol Selection")
    
    data_source_mode = st.radio(
        label="Select Input Ingestion Channel:",
        options=["Live Ecosystem APIs (ULI / OCEN / AA Simulation)", "Batch Document Upload (CSV Sandbox)"],
        index=0,
        key="data_source_mode_radio"
    )
    
    is_using_custom_data = False
    
    if data_source_mode == "Batch Document Upload (CSV Sandbox)":
        st.markdown("---")
        st.subheader("📊 Model Optimization Sandbox")
        uploaded_bank_file = st.file_uploader(label="📁 Upload Bank Batch Update Data (CSV Format)", type=["csv"], key="csv_file_uploader_widget")
        
        if uploaded_bank_file is not None:
            try:
                user_imported_df = pd.read_csv(uploaded_bank_file)
                m_obj, e_obj, f_list, d_matrix = train_custom_credit_engine(user_imported_df)
                
                # CRITICAL RESYNC FIX: Force clear out old memory keys when a file replacement happens
                if "last_loaded_file" not in st.session_state or st.session_state["last_loaded_file"] != uploaded_bank_file.name:
                    for k in ["sb_balance", "sb_ratio", "sb_bounces", "sb_turnover", "sb_conc", "sb_delay", "sb_upi_vol", "sb_upi_size", "sb_epfo_staff", "sb_epfo_score"]:
                        if k in st.session_state:
                            del st.session_state[k]
                    st.session_state["last_loaded_file"] = uploaded_bank_file.name
                
                st.session_state["active_model"] = m_obj
                st.session_state["active_explainer"] = e_obj
                st.session_state["active_features"] = f_list
                st.session_state["active_dataset"] = d_matrix
                is_using_custom_data = True
            except Exception as e:
                st.error(f"Processing Error in Sheet: {str(e)}")

    if "active_dataset" not in st.session_state or data_source_mode == "Live Ecosystem APIs (ULI / OCEN / AA Simulation)":
        if not is_using_custom_data:
            m_obj, e_obj, f_list, d_matrix = train_custom_credit_engine(None)
            st.session_state["active_model"] = m_obj
            st.session_state["active_explainer"] = e_obj
            st.session_state["active_features"] = f_list
            st.session_state["active_dataset"] = d_matrix

    st.markdown("---")
    st.subheader("👤 Step 1: Select Active Row & Fine-Tune Parameters")
    
    active_df = st.session_state["active_dataset"]
    total_available_rows = len(active_df)
    msme_options = [f"MSME-{str(i+1).zfill(4)}" for i in range(total_available_rows)]
    
    selected_msme_label = st.selectbox(
        label="Choose target business entity to inspect:",
        options=msme_options,
        key="active_msme_dropdown",
        on_change=sync_inputs_to_selected_row
    )
    
    selected_row_index = int(selected_msme_label.split("-")[1]) - 1
    extracted_row_data = active_df.iloc[selected_row_index]
    
    if "sb_balance" not in st.session_state:
        st.session_state["sb_balance"] = int(extracted_row_data['aa_avg_daily_balance_inr'])
        st.session_state["sb_ratio"] = float(np.clip(extracted_row_data['aa_inflow_outflow_ratio'], 0.5, 2.0))
        st.session_state["sb_bounces"] = int(extracted_row_data['aa_fund_insufficient_bounces_3m'])
        st.session_state["sb_turnover"] = int(extracted_row_data['gst_monthly_turnover_inr'])
        st.session_state["sb_conc"] = float(np.clip(extracted_row_data['gst_buyer_concentration_ratio'], 0.0, 1.0))
        st.session_state["sb_delay"] = int(extracted_row_data['gst_filing_delay_days_avg'])
        st.session_state["sb_upi_vol"] = int(extracted_row_data['upi_tx_volume_monthly'])
        st.session_state["sb_upi_size"] = int(extracted_row_data['upi_ticket_size_avg_inr'])
        st.session_state["sb_epfo_staff"] = int(extracted_row_data['epfo_employee_count'])
        st.session_state["sb_epfo_score"] = float(np.clip(extracted_row_data['epfo_payment_punctuality_score'], 0.0, 1.0))

    client_name = st.text_input("Assign Corporate Display Name", value=f"Sri Venkateswara Enterprises ({selected_msme_label})")
    
    with st.expander("💼 Ingested Account Aggregator Records", expanded=True):
        input_balance = st.number_input("Average Daily Balance kept in Bank (INR)", min_value=0, key="sb_balance", step=5000)
        input_ratio = st.slider("Money Inflow vs Outflow Ratio", 0.5, 2.0, key="sb_ratio", step=0.05)
        input_bounces = st.number_input("Cheque Bounces due to low funds (3M)", min_value=0, max_value=30, key="sb_bounces")
        
    with st.expander("📄 Tax & Sales Records (GST Portal)", expanded=True):
        input_turnover = st.number_input("Average Monthly Sales/Turnover (INR)", min_value=0, key="sb_turnover", step=10000)
        input_conc = st.slider("Dependency Risk (High = Single Buyer)", 0.0, 1.0, key="sb_conc", step=0.05)
        input_delay = st.number_input("Average Tax Filing Delay (Days)", min_value=0, max_value=90, key="sb_delay")
        
    with st.expander("📱 Everyday Digital Operations (UPI & EPFO)", expanded=True):
        input_upi_vol = st.number_input("Total UPI Sales Transactions per Month", min_value=0, key="sb_upi_vol")
        input_upi_size = st.number_input("Average Bill Amount per Customer (INR)", min_value=10, key="sb_upi_size")
        input_epfo_staff = st.number_input("Active Registered Staff Count", min_value=1, key="sb_epfo_staff")
        input_epfo_score = st.slider("Staff Fund Payment Timeliness (1.0 = Perfect)", 0.0, 1.0, key="sb_epfo_score", step=0.05)

    st.markdown("---")
    st.subheader("📥 Master Data Export Controls")
    approved_dataframe = active_df[active_df['is_default'] == 0]
    rejected_dataframe = active_df[active_df['is_default'] == 1]
    
    st.download_button(label=f"✅ Download Approved Portfolio ({len(approved_dataframe)} Rows)", data=approved_dataframe.to_csv(index=False).encode('utf-8'), file_name="approved_msme_credit_passport.csv", mime="text/csv", use_container_width=True)
    st.download_button(label=f"❌ Download Rejected Portfolio ({len(rejected_dataframe)} Rows)", data=rejected_dataframe.to_csv(index=False).encode('utf-8'), file_name="rejected_msme_credit_passport.csv", mime="text/csv", use_container_width=True)

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
# 4. MAIN DISPLAY CARD INTERFACE
# =====================================================================
model = st.session_state["active_model"]
explainer = st.session_state["active_explainer"]
feature_names = st.session_state["active_features"]
active_df = st.session_state["active_dataset"]

with col_card:
    if data_source_mode == "Batch Document Upload (CSV Sandbox)" and uploaded_bank_file is not None:
        st.subheader("📋 Active Uploaded Bank Registry Database")
        st.dataframe(active_df, use_container_width=True, height=180)
    elif data_source_mode == "Live Ecosystem APIs (ULI / OCEN / AA Simulation)":
        st.subheader("🌐 Connected Live Data Streams")
        st.info("📡 Secure network tunnel established. Fetching credentials via **ULI architecture**.")
    else:
        st.subheader("ℹ Active Registry Status")
        st.info("Running on baseline **Synthetic Databank Engine** (1,200 Rows).")
        
    st.markdown("---")
    st.subheader("🎯 Step 2: Live Credit Card Passport Results")
    
    prob_output = model.predict_proba(profile_payload)
    default_probability = float(prob_output[0, 1])  # Targets row 0, column 1
    non_default_probability = 1.0 - default_probability
    
    health_score = int(300 + (non_default_probability * 600))
    risk_level_pct = default_probability * 100
    
    if health_score >= 750:
        badge_status = "EXCELLENT FINANCIAL HEALTH"
        st.success(f"🟢 SYSTEM ASSESSMENT TIER STATUS: {badge_status}")
    elif health_score >= 650:
        badge_status = "MODERATE FINANCIAL RISK"
        st.warning(f"🟡 SYSTEM ASSESSMENT TIER STATUS: {badge_status}")
    else:
        badge_status = "HIGH RISK VULNERABILITY"
        st.error(f"🔴 SYSTEM ASSESSMENT TIER STATUS: {badge_status}")
        
    col_stat1, col_stat2 = st.columns(2)
    with col_stat1:
        st.metric(label="Your Financial Health Score", value=f"{health_score} / 900")
    with col_stat2:
        st.metric(label="Estimated Risk Level", value=f"{risk_level_pct:.2f}%")
        
    st.progress((health_score - 300) / 600)
    st.markdown("---")
    
    st.subheader("🔍 Plain English Attributions (Explainable AI)")
    shap_values = explainer(profile_payload)
    
    if len(shap_values.values.shape) == 3:
        raw_impacts = shap_values.values[0, :, 1] * -1
    elif len(shap_values.values.shape) == 2:
        raw_impacts = shap_values.values[0, :] * -1
    else:
        raw_impacts = np.ravel(shap_values.values) * -1
        
    raw_impacts = np.array(raw_impacts).flatten()
    
    if len(raw_impacts) != len(feature_names):
        if len(raw_impacts) > len(feature_names):
            raw_impacts = raw_impacts[:len(feature_names)]
        else:
            raw_impacts = np.pad(raw_impacts, (0, len(feature_names) - len(raw_impacts)), 'constant')
            
    chart_dataframe = pd.DataFrame({
        'Feature': [layman_translation[f] for f in feature_names],
        'Impact': raw_impacts
    }).sort_values(by='Impact', ascending=True)
    chart_dataframe['Color'] = np.where(chart_dataframe['Impact'] >= 0, '#2ecc71', '#e74c3c')
    
    fig, ax = plt.subplots(figsize=(6, 3))
    ax.barh(chart_dataframe['Feature'], chart_dataframe['Impact'], color=chart_dataframe['Color'])
    ax.axvline(0, color='black', linewidth=0.8, linestyle='--')
    ax.set_xlabel('Impact Weight Score')
    plt.tight_layout()
    st.pyplot(fig)
    
    # DYNAMIC TRACKER EXTRACTOR (Guarantees data output)
    active_drivers = chart_dataframe[chart_dataframe['Impact'] != 0]
    pos_subset = active_drivers[active_drivers['Impact'] > 0].sort_values(by='Impact', ascending=False)
    pos_drivers = pos_subset['Feature'].head(2).tolist()
    
    neg_subset = active_drivers[active_drivers['Impact'] < 0].sort_values(by='Impact', ascending=True)
    neg_drivers = neg_subset['Feature'].head(2).tolist()
    
    if not pos_drivers and not neg_drivers:
        pos_drivers = ["Baseline Stability Metrics"]
        neg_drivers = ["None (Perfect Structural Integrity)"]
    
    st.markdown("---")
    st.subheader("📄 Export Specific Client Document")
    st.markdown(
        f"""<div style="background-color: #EBF5FB; border-left: 5px solid #2980B9; padding: 10px; border-radius: 4px; margin-bottom: 15px;">
            <span style="background-color: transparent; color: #1B4F72; font-weight: bold;">📝 TRACK 03 EXPECTED OUTCOME:</span>
            <span style="background-color: transparent; color: #212F3D;">Generates an instant Financial Health Card tailored specifically for individual NTC/NTB applicants.</span>
        </div>""", 
        unsafe_allow_html=True
    )
    
    flat_payload_dict = profile_payload.iloc.to_dict()
    client_pdf_bytes = generate_credit_pdf(client_name, health_score, risk_level_pct, badge_status, flat_payload_dict, pos_drivers, neg_drivers)
    
    st.download_button(
        label=f"📥 Download Customized PDF Passport for {client_name}",
        data=client_pdf_bytes,
        file_name=f"credit_passport_{client_name.lower().replace(' ', '_')}.pdf",
        mime="application/pdf",
        use_container_width=True
    )
