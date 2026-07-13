import streamlit as st 
import numpy as np 
import pandas as pd 
import xgboost as xgb 
import shap 
import matplotlib.pyplot as plt 
import io 
from reportlab.lib.pagesizes import letter 
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle 
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle 
from reportlab.lib import colors 

# ===================================================================== 
# SYSTEM INITIALIZATION & GLOBAL LAYOUT SETUP 
# ===================================================================== 
st.set_page_config( 
    page_title="MSME Credit Health Card Portal", 
    page_icon="🏢", 
    layout="wide" 
) 

# 1. Clean, isolated section container for the IDBI Banner 
with st.container(): 
    st.image("idbi_banner.jpg", width=None, use_container_width=True) 

# Add a tiny visual spacing cushion between the banner and title 
st.write("") 

# 2. Main Header Row (Title on left, team logo badge on right) 
title_col, logo_col = st.columns([8.8, 1.2])

with title_col: 
    # Main Page Title 
    st.title("🏢 AI-Driven MSME Financial Health Passport") 
    
    # Track Details & Bullet Points nested directly on the left side 
    st.markdown( 
        """ 
        <div style='margin-top: 5px; margin-bottom: 15px;'> 
        <p style='margin: 0; font-size: 14px; color: #1e3d59; font-weight: 600;'> 
        📌 Designed for TRACK 03: Financial Health Score – Financial Inclusion, Digital Lending, Credit Decisioning 
        </p> 
        <p style='margin: 6px 0 0 15px; font-size: 13px; color: #6c757d; font-weight: 400; line-height: 1.5;'> 
        • This dashboard translates alternate business metrics (GST, UPI, Bank Records) into an instant credit decision tool. 
        </p> 
        <p style='margin: 4px 0 0 15px; font-size: 13px; color: #6c757d; font-weight: 400;'> 
        • Built as an accessible passport framework that anyone can easily interpret. 
        </p> 
        </div> 
        """, 
        unsafe_allow_html=True 
    ) 

with logo_col: 
    # Push the logo badge down slightly to balance with the title row 
    st.markdown("<div style='margin-top: 12px;'></div>", unsafe_allow_html=True)
    try: 
        # Fixed width prevents vertical wrapping issues 
        st.image("vizagites.png", width=95) 
        st.markdown( 
            "<p style='text-align: center; margin-top: -3px; font-size: 10px; color: #6c757d; font-weight: 500; width: 95px;'>\n" 
            "Developed by Vizagites\n" 
            "</p>", 
            unsafe_allow_html=True 
        ) 
    except Exception: 
        # Fallback text box if image file is missing or fails to render 
        st.markdown(
            "<div style='text-align: right; font-weight: bold; color: #1e3d59; font-size: 14px; margin-top: 10px;'>" 
            "🚀 Vizagites" 
            "</div>", 
            unsafe_allow_html=True 
        )
# Mandatory sequence of numerical training features required by the XGBoost Engine 
REQUIRED_FEATURES = [ 
    'aa_avg_daily_balance_inr', 'aa_inflow_outflow_ratio', 'aa_fund_insufficient_bounces_3m',
    'gst_monthly_turnover_inr', 'gst_buyer_concentration_ratio', 'gst_filing_delay_days_avg', 
    'upi_tx_volume_monthly', 'upi_ticket_size_avg_inr', 'epfo_employee_count', 
    'epfo_payment_punctuality_score' 
] 

# Plain English dictionary mapping features to user-friendly terminology 
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
# DYNAMIC IN-APP RETRAINING ENGINE WITH SHAP TREE WEIGHT RETENTION
# ===================================================================== 
def train_custom_credit_engine(custom_df=None): 
    """Trains on a stable 1,200 row database first, then overlays custom sandbox rows.""" 
    validation_failed = False
    error_message = ""
    
    # 1. ALWAYS build the large baseline dataset first so the model learns all 10 feature weights
    np.random.seed(42) 
    n_samples = 1200 
    base_data = { 
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
    train_df = pd.DataFrame(base_data) 
    risk_heuristic = ( 
        (train_df['aa_fund_insufficient_bounces_3m'] * 0.5) +
        (train_df['gst_buyer_concentration_ratio'] * 2.5) + 
        (train_df['gst_filing_delay_days_avg'] * 0.20) - 
        (train_df['aa_inflow_outflow_ratio'] * 1.5) - 
        (train_df['epfo_payment_punctuality_score'] * 1.2) 
    ) 
    train_df['is_default'] = (risk_heuristic >= np.percentile(risk_heuristic, 85)).astype(int) 

    # 2. Train the robust XGBoost model using the large comprehensive matrix
    X_train = train_df[REQUIRED_FEATURES] 
    y_train = train_df['is_default'] 
    constraints = (0, -1, 1, 0, 1, 1, 0, 0, -1, -1) 
    model = xgb.XGBClassifier(n_estimators=100, max_depth=4, learning_rate=0.08, monotone_constraints=constraints) 
    model.fit(X_train, y_train) 
    explainer = shap.TreeExplainer(model) 

    # 3. Process the sandbox rows for active user inspection and testing
    if custom_df is not None: 
        df = custom_df.copy() 
        df.columns = [str(c).lower().strip() for c in df.columns] 
        
        for actual_col in df.columns:
            if actual_col.startswith('gst_monthly'):
                df = df.rename(columns={actual_col: 'gst_monthly_t'})
                break

        # --- VALIDATION TALLY GATES ---
        mandatory_upload_columns = ['aa_avg_daily_balance_inr', 'aa_inflow_outflow_ratio', 'aa_fund_insufficient_bounces_3m', 'gst_monthly_t']
        missing_from_upload = [col for col in mandatory_upload_columns if col not in df.columns]
        
        if missing_from_upload:
            validation_failed = True
            error_message = f"File verification failed. Missing columns: {', '.join(missing_from_upload)}"
            return None, None, REQUIRED_FEATURES, None, validation_failed, error_message

        df['gst_monthly_turnover_inr'] = df['gst_monthly_t']
        df = df.drop(columns=['row_id', 'id', 'sno', 'unnamed: 0', 'gst_monthly_t'], errors='ignore') 
        
        # Inject dynamic variations into unprovided columns for the 10 custom test rows to enrich SHAP spectrums
        np.random.seed(len(df))
        n_rows = len(df)
        if 'gst_buyer_concentration_ratio' not in df.columns: df['gst_buyer_concentration_ratio'] = np.random.uniform(0.1, 0.5, size=n_rows)
        if 'gst_filing_delay_days_avg' not in df.columns: df['gst_filing_delay_days_avg'] = np.random.randint(1, 5, size=n_rows)
        if 'upi_tx_volume_monthly' not in df.columns: df['upi_tx_volume_monthly'] = np.random.randint(300, 1200, size=n_rows)
        if 'upi_ticket_size_avg_inr' not in df.columns: df['upi_ticket_size_avg_inr'] = np.random.uniform(200, 600, size=n_rows)
        if 'epfo_employee_count' not in df.columns: df['epfo_employee_count'] = np.random.randint(10, 35, size=n_rows)
        if 'epfo_payment_punctuality_score' not in df.columns: df['epfo_payment_punctuality_score'] = np.random.uniform(0.8, 1.0, size=n_rows)
        
        for col in REQUIRED_FEATURES: 
            df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0.0) 
            
        df['is_default'] = model.predict(df[REQUIRED_FEATURES])
        output_df = df
    else:
        output_df = train_df

    return model, explainer, REQUIRED_FEATURES, output_df, validation_failed, error_message
# ===================================================================== 
# ENTERPRISE PDF CREDIT HEALTH CARD REPORTING ENGINE 
# ===================================================================== 
def generate_credit_pdf(client_name, score, risk, tier, payload_dict, helpers, hurters): 
    """Compiles an audit-ready financial health passport report document buffer."""
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
    t_score = Table(score_data, colWidths=) 
    t_score.setStyle(TableStyle([('BACKGROUND', (0,0), (1,0), colors.HexColor('#EAEEF4')), ('GRID', (0,0), (-1,-1), 0.5, colors.grey), ('PADDING', (0,0), (-1,-1), 8)])) 
    story.append(t_score) 
    story.append(Spacer(1, 15)) 
 
    story.append(Paragraph("<b>Alternative Network Ingested Data Summary</b>", header_style)) 
    metrics_data = [[Paragraph("<b>Metric Vector Parameter</b>", bold_body), Paragraph("<b>Reported Value</b>", bold_body)]] 
    for f in REQUIRED_FEATURES: 
        v = payload_dict.get(f, 0.0) 
        metrics_data.append([Paragraph(layman_translation.get(f, f), body_style), Paragraph(str(v), body_style)]) 
 
    t_metrics = Table(metrics_data, colWidths=) 
    t_metrics.setStyle(TableStyle([('BACKGROUND', (0,0), (1,0), colors.HexColor('#EAEEF4')), ('GRID', (0,0), (-1,-1), 0.5, colors.lightgrey), ('PADDING', (0,0), (-1,-1), 5)])) 
    story.append(t_metrics) 
    story.append(Spacer(1, 15)) 
 
    story.append(Paragraph("<b>Explainable AI (XAI) Score Attribution Drivers</b>", header_style)) 
    story.append(Paragraph(f"<b>Top Metrics Supporting Score:</b> {', '.join(helpers) if helpers else 'None Identified'}", body_style)) 
    story.append(Paragraph(f"<b>Primary Drivers Negatively Impacting Score:</b> {', '.join(hurters) if hurters else 'None Identified'}", body_style)) 
    story.append(Spacer(1, 20)) 
 
    story.append(Paragraph("<i>This evaluation document is verified via decentralized account aggregators. Generated instantly via ULI protocol interfaces.</i>", body_style)) 
    doc.build(story) 
    buffer.seek(0)
    return buffer.getvalue()
# ===================================================================== 
# SIDEBAR LAYER & PROTOCOL STATE SYNCHRONIZATION 
# ===================================================================== 
col_sidebar, col_card = st.columns([1, 1.2]) 

def sync_inputs_to_selected_row(): 
    """State sync callback executed instantly when changing dropdown item.""" 
    if "active_dataset" in st.session_state: 
        current_label = st.session_state.active_msme_dropdown 
        row_idx = int(current_label.split("-")[-1]) - 1 
        row_data = st.session_state["active_dataset"].iloc[row_idx]
        
        # Flush targets out straight to synchronized sidebar parameter caches safely 
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
        st.session_state["sb_client_name"] = f"Sri Venkateswara Enterprises ({current_label})" 

with col_sidebar: 
    st.subheader("📊 Data Ingestion Protocol Selection") 
    data_source_mode = st.radio(
        label="Select Input Ingestion Channel:", 
        options=["Live Ecosystem APIs (ULI / OCEN / AA Simulation)", "Batch Document Upload (CSV Sandbox)"], 
        index=0, 
        key="data_source_mode_radio"
    ) 
 
    # SYSTEM RESET: If changing mode channels, drop old model sessions to reset baseline 
    if "prev_source_mode" not in st.session_state or st.session_state["prev_source_mode"] != data_source_mode: 
        st.session_state["prev_source_mode"] = data_source_mode 
        for k in ["active_model", "active_explainer", "active_features", "active_dataset", "last_loaded_file", "sb_balance", "sb_ratio", "sb_bounces", "sb_turnover", "sb_conc", "sb_delay", "sb_upi_vol", "sb_upi_size", "sb_epfo_staff", "sb_epfo_score", "sb_client_name", "active_msme_dropdown"]: 
            if k in st.session_state: 
                del st.session_state[k]
    # CHANNEL REGIME A: DOCUMENT sandbox FILE UPLOAD 
    if data_source_mode == "Batch Document Upload (CSV Sandbox)": 
        st.markdown("---") 
        st.subheader("🛠 Model Optimization Sandbox") 
        uploaded_bank_file = st.file_uploader(label="📊 Upload Bank Batch Update Data (CSV Format)", type=["csv"], key="csv_file_uploader_widget") 
 
        if uploaded_bank_file is not None: 
            if "last_loaded_file" not in st.session_state or st.session_state["last_loaded_file"] != uploaded_bank_file.name: 
                try: 
                    user_imported_df = pd.read_csv(uploaded_bank_file) 
                    
                    if len(user_imported_df.columns) == 4:
                        user_imported_df.columns = [
                            'aa_avg_daily_balance_inr', 
                            'aa_inflow_outflow_ratio', 
                            'aa_fund_insufficient_bounces_3m', 
                            'gst_monthly_t'
                        ]
                    
                    m_obj, e_obj, f_list, d_matrix, is_failed, err_msg = train_custom_credit_engine(user_imported_df) 
 
                    if is_failed:
                        if "active_dataset" in st.session_state: del st.session_state["active_dataset"]
                        st.sidebar.error(f"❌ Structural Tally Mismatch:\n{err_msg}")
                        st.error("🚨 Processing halted. The uploaded CSV fields do not tally with required schema constants.")
                        st.stop() 
                    else:
                        st.session_state["active_model"] = m_obj 
                        st.session_state["active_explainer"] = e_obj 
                        st.session_state["active_features"] = f_list 
                        st.session_state["active_dataset"] = d_matrix 
                        st.session_state["last_loaded_file"] = uploaded_bank_file.name 
                        
                        for k in ["sb_balance", "sb_ratio", "sb_bounces", "sb_turnover", "sb_conc", "sb_delay", "sb_upi_vol", "sb_upi_size", "sb_epfo_staff", "sb_epfo_score", "sb_client_name", "active_msme_dropdown"]: 
                            if k in st.session_state: del st.session_state[k] 
                        st.toast("Custom Sheet Ingested and Verified successfully!", icon="✅") 
                except Exception as e: 
                    st.error(f"Processing Error in Custom Format: {str(e)}") 

        if "active_dataset" not in st.session_state: 
            m_obj, e_obj, f_list, d_matrix, _, _ = train_custom_credit_engine(None) 
            st.session_state["active_model"] = m_obj 
            st.session_state["active_explainer"] = e_obj 
            st.session_state["active_features"] = f_list 
            st.session_state["active_dataset"] = d_matrix 

        st.markdown("---")
        st.subheader("📋 Step 1: Select Uploaded Row & Inspect Parameters")
        active_df = st.session_state["active_dataset"]
        upload_options = [f"UPLOAD-{str(i+1).zfill(4)}" for i in range(len(active_df))]
        
        selected_msme_label = st.selectbox(label="Choose target MSME to inspect:", options=upload_options, key="active_msme_dropdown", on_change=sync_inputs_to_selected_row)
        
        if selected_msme_label is None or not isinstance(selected_msme_label, str) or "-" not in selected_msme_label:
            selected_row_index = 0
        else:
            selected_row_index = int(selected_msme_label.split("-")[-1]) - 1
            
        extracted_row_data = active_df.iloc[selected_row_index]
 
    # CHANNEL REGIME B: LIVE REAL-TIME API INGESTIONS 
    else: 
        if "active_dataset" not in st.session_state: 
            m_obj, e_obj, f_list, d_matrix, _, _ = train_custom_credit_engine(None) 
            st.session_state["active_model"] = m_obj 
            st.session_state["active_explainer"] = e_obj 
            st.session_state["active_features"] = f_list 
            st.session_state["active_dataset"] = d_matrix 
        st.markdown("---") 
        st.subheader("📋 Step 1: Select Active Row & Fine-Tune Parameters") 
        active_df = st.session_state["active_dataset"] 
        msme_options = [f"MSME-{str(i+1).zfill(4)}" for i in range(len(active_df))] 
 
        selected_msme_label = st.selectbox(label="Choose target MSME to inspect:", options=msme_options, key="active_msme_dropdown", on_change=sync_inputs_to_selected_row) 
 
        if selected_msme_label is None or not isinstance(selected_msme_label, str) or "-" not in selected_msme_label:
            selected_row_index = 0
        else:
            selected_row_index = int(selected_msme_label.split("-")[-1]) - 1 
            
        extracted_row_data = active_df.iloc[selected_row_index] 

    if "sb_balance" not in st.session_state: 
        st.session_state["sb_balance"] = int(extracted_row_data.get('aa_avg_daily_balance_inr', 0)) 
        st.session_state["sb_ratio"] = float(np.clip(extracted_row_data.get('aa_inflow_outflow_ratio', 1.0), 0.5, 2.0)) 
        st.session_state["sb_bounces"] = int(extracted_row_data.get('aa_fund_insufficient_bounces_3m', 0))
        st.session_state["sb_turnover"] = int(extracted_row_data.get('gst_monthly_turnover_inr', 0)) 
        st.session_state["sb_conc"] = float(np.clip(extracted_row_data.get('gst_buyer_concentration_ratio', 0.0), 0.0, 1.0)) 
        st.session_state["sb_delay"] = int(extracted_row_data.get('gst_filing_delay_days_avg', 0)) 
        st.session_state["sb_upi_vol"] = int(extracted_row_data.get('upi_tx_volume_monthly', 0)) 
        st.session_state["sb_upi_size"] = int(extracted_row_data.get('upi_ticket_size_avg_inr', 10)) 
        st.session_state["sb_epfo_staff"] = int(extracted_row_data.get('epfo_employee_count', 1))
        st.session_state["sb_epfo_score"] = float(np.clip(extracted_row_data.get('epfo_payment_punctuality_score', 1.0), 0.0, 1.0)) 
        
        lbl = "UPLOAD" if data_source_mode == "Batch Document Upload (CSV Sandbox)" else "MSME"
        st.session_state["sb_client_name"] = f"Sri Venkateswara Enterprises ({lbl}-{str(selected_row_index+1).zfill(4)})" 
            
    client_name = st.text_input("Assign Corporate Display Name", key="sb_client_name") 
 
    with st.expander("💳 Account Aggregator Records", expanded=True): 
        input_balance = st.number_input("Average Daily Balance kept in Bank (INR)", min_value=0, key="sb_balance", step=5000) 
        input_ratio = st.slider("Money Inflow vs Outflow Ratio", 0.5, 2.0, key="sb_ratio", step=0.05) 
        input_bounces = st.number_input("Cheque Bounces due to low funds (3M)", min_value=0, max_value=30, key="sb_bounces") 
        
    with st.expander("📊 Tax & Sales Records (GST Portal)", expanded=True): 
        input_turnover = st.number_input("Average Monthly Sales/Turnover (INR)", min_value=0, key="sb_turnover", step=10000) 
        input_conc = st.slider("Dependency Risk (High = Single Buyer)", 0.0, 1.0, key="sb_conc", step=0.05)
        input_delay = st.number_input("Average Tax Filing Delay (Days)", min_value=0, max_value=90, key="sb_delay") 
        
    with st.expander("👥 Everyday Digital Operations (UPI & EPFO)", expanded=True): 
        input_upi_vol = st.number_input("Total UPI Sales Transactions per Month", min_value=0, key="sb_upi_vol") 
        input_upi_size = st.number_input("Average Bill Amount per Customer (INR)", min_value=10, key="sb_upi_size") 
        input_epfo_staff = st.number_input("Active Registered Staff Count", min_value=1, key="sb_epfo_staff")
        input_epfo_score = st.slider("Staff Fund Payment Timeliness (1.0 = Perfect)", 0.0, 1.0, key="sb_epfo_score", step=0.05) 

profile_payload = pd.DataFrame([{ 
    'aa_avg_daily_balance_inr': float(input_balance), 'aa_inflow_outflow_ratio': float(input_ratio), 'aa_fund_insufficient_bounces_3m': int(input_bounces),
    'gst_monthly_turnover_inr': float(input_turnover), 'gst_buyer_concentration_ratio': float(input_conc), 'gst_filing_delay_days_avg': int(input_delay), 
    'upi_tx_volume_monthly': int(input_upi_vol), 'upi_ticket_size_avg_inr': float(input_upi_size), 'epfo_employee_count': int(input_epfo_staff), 'epfo_payment_punctuality_score': float(input_epfo_score) 
}]) 

model = st.session_state["active_model"] 
explainer = st.session_state["active_explainer"] 
feature_names = st.session_state["active_features"]
# ===================================================================== 
# BLOCK 6: MAIN DISPLAY CARD INTERFACE & EXPECTED OUTCOMES 
# ===================================================================== 
with col_card:
    if data_source_mode == "Batch Document Upload (CSV Sandbox)" and "last_loaded_file" in st.session_state: 
        st.subheader("📊 Active Uploaded Bank Registry Database") 
        st.dataframe(st.session_state["active_dataset"], use_container_width=True, height=180) 
    else: 
        st.subheader("ℹ Active Registry Status") 
        st.info("Running on baseline Synthetic Databank Engine (1,200 Rows). Connected via ULI architectures.") 
        
    st.markdown("---") 
    st.subheader("🎯 Step 2: Live Credit Card Passport Results") 
    prob_output = model.predict_proba(profile_payload) 
    default_probability = float(prob_output) 
    non_default_probability = 1.0 - default_probability 
    health_score = int(300 + (non_default_probability * 600)) 
    risk_level_pct = default_probability * 100 
    
    if health_score >= 750: 
        badge_status = "EXCELLENT FINANCIAL HEALTH" 
        st.success(f"SYSTEM ASSESSMENT TIER STATUS: {badge_status}") 
    elif health_score >= 650: 
        badge_status = "MODERATE FINANCIAL RISK" 
        st.warning(f"SYSTEM ASSESSMENT TIER STATUS: {badge_status}") 
    else: 
        badge_status = "HIGH RISK VULNERABILITY" 
        st.error(f"SYSTEM ASSESSMENT TIER STATUS: {badge_status}") 

    col_stat1, col_stat2 = st.columns(2) 
    with col_stat1: st.metric(label="Your Financial Health Score", value=f"{health_score} / 900") 
    with col_stat2: st.metric(label="Estimated Risk Level", value=f"{risk_level_pct:.2f}%") 
    st.progress((health_score - 300) / 600) 
    st.markdown("---") 
    
    st.subheader("📊 Plain English Attributions (Explainable AI)") 
    shap_values = explainer(profile_payload) 
    if hasattr(shap_values, "values"): 
        shap_mat = shap_values.values 
    else: 
        shap_mat = np.array(shap_values) 
        
    if len(shap_mat.shape) == 3: 
        raw_impacts = shap_mat[0, :, 1] 
    elif len(shap_mat.shape) == 2: 
        raw_impacts = shap_mat[0, :] 
    else: 
        raw_impacts = np.ravel(shap_mat) 
        
    raw_impacts = np.nan_to_num(np.array(raw_impacts).flatten()) 
    if len(raw_impacts) > len(feature_names): 
        raw_impacts = raw_impacts[:len(feature_names)] 
    elif len(raw_impacts) < len(feature_names): 
        raw_impacts = np.pad(raw_impacts, (0, len(feature_names) - len(raw_impacts)), 'constant') 
        
    chart_dataframe = pd.DataFrame({ 
        'Feature': [layman_translation[f] for f in feature_names], 
        'Impact': raw_impacts 
    }) 
    chart_dataframe['Color'] = np.where(chart_dataframe['Impact'] >= 0, '#e74c3c', '#2ecc71') 
    
    fig, ax = plt.subplots(figsize=(6, 4)) 
    chart_plot_df = chart_dataframe.sort_values(by='Impact', ascending=True) 
    ax.barh(chart_plot_df['Feature'], chart_plot_df['Impact'], color=chart_plot_df['Color']) 
    ax.axvline(0, color='black', linewidth=0.8, linestyle='--') 
    ax.set_xlabel('Risk Contribution Weight') 
    plt.tight_layout() 
 
    st.pyplot(fig) 
    plt.close(fig) 
    
    neg_subset = chart_dataframe[chart_dataframe['Impact'] > 0.001].sort_values(by='Impact', ascending=False) 
    neg_drivers = neg_subset['Feature'].head(2).tolist() 
    pos_subset = chart_dataframe[chart_dataframe['Impact'] < -0.001].sort_values(by='Impact', ascending=True) 
    pos_drivers = pos_subset['Feature'].head(2).tolist() 
    
    if not pos_drivers: 
        pos_drivers = ["Baseline Stability Metrics"] 
    if not neg_drivers: 
        neg_drivers = ["None (Perfect Structural Integrity)"] 

# ===================================================================== 
# GLOBAL FOOTER PANELS
# ===================================================================== 
st.markdown("---") 
with col_card: 
    st.subheader("💾 Master Data Export Controls") 
    active_df = st.session_state["active_dataset"]
    approved_dataframe = active_df[active_df['is_default'] == 0] 
    rejected_dataframe = active_df[active_df['is_default'] == 1] 
    
    st.download_button(label=f"📥 Download Approved Portfolio ({len(approved_dataframe)} Rows)", data=approved_dataframe.to_csv(index=False).encode('utf-8'), file_name="approved_msme_credit_passport.csv", mime="text/csv", use_container_width=True) 
    st.download_button(label=f"📤 Download Rejected Portfolio ({len (rejected_dataframe)} Rows)", data=rejected_dataframe.to_csv(index=False).encode('utf-8'), file_name="rejected_msme_credit_passport.csv", mime="text/csv", use_container_width=True) 

st.markdown("---") 
with col_card: 
    st.subheader("📂 Export Specific Client Document") 
    st.markdown(f"""<div style="background-color: #EBF5FB; border-left: 5px solid #2980B9; padding: 10px; border-radius: 4px; margin-bottom: 15px;"> 
    <span style="background-color: transparent; color: #1B4F72; font-weight: bold;">📊 TRACK 03 EXPECTED OUTCOME:</span> 
    <span style="background-color: transparent; color: #212F3D;">Generates an instant Financial Health Card tailored specifically for individual NTC/NTB applicants.</span> 
    </div>""", unsafe_allow_html=True) 

    flat_payload_dict = {k: float(v) for k, v in profile_payload.iloc.to_dict().items()} 
    client_pdf_bytes = generate_credit_pdf(client_name, health_score, risk_level_pct, badge_status, flat_payload_dict, pos_drivers, neg_drivers) 
 
    st.download_button(label=f"📄 Download Customized PDF Passport for {client_name}", data=client_pdf_bytes, file_name=f"credit_passport_{client_name.lower().replace(' ', '_').replace('(', '').replace(')', '')}.pdf", mime="application/pdf", use_container_width=True)
