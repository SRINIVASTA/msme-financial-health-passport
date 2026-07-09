# 🏦 AI-Driven MSME Financial Health Passport 
### 🚀 Track 03: Financial Inclusion, Digital Lending, and Credit Decisioning

An end-to-end AI/ML credit underwriting engine built to onboard **New-to-Credit (NTC)** and **New-to-Bank (NTB)** small businesses. By replacing slow paper trails with live, consent-backed alternative footprints, this platform translates daily digital operations into an instant, bank-ready credit identity passport.

🌐 **Live Interactive Web App:** [Launch Live Streamlit Dashboard](https://msme-financial-health-passport-7smsi6mdtx3jcawyp8qnx9.streamlit.app/))

---

## 🛠️ Created By
* **Srinivasta**
* **Pujitha Sri**

---

## 💡 The Core Problem (Layman Terms)
Millions of small shop owners, local merchants, and micro-enterprises across India are locked out of the formal financial system. Traditional banks instantly reject up to 80% of them because they lack audited balance sheets, credit history bureau scores, or physical collateral. 

However, these same businesses process high transaction volumes daily through **UPI payments, GST invoicing networks, and employee records**. Because this rich alternative data remains unorganized, banks miss viable borrowers, slowing down financial inclusion.

## 🛡️ Our Solution
Our system creates a unified assessment framework by aggregating real-time, consent-backed alternative data feeds across 4 critical corporate pillars. An optimized **XGBoost Machine Learning Engine** processes these non-linear variables safely to calculate a comprehensive **Financial Health Score (300 to 900)** and an accurate default risk probability.

### 📊 The 4 Integrated Data Ingestion Pillars:
1. **Cash Flow Velocity (Account Aggregator API)**: Evaluates average daily liquid balance cushions, cash trends, and fund-insufficiency debit bounces.
2. **Sales Operational Stability (GSTN Invoicing Network)**: Analyzes monthly B2B turnover trends and calculates concentration risk (dependency on a single buyer).
3. **Everyday Sales Traction (UPI/QR Code Logs)**: Tracks retail transaction velocity counts and customer ticket sizes.
4. **Compliance Standing (EPFO Payroll Registers)**: Monitors staff headcount stability and salary fund payment punctuality to ensure business health.

---

## 🧬 Engine Framework Architecture & Explainable AI (XAI)
Lenders routinely reject machine learning models because they operate as untrusted "black boxes" that cannot pass banking regulatory audit boards. 

To bridge this regulatory gap, our engine embeds a **SHAP Explainability pipeline**. The app doesn't just show a raw score number; it automatically outputs a live, dynamic **Matplotlib horizontal bar chart** that breaks down the exact positive or negative directional weight of each business behavior. It then translates these complex mathematical weights into plain English insights and automated next-step suggestions that any small business owner or branch manager can instantly understand.

---

## 🌐 Enterprise Integration: Seamlessly Embedded
This application is built with high modularity in mind. By appending standard embed parameters (`?embed=true`), this entire AI scoring engine can be embedded via an HTML `<iframe>` tag directly into an enterprise bank's corporate portal or an MSME's digital accounting dashboard in under five minutes:

```html
<iframe
  src="https://streamlit.app?embed=true"
  style="height: 800px; width: 100%; border: none;"
  title="MSME Financial Health Passport">
</iframe>
```
Lenders do not need to overhaul their legacy software infrastructure; they can simply layer our Financial Health Passport into their existing onboarding workflows.

---

## ⚙️ Project Installation Setup & Execution Guide

### 1. Clone the project repository:
```bash
git clone https://github.com
cd msme-financial-health-passport
```

### 2. Install the verified machine learning and dashboard dependencies:
```bash
pip install -r requirements.txt
```

### 3. Launch the live application layout locally:
```bash
streamlit run app.py
```

---

## 📈 Real-World Ecosystem Rails Integration
This system is engineered as a lightweight, scalable microservice packaged to connect seamlessly with India's emerging public financial digital rails:
* **OCEN Protocol (Open Credit Network)**: Formats credit profiles as micro-JSON data tokens so Loan Service Providers (LSPs) can instantly broadcast payloads to multiple competing lenders.
* **Unified Lending Interface (ULI)**: Connects directly with central identity and bank data infrastructures to reduce corporate commercial appraisal timelines from weeks down to 60 seconds.

## ✒️ Author and Credits

* **Lead Architect & Developer:** [Srinivasta](https://github.com/SRINIVASTA) & My Team Mate:T.Pujitha Sri, BTECH (ECE), 2ND YEAR Student.

### Connect with Me
- [![LinkedIn](https://img.shields.io/badge/LinkedIn-0077B5?style=for-the-badge&logo=linkedin&logoColor=white)](https://www.linkedin.com/in/srinivas-t-a-557637119/)  
- [![Kaggle](https://img.shields.io/badge/Kaggle-20BEFF?style=for-the-badge&logo=kaggle&logoColor=white)](https://www.kaggle.com/srinivasta)  
- [![Email](https://img.shields.io/badge/Email-D14836?style=for-the-badge&logo=gmail&logoColor=white)](mailto:tasrinivass@gmail.com)  
- [![GitHub](https://img.shields.io/badge/GitHub-181717?style=for-the-badge&logo=github&logoColor=white)](https://github.com/srinivasta)

