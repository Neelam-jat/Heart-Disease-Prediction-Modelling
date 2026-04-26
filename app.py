import streamlit as st
import pandas as pd
import pickle
import numpy as np

# --- PAGE CONFIGURATION ---
st.set_page_config(page_title="CardioRisk AI Predictor", page_icon="🩺", layout="wide")

# --- HEADER ---
st.title("🩺 CardioRisk AI Predictor")
st.markdown("**Clinical Heart Disease Risk Assessment Tool** *(Powered by UCI Cleveland Dataset)*")
st.divider()

# --- LOAD MODEL & SCALER ---
@st.cache_resource
def load_artifacts():
    try:
        with open('heart_disease_rf_model.pkl', 'rb') as f:
            return pickle.load(f)
    except FileNotFoundError:
        st.error("❌ Model not found! Please run your ML pipeline first to generate 'heart_disease_rf_model.pkl'.")
        st.stop()

artifacts = load_artifacts()
model = artifacts['model']
scaler = artifacts['scaler']

# --- CLINICIAN-FRIENDLY INPUTS (Sidebar) ---
st.sidebar.header("📋 Patient Information")

# 1. Demographics
st.sidebar.subheader("Demographics")
col1, col2 = st.sidebar.columns(2)
with col1:
    age = st.number_input("Age", min_value=18, max_value=120, value=55)
with col2:
    sex_options = {"Female": 0, "Male": 1}
    sex_label = st.selectbox("Biological Sex", list(sex_options.keys()))
    sex = sex_options[sex_label]

# 2. Vitals & Bloodwork
st.sidebar.subheader("Vitals & Bloodwork")
trestbps = st.sidebar.number_input("Resting BP (mmHg)", min_value=80, max_value=220, value=120)
chol = st.sidebar.number_input("Serum Cholesterol (mg/dl)", min_value=100, max_value=600, value=200)

fbs_options = {"<= 120 mg/dl (Normal)": 0, "> 120 mg/dl (Elevated)": 1}
fbs_label = st.sidebar.selectbox("Fasting Blood Sugar", list(fbs_options.keys()))
fbs = fbs_options[fbs_label]

# 3. Cardiac Evaluation
st.sidebar.subheader("Cardiac Evaluation")
cp_options = {
    "Typical Angina": 0,
    "Atypical Angina": 1,
    "Non-anginal Pain": 2,
    "Asymptomatic": 3
}
cp_label = st.sidebar.selectbox("Chest Pain Type", list(cp_options.keys()))
cp = cp_options[cp_label]

restecg_options = {
    "Normal": 0, 
    "ST-T Wave Abnormality": 1, 
    "Left Ventricular Hypertrophy": 2
}
restecg_label = st.sidebar.selectbox("Resting ECG Results", list(restecg_options.keys()))
restecg = restecg_options[restecg_label]

thalach = st.sidebar.number_input("Max Heart Rate Achieved", min_value=60, max_value=220, value=150)

exang_options = {"No": 0, "Yes": 1}
exang_label = st.sidebar.selectbox("Exercise Induced Angina", list(exang_options.keys()))
exang = exang_options[exang_label]

oldpeak = st.sidebar.number_input("ST Depression (Exercise vs Rest)", min_value=0.0, max_value=10.0, value=1.0, step=0.1)

slope_options = {"Upsloping": 0, "Flat": 1, "Downsloping": 2}
slope_label = st.sidebar.selectbox("ST Segment Slope", list(slope_options.keys()))
slope = slope_options[slope_label]

ca = st.sidebar.selectbox("Major Vessels Colored by Fluoroscopy (0-4)", [0, 1, 2, 3, 4])

thal_options = {
    "Unknown": 0, 
    "Normal": 1, 
    "Fixed Defect": 2, 
    "Reversible Defect": 3
}
thal_label = st.sidebar.selectbox("Thalassemia", list(thal_options.keys()))
thal = thal_options[thal_label]

# --- PREDICTION LOGIC ---
if st.button("🔬 **Generate Risk Assessment**", type="primary", use_container_width=True):
    
    # 1. Map inputs directly to a DataFrame with EXACT pipeline column names
    input_data = pd.DataFrame({
        'age': [age],
        'sex': [sex],
        'cp': [cp],
        'trestbps': [trestbps],
        'chol': [chol],
        'fbs': [fbs],
        'restecg': [restecg],
        'thalach': [thalach],
        'exang': [exang],
        'oldpeak': [oldpeak],
        'slope': [slope],
        'ca': [ca],
        'thal': [thal]
    })
    
    # 2. Scale features exactly as done in training
    scaled_features = scaler.transform(input_data)
    
    # 3. Predict Probability and Class
    prob = model.predict_proba(scaled_features)[0][1]
    prediction = model.predict(scaled_features)[0]
    
    # --- RESULTS DASHBOARD ---
    st.subheader("📊 Diagnostic Risk Report")
    
    # Patient Metrics Summary
    col_m1, col_m2, col_m3, col_m4 = st.columns(4)
    col_m1.metric("Age / Sex", f"{age} / {sex_label[0]}")
    col_m2.metric("Blood Pressure", f"{trestbps} mmHg")
    col_m3.metric("Cholesterol", f"{chol} mg/dl")
    col_m4.metric("Max Heart Rate", f"{thalach} bpm")
    
    st.divider()
    
    # Risk Analysis Box
    if prediction == 1:
        st.error("🚨 **HIGH RISK DETECTED**")
        st.markdown(f"**Cardiovascular Disease Probability: {prob * 100:.1f}%**")
        st.progress(float(prob))
        st.warning("⚠️ **Clinical Action Required:** Immediate consultation with a cardiologist is recommended for further diagnostic testing (e.g., stress echocardiography, angiography).")
    else:
        st.success("✅ **LOW RISK DETECTED**")
        st.markdown(f"**Cardiovascular Disease Probability: {prob * 100:.1f}%**")
        st.progress(float(prob))
        st.info("💡 **Clinical Recommendation:** Routine monitoring. Advise the patient to maintain a heart-healthy diet and regular cardiovascular exercise.")
    
    # Contributing Risk Factors Analysis
    st.subheader("⚠️ Potential Contributing Factors")
    risks = []
    if age >= 60: risks.append("👴 **Age:** Advanced age is a baseline risk factor.")
    if chol > 240: risks.append("🍔 **Cholesterol:** Elevated serum cholesterol (>240 mg/dl).")
    if trestbps > 140: risks.append("📈 **Blood Pressure:** Elevated resting blood pressure (Hypertension).")
    if thalach < 120: risks.append("💓 **Heart Rate:** Lower maximum heart rate achieved.")
    if oldpeak >= 2.0: risks.append("📉 **ECG:** Significant ST depression observed.")
    if exang == 1: risks.append("🏃 **Angina:** Exercise-induced angina present.")
    
    if risks:
        for risk in risks:
            st.write(risk)
    else:
        st.write("✅ No primary physiological red flags detected in current vitals.")

# --- FOOTER ---
st.divider()
st.caption("Powered by Random Forest | Trained on UCI Cleveland Dataset | **Note:** This tool is for educational and clinical decision support purposes only and does not replace professional medical diagnosis.")
