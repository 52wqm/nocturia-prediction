"""
Nocturia Risk Prediction App — CatBoost 16-feature model
LBXHA (Hepatitis A antibody) excluded. Based on NHANES 2005-2023.
Prediction path: raw inputs -> (raw - mean) / std -> CatBoost -> probability
"""
import streamlit as st
import numpy as np
import joblib
import os

st.set_page_config(page_title="Nocturia Risk Prediction", page_icon="🌙", layout="wide")
st.title("🌙 Nocturia Risk Prediction")
st.caption("CatBoost 16-feature model (AUC=0.7428), training data: NHANES 2005-2023 (n=39,494)")

# ---- Load model ----
@st.cache_resource
def load_model():
    path = os.path.join(os.path.dirname(__file__), "models", "models_16feat.pkl")
    data = joblib.load(path)
    return data["model"], data["features"], data["z_params"], data["threshold"], data["auc"]

model, features, z_params, THRESHOLD, MODEL_AUC = load_model()

# ---- Label mappings ----
FEAT_CN = {
    "RIDAGEYR": "Age (years)",
    "INDFMPIR": "Income/Poverty Ratio",
    "CKD": "Chronic Kidney Disease",
    "LBXSAPSI": "Alkaline Phosphatase (U/L)",
    "LBXSCLSI": "Chloride (mmol/L)",
    "LBXSGB": "Globulin (g/dL)",
    "LBXSOSSI": "Serum Osmolality (mOsm/kg)",
    "BMXWAIST": "Waist Circumference (cm)",
    "BPQ020": "History of Hypertension",
    "BP_DIA_MEAN": "Diastolic Blood Pressure (mmHg)",
    "LBXRBCSI": "Red Blood Cell Count (x10^6/uL)",
    "LBXRDW": "Red Cell Distribution Width (%)",
    "DPQ_TOTAL": "PHQ-9 Depression Score",
    "MCQ220": "History of Cancer",
    "Race_NH_Black": "Non-Hispanic Black",
    "Edu_Lt9th": "Education < 9th Grade",
}

# ---- Sidebar ----
st.sidebar.header("About This Tool")
st.sidebar.markdown(f"""
This tool predicts nocturia risk (≥2 voids/night) using **16 clinical and demographic features**.

### Feature Selection
1. **Boruta** feature selection (perc=100)
2. **1000x LASSO stability selection** (C=0.0336, ≥95%)
3. Excluded 3 leakage/confounding + 1 SI duplicate -> 17 features
4. Excluded **LBXHA (Hepatitis A antibody)** — no biological link to nocturia -> **16 features**

### Model Info
- **Algorithm**: CatBoost (n=200, depth=6, lr=0.05)
- **Test AUC**: {MODEL_AUC}
- **Threshold**: {THRESHOLD} (F1-optimal)
- **Train/Test Split**: 80/20, stratified, seed=42

### Data Source
NHANES 2005-2023, n=39,494 (nocturia 13,097 cases, 33.2%)
""")

# ========================
# Input Form
# ========================
col1, col2, col3 = st.columns(3)

with col1:
    st.subheader("📋 Demographics")
    age = st.number_input("Age (years)", min_value=20, max_value=85, value=50, step=1)
    income_poverty = st.number_input(
        "Income/Poverty Ratio", min_value=0.0, max_value=5.0, value=2.3, step=0.1,
        help="INDFMPIR: household income / federal poverty line, 0-5")
    race_black = st.checkbox("Non-Hispanic Black", value=False)
    edu_lt9 = st.checkbox("Education < 9th Grade", value=False)

with col2:
    st.subheader("🩺 Medical History")
    hypertension = st.checkbox("History of Hypertension", value=False)
    ckd = st.checkbox("Chronic Kidney Disease (CKD)", value=False)
    cancer = st.checkbox("History of Cancer/Malignancy", value=False,
                         help="MCQ220: ever told by a doctor you have cancer or malignancy")

    st.subheader("📏 Physical Exam")
    waist = st.number_input("Waist Circumference (cm)", min_value=55.0, max_value=188.0, value=98.0, step=0.5)
    dia_bp = st.number_input("Diastolic BP (mmHg)", min_value=40.0, max_value=140.0, value=72.0, step=0.5)

with col3:
    st.subheader("🔬 Lab Tests")
    rbc = st.number_input("RBC Count (x10^6/uL)", min_value=1.5, max_value=8.5, value=4.7, step=0.1)
    rdw = st.number_input("RDW (%)", min_value=10.0, max_value=40.0, value=13.2, step=0.1)
    globulin = st.number_input("Globulin (g/dL)", min_value=0.5, max_value=7.5, value=2.9, step=0.1)
    alp = st.number_input("Alkaline Phosphatase (U/L)", min_value=5, max_value=750, value=69, step=1)
    chloride = st.number_input("Chloride (mmol/L)", min_value=70.0, max_value=135.0, value=103.0, step=0.5)
    osmolality = st.number_input("Serum Osmolality (mOsm/kg)", min_value=200.0, max_value=335.0, value=279.0, step=0.5)

    st.subheader("🧠 Mental Health")
    phq9 = st.number_input("PHQ-9 Total Score", min_value=0, max_value=27, value=2, step=1,
                           help="Patient Health Questionnaire-9, range 0-27, >=10 suggests moderate-severe depression")

# ---- Map inputs to raw values ----
raw_values = {
    "RIDAGEYR": age, "INDFMPIR": income_poverty,
    "CKD": 1.0 if ckd else 0.0,
    "LBXSAPSI": alp, "LBXSCLSI": chloride, "LBXSGB": globulin,
    "LBXSOSSI": osmolality, "BMXWAIST": waist,
    "BPQ020": 1.0 if hypertension else 0.0,
    "BP_DIA_MEAN": dia_bp,
    "LBXRBCSI": rbc, "LBXRDW": rdw,
    "DPQ_TOTAL": phq9,
    "MCQ220": 1.0 if cancer else 0.0,
    "Race_NH_Black": 1.0 if race_black else 0.0,
    "Edu_Lt9th": 1.0 if edu_lt9 else 0.0,
}

# ---- Single-step standardization: raw -> (raw - mean) / std ----
X = np.array([[
    (raw_values[f] - z_params[f]["mean"]) / z_params[f]["std"]
    for f in features
]])
prob = model.predict_proba(X)[0, 1]
pred = 1 if prob >= THRESHOLD else 0

# ---- Results ----
st.markdown("---")
st.header("Prediction Result")

risk_col, gauge_col = st.columns([1, 2])

with risk_col:
    st.metric("Predicted Probability", f"{prob:.1%}")

    if pred == 1:
        st.error("⚠️ **High Risk** — Predicted nocturia positive")
    else:
        st.success("✅ **Low Risk** — Predicted nocturia negative")

    st.caption(f"Threshold: {THRESHOLD:.3f}  |  Model AUC: {MODEL_AUC}")

    st.subheader("Most Deviant Features")
    z_scores = [(f, abs(X[0, i])) for i, f in enumerate(features)]
    z_scores.sort(key=lambda x: -x[1])

    for name, abs_z in z_scores[:5]:
        direction = "high" if raw_values[name] > z_params[name]["mean"] else "low"
        st.text(f"{FEAT_CN.get(name, name)}: {abs_z:.2f} sigma ({direction})")

with gauge_col:
    if prob < 0.15:
        color, level_text = "#27ae60", "Low Risk"
    elif prob < 0.30:
        color, level_text = "#f39c12", "Medium Risk"
    else:
        color, level_text = "#e74c3c", "High Risk"
    st.markdown(f"""
    <div style="background:#f0f2f6; border-radius:15px; padding:30px; text-align:center;">
        <div style="font-size:14px; color:#666; margin-bottom:10px;">Risk Level</div>
        <div style="font-size:64px; font-weight:bold; color:{color};">{level_text}</div>
        <div style="font-size:24px; color:#333; margin-top:5px;">{prob:.1%}</div>
        <div style="height:20px; background:#ddd; border-radius:10px; margin-top:15px;">
            <div style="height:100%; width:{min(prob*100, 100)}%; background:{color}; border-radius:10px;"></div>
        </div>
        <div style="display:flex; justify-content:space-between; font-size:11px; color:#999; margin-top:5px;">
            <span>0%</span><span>25%</span><span>50%</span><span>75%</span><span>100%</span>
        </div>
    </div>
    """, unsafe_allow_html=True)

# ---- Feature Contribution Details ----
st.markdown("---")
with st.expander("📊 Feature Contribution Details (Z-score)"):
    contrib_data = []
    for i, f in enumerate(features):
        z_val = X[0, i]
        raw = raw_values[f]
        p = z_params[f]
        contrib_data.append({
            "Feature": FEAT_CN.get(f, f),
            "Variable": f,
            "Raw Value": f"{raw:.3f}" if raw == int(raw) else f"{raw:.1f}",
            "Z-score": f"{z_val:.3f}",
            "Reference (mean ± std)": f"{p['mean']:.1f} ± {p['std']:.1f}",
        })
    st.dataframe(contrib_data, use_container_width=True)

# ---- Reference Ranges ----
st.markdown("---")
with st.expander("📋 NHANES Clinical Reference Ranges (Training Set n=31,595)"):
    ref_data = {
        "Age": "20-85 yr (mean: 50.7)",
        "Income/Poverty": "0.01-5.0 (mean: 2.64)",
        "Alkaline Phosphatase": "7-729 U/L (mean: 72.7)",
        "Chloride": "73-133 mmol/L (mean: 103.1)",
        "Globulin": "0.7-7.5 g/dL (mean: 2.96)",
        "Serum Osmolality": "201-334 mOsm/kg (mean: 279.1)",
        "Waist Circumference": "55.5-187.5 cm (mean: 99.9)",
        "Diastolic BP": "10-139 mmHg (mean: 71.6)",
        "RBC Count": "1.67-8.3 x10^6/uL (mean: 4.68)",
        "RDW": "10.7-37.8% (mean: 13.4)",
        "PHQ-9": "0-27 (mean: 3.3)",
    }
    for k, v in ref_data.items():
        st.text(f"{k}: {v}")

    st.caption("")
    st.caption("Binary variable prevalence (training set):")
    st.text("CKD: 20.7%  |  Hypertension: 36.9%  |  Cancer: 10.4%")
    st.text("Non-Hispanic Black: 20.0%  |  Education <9th: 9.2%")
