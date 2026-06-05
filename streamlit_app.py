"""
夜尿症预测 App — CatBoost 模型 (17特征)
"""
import streamlit as st
import numpy as np
import joblib
import os

st.set_page_config(page_title="夜尿症风险预测", page_icon="🌙", layout="wide")
st.title("🌙 夜尿症风险预测")
st.caption("基于 CatBoost 模型 (AUC=0.7446)，训练数据：NHANES 2005–2023 (n=39,494)")

# ---- 加载模型 ----
@st.cache_resource
def load_model():
    path = os.path.join(os.path.dirname(__file__), "models", "models.pkl")
    data = joblib.load(path)
    return data["models"]["CatBoost"], data["scaler"], data["features"]

model, scaler, features = load_model()

# ---- 训练集原始均值和标准差 (用于 z-score 标准化) ----
Z_PARAMS = {
    "RIDAGEYR":    (50.6626, 17.6286),
    "INDFMPIR":    (2.6000,  1.5477),
    "LBXSAPSI":    (72.7436, 25.8047),
    "LBXSCLSI":    (103.0547, 3.1290),
    "LBXSGB":      (2.9556,  0.4601),
    "LBXSOSSI":    (279.1403, 5.3739),
    "BMXWAIST":    (99.8657, 16.2302),
    "BP_DIA_MEAN": (71.5749, 11.5407),
    "LBXRBCSI":    (4.6766,  0.4987),
    "LBXRDW":      (13.4292, 1.3944),
    "DPQ_TOTAL":   (3.2789,  4.3330),
}

THRESHOLD = 0.316

# ---- 侧边栏 ----
st.sidebar.header("关于本工具")
st.sidebar.markdown("""
本工具使用 **17 个临床与人口学特征** 预测夜尿症风险。

### 特征筛选方法
1. **Boruta** 特征选择 (perc=100)
2. **1000 次 LASSO 稳定性选择** (C=0.0336, ≥95%)
3. **Spearman** 相关性过滤 (|ρ| < 0.5)

### 模型信息
- **算法**: CatBoost
- **测试集 AUC**: 0.7446
- **阈值**: 0.316 (F1 最优)

### 数据来源
NHANES 2005–2023, n = 39,494 (夜尿症 13,097 例, 33.2%)
""")

# ========================
# 输入表单
# ========================
col1, col2, col3 = st.columns(3)

with col1:
    st.subheader("📋 人口学特征")
    age = st.number_input("年龄 (岁)", min_value=20, max_value=85, value=50, step=1)
    income_poverty = st.number_input("收入/贫困线比值", min_value=0.0, max_value=5.0, value=2.3, step=0.1,
                                     help="INDFMPIR: 家庭收入与联邦贫困线的比值, 0-5")
    race_black = st.checkbox("非西班牙裔黑人", value=False)
    edu_lt9 = st.checkbox("受教育程度 < 9 年级", value=False)

with col2:
    st.subheader("🩺 既往病史")
    hypertension = st.checkbox("高血压病史", value=False)
    ckd = st.checkbox("慢性肾脏病 (CKD)", value=False)
    cancer = st.checkbox("癌症/恶性肿瘤史", value=False)
    diabetes = st.checkbox("糖尿病 (HbA1c ≥ 6.5% 或已诊断)", value=False)

    st.subheader("📏 体格检查")
    waist = st.number_input("腰围 (cm)", min_value=55.0, max_value=188.0, value=98.0, step=0.5)
    dia_bp = st.number_input("舒张压 (mmHg)", min_value=40.0, max_value=140.0, value=72.0, step=0.5)

with col3:
    st.subheader("🔬 实验室检查")
    rbc = st.number_input("红细胞计数 (×10⁶/µL)", min_value=1.5, max_value=8.5, value=4.7, step=0.1)
    rdw = st.number_input("红细胞分布宽度 RDW (%)", min_value=10.0, max_value=40.0, value=13.2, step=0.1)
    globulin = st.number_input("球蛋白 (g/dL)", min_value=0.5, max_value=7.5, value=2.9, step=0.1)
    alp = st.number_input("碱性磷酸酶 ALP (U/L)", min_value=5, max_value=750, value=69, step=1)
    chloride = st.number_input("氯离子 (mmol/L)", min_value=70.0, max_value=135.0, value=103.0, step=0.5)
    osmolality = st.number_input("血清渗透压 (mOsm/kg)", min_value=200.0, max_value=335.0, value=279.0, step=0.5)

    st.subheader("🧠 心理健康")
    phq9 = st.number_input("PHQ-9 抑郁量表总分", min_value=0, max_value=27, value=2, step=1,
                           help="Patient Health Questionnaire-9, 范围 0–27, ≥10 提示中重度抑郁")

# ---- 特征工程 ----
def zscore(val, mean, std):
    return (val - mean) / std

input_dict = {
    "RIDAGEYR":    zscore(age, *Z_PARAMS["RIDAGEYR"]),
    "INDFMPIR":    zscore(income_poverty, *Z_PARAMS["INDFMPIR"]),
    "CKD":         1.0 if ckd else 0.0,
    "LBXSAPSI":    zscore(alp, *Z_PARAMS["LBXSAPSI"]),
    "LBXSCLSI":    zscore(chloride, *Z_PARAMS["LBXSCLSI"]),
    "LBXSGB":      zscore(globulin, *Z_PARAMS["LBXSGB"]),
    "LBXSOSSI":    zscore(osmolality, *Z_PARAMS["LBXSOSSI"]),
    "BMXWAIST":    zscore(waist, *Z_PARAMS["BMXWAIST"]),
    "BPQ020":      1.0 if hypertension else 0.0,
    "BP_DIA_MEAN": zscore(dia_bp, *Z_PARAMS["BP_DIA_MEAN"]),
    "LBXRBCSI":    zscore(rbc, *Z_PARAMS["LBXRBCSI"]),
    "LBXRDW":      zscore(rdw, *Z_PARAMS["LBXRDW"]),
    "DPQ_TOTAL":   zscore(phq9, *Z_PARAMS["DPQ_TOTAL"]),
    "LBXHA":       1.0 if diabetes else 0.0,
    "MCQ220":      1.0 if cancer else 0.0,
    "Race_NH_Black": 1.0 if race_black else 0.0,
    "Edu_Lt9th":     1.0 if edu_lt9 else 0.0,
}

X = np.array([[input_dict[f] for f in features]])
X_scaled = scaler.transform(X)

prob = model.predict_proba(X_scaled)[0, 1]
pred = 1 if prob >= THRESHOLD else 0

# ---- 结果输出 ----
st.markdown("---")
st.header("预测结果")

risk_col, gauge_col = st.columns([1, 2])

with risk_col:
    st.metric("预测概率", f"{prob:.1%}")

    if pred == 1:
        st.error("⚠️ **高风险** — 预测为夜尿症阳性")
    else:
        st.success("✅ **低风险** — 预测为夜尿症阴性")

    st.caption(f"判定阈值: {THRESHOLD:.3f}  |  模型 AUC: 0.7446")

    st.subheader("偏离均值最多的特征")
    contributions = []
    for name, val in input_dict.items():
        contributions.append((name, abs(val)))
    contributions.sort(key=lambda x: x[1], reverse=True)

    labels_cn = {
        "RIDAGEYR": "年龄", "INDFMPIR": "收入/贫困线", "CKD": "慢性肾脏病",
        "LBXSAPSI": "碱性磷酸酶", "LBXSCLSI": "氯离子", "LBXSGB": "球蛋白",
        "LBXSOSSI": "渗透压", "BMXWAIST": "腰围", "BPQ020": "高血压史",
        "BP_DIA_MEAN": "舒张压", "LBXRBCSI": "红细胞计数", "LBXRDW": "RDW",
        "DPQ_TOTAL": "PHQ-9 抑郁评分", "LBXHA": "糖尿病", "MCQ220": "癌症史",
        "Race_NH_Black": "非西班牙裔黑人", "Edu_Lt9th": "教育 < 9 年级",
    }
    for name, abs_z in contributions[:5]:
        direction = "↑" if input_dict[name] > 0 else "↓"
        st.text(f"{direction} {labels_cn.get(name, name)}: {abs_z:.2f} σ")

with gauge_col:
    if prob < 0.15:
        color, level_text = "#27ae60", "低风险"
    elif prob < 0.30:
        color, level_text = "#f39c12", "中风险"
    else:
        color, level_text = "#e74c3c", "高风险"
    st.markdown(f"""
    <div style="background:#f0f2f6; border-radius:15px; padding:30px; text-align:center;">
        <div style="font-size:14px; color:#666; margin-bottom:10px;">风险等级</div>
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

# ---- 参考范围 ----
st.markdown("---")
with st.expander("📊 NHANES 临床参考范围"):
    ref_data = {
        "年龄": "20–85 岁 (中位数: 51)",
        "收入/贫困线": "0.01–5.0 (中位数: 2.3)",
        "碱性磷酸酶 ALP": "7–729 U/L (中位数: 69)",
        "氯离子": "73–133 mmol/L (中位数: 103)",
        "球蛋白": "0.7–7.5 g/dL (中位数: 2.9)",
        "血清渗透压": "201–334 mOsm/kg (中位数: 279)",
        "腰围": "55.5–187.5 cm (中位数: 98.5)",
        "舒张压": "10–139 mmHg (中位数: 71.3)",
        "红细胞计数": "1.67–8.3 ×10⁶/µL (中位数: 4.66)",
        "RDW": "10.7–37.8% (中位数: 13.2)",
        "PHQ-9": "0–27 分 (中位数: 2)",
    }
    for k, v in ref_data.items():
        st.text(f"{k}: {v}")
