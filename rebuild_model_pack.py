"""
Rebuild deploy/models_16feat.pkl — clean version
Remove redundant StandardScaler; z_params covers all 16 features uniformly.
Prediction path: raw_input -> (raw - mean) / std -> model.predict_proba()

Key facts:
  - nhanes_ml_ready.csv: continuous cols are pre-z-scored (mean~0, std~1)
  - Binary cols are raw 0/1 (mean=proportion, std=sqrt(p*(1-p)))
  - Training scaler: near-identity for continuous, real z-score for binary
  - Model was trained on fully z-scored values for all 16 features

So z_params stores original-scale mean/std for every feature:
  Continuous -> raw NHANES scale (e.g. age mean=50.66, std=17.63)
  Binary     -> training 0/1 proportion (e.g. CKD mean=0.207, std=0.405)
"""
import numpy as np
import pandas as pd
import pickle, os
from sklearn.model_selection import train_test_split

RANDOM_STATE = 42
FEATURES_16 = [
    "RIDAGEYR", "INDFMPIR", "CKD", "LBXSAPSI", "LBXSCLSI",
    "LBXSGB", "LBXSOSSI", "BMXWAIST", "BPQ020", "BP_DIA_MEAN",
    "LBXRBCSI", "LBXRDW", "DPQ_TOTAL", "MCQ220",
    "Race_NH_Black", "Edu_Lt9th",
]
BINARY_FEATURES = {"CKD", "BPQ020", "MCQ220", "Race_NH_Black", "Edu_Lt9th"}

PIPELINE_DIR = "C:/Users/34807/Desktop/nocturia/output/ml_results/pipeline_16feat"
EXISTING_DEPLOY = "C:/Users/34807/Desktop/nocturia/deploy/models/models_16feat.pkl"

# -- 1. Load trained CatBoost --
with open(f"{PIPELINE_DIR}/models.pkl", "rb") as f:
    bundle = pickle.load(f)
catboost_model = bundle["models"]["CatBoost"]
print("[1/2] Loaded CatBoost model from pipeline output")

# -- 2. Build z_params: original-scale mean/std for every feature --
import joblib
old = joblib.load(EXISTING_DEPLOY)
old_z_params = old["z_params"]
old_scaler = old["scaler"]

z_params = {}
for i, f in enumerate(FEATURES_16):
    if f in BINARY_FEATURES:
        # Model was trained on z-scored binary values:
        #   z = (0/1 - proportion) / sqrt(p*(1-p))
        # The scaler's mean_/scale_ ARE the training 0/1 proportion and std
        mean_v = float(old_scaler.mean_[i])
        std_v = float(old_scaler.scale_[i])
    else:
        # Continuous: use original NHANES-scale mean/std from existing deploy
        # The scaler has mean_~0, scale_~1 (fitted on pre-z-scored data)
        mean_v = old_z_params[f]["mean"]
        std_v = old_z_params[f]["std"]
    z_params[f] = {"mean": mean_v, "std": std_v}
    print(f"  {f:<20s}  mean={mean_v:>10.4f}  std={std_v:>10.4f}")

# -- 3. Verify old path == new path for a test case --
print("\n[Verify] Old (2-step) vs New (1-step) prediction...")

test_raw = {
    "RIDAGEYR": 50, "INDFMPIR": 2.3, "CKD": 0, "LBXSAPSI": 69, "LBXSCLSI": 103,
    "LBXSGB": 2.9, "LBXSOSSI": 279, "BMXWAIST": 98, "BPQ020": 0, "BP_DIA_MEAN": 72,
    "LBXRBCSI": 4.7, "LBXRDW": 13.2, "DPQ_TOTAL": 2, "MCQ220": 0,
    "Race_NH_Black": 0, "Edu_Lt9th": 0,
}

# Old path: z_params z-score -> scaler -> model
X_old = np.zeros((1, 16))
for i, f in enumerate(FEATURES_16):
    raw = test_raw[f]
    if f in BINARY_FEATURES:
        X_old[0, i] = float(raw)  # pass through as 0/1
    else:
        X_old[0, i] = (raw - old_z_params[f]["mean"]) / old_z_params[f]["std"]
X_old_final = old_scaler.transform(X_old)
prob_old = catboost_model.predict_proba(X_old_final)[0, 1]

# New path: single z_params z-score -> model
X_new = np.zeros((1, 16))
for i, f in enumerate(FEATURES_16):
    raw = test_raw[f]
    X_new[0, i] = (raw - z_params[f]["mean"]) / z_params[f]["std"]
prob_new = catboost_model.predict_proba(X_new)[0, 1]

print(f"  Old path (2-step): {prob_old:.8f}")
print(f"  New path (1-step): {prob_new:.8f}")
print(f"  Difference:        {abs(prob_old - prob_new):.12f}")
assert abs(prob_old - prob_new) < 1e-10, "Predictions do not match!"
print("  OK - predictions are identical")

# -- 4. Save --
OUT_PATH = "C:/Users/34807/Desktop/nocturia/deploy/models/models_16feat.pkl"
pack = {
    "model": catboost_model,
    "features": FEATURES_16,
    "z_params": z_params,
    "threshold": 0.336,
    "auc": 0.7428,
}
joblib.dump(pack, OUT_PATH)
print(f"\n[2/2] Saved -> {OUT_PATH}")
print(f"  Contains: model, features({len(FEATURES_16)}), z_params({len(z_params)}), threshold=0.336, auc=0.7428")
print(f"  Removed: scaler (all standardization now in z_params)")
