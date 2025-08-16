import pandas as pd
from pathlib import Path
import numpy as np, json, matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.metrics import (
    roc_auc_score, average_precision_score, precision_recall_curve,
    f1_score, confusion_matrix, classification_report,
    RocCurveDisplay, PrecisionRecallDisplay, precision_score, recall_score
)

import xgboost as xgb

# Load & prep
df = pd.read_csv("data/processed/df_model_ready.csv")
if "ID" in df.columns:
    df = df.drop(columns=["ID"])

X = df.drop(columns=["Diabetes_Diagnosed"])
y = df["Diabetes_Diagnosed"].astype(int)
print("[Class balance]", y.value_counts().to_dict())

# Remove leakage
leak_prefixes = ["Takes_Insulin", "Prediabetes_Diagnosed"]
leak_cols = [c for c in X.columns if any(c.startswith(p) for p in leak_prefixes)]
if leak_cols:
    print("[Leakage removed]", leak_cols)
    X = X.drop(columns=leak_cols)

# Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

# Model: XGBoost (uncalibrated)
Path("models").mkdir(exist_ok=True)
Path("reports").mkdir(exist_ok=True)

pos = int((y_train == 1).sum())
neg = int((y_train == 0).sum())
spw = max(1.0, neg / max(1, pos))  # imbalance ratio

xgb_model = xgb.XGBClassifier(
    n_estimators=2000,
    learning_rate=0.03,
    max_depth=4,
    min_child_weight=1,
    subsample=0.9,
    colsample_bytree=0.8,
    reg_lambda=1.0,
    objective="binary:logistic",
    eval_metric="aucpr",       # focus on PR behavior
    scale_pos_weight=spw,
    tree_method="hist",
    n_jobs=-1,
    random_state=42,
)

# keep XGBClassifier exactly as defined
xgb_model.fit(
    X_train, y_train,
    eval_set=[(X_test, y_test)],
    verbose=False
)
# continue with y_prob_xgb = xgb_model.predict_proba(X_test)[:, 1]


# Probabilities and global metrics
y_prob_xgb = xgb_model.predict_proba(X_test)[:, 1]
roc_xgb = roc_auc_score(y_test, y_prob_xgb)
pr_xgb  = average_precision_score(y_test, y_prob_xgb)

# Threshold tuning (maximize F1)
prec, rec, thr = precision_recall_curve(y_test, y_prob_xgb)
f1s = [f1_score(y_test, (y_prob_xgb >= t).astype(int)) for t in thr]
best_idx = int(np.argmax(f1s))
best_thr_xgb = float(thr[best_idx])

y_pred_xgb = (y_prob_xgb >= best_thr_xgb).astype(int)
tn, fp, fn, tp = confusion_matrix(y_test, y_pred_xgb, labels=[0,1]).ravel()
report_xgb = classification_report(y_test, y_pred_xgb, output_dict=True)

# Save metrics JSON
with open("reports/xgb_uncalibrated.json", "w") as f:
    json.dump({
        "model": "xgboost_uncalibrated",
        "roc_auc": float(roc_xgb),
        "pr_auc": float(pr_xgb),
        "threshold": best_thr_xgb,                      # tuned-by-F1 threshold
        "precision_tuned": float(prec[best_idx]),
        "recall_tuned": float(rec[best_idx]),
        "f1_tuned": float(f1s[best_idx]),
        "support": {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)},
        "report": report_xgb,
        "chosen_operating_threshold": best_thr_xgb      # same as threshold in streamlit
    }, f, indent=2)

# Curves
plt.figure()
PrecisionRecallDisplay(precision=prec, recall=rec).plot()
plt.title("XGBoost (uncalibrated) PR Curve")
plt.savefig("reports/xgb_pr_curve.png", bbox_inches="tight"); plt.close()

plt.figure()
RocCurveDisplay.from_predictions(y_test, y_prob_xgb)
plt.title("XGBoost (uncalibrated) ROC Curve")
plt.savefig("reports/xgb_roc_curve.png", bbox_inches="tight"); plt.close()

# Feature importance
fi_xgb = pd.DataFrame({
    "feature": X.columns,
    "importance": xgb_model.feature_importances_
}).sort_values("importance", ascending=False)
fi_xgb.to_csv("reports/xgb_feature_importance.csv", index=False)

# Persist model
xgb_model.save_model("models/xgb_model.json")

print("\nXGBoost Summary")
print(f"ROC AUC: {roc_xgb:.3f} | PR AUC: {pr_xgb:.3f} | tuned thr: {best_thr_xgb:.3f}")
print(f"Precision@tuned: {prec[best_idx]:.3f} | Recall@tuned: {rec[best_idx]:.3f} | F1@tuned: {f1s[best_idx]:.3f}")
print("Artifacts saved to ./reports and model to ./models/xgb_model.json")

# Fairness slices (AgeGroup, Education_Level) at chosen threshold
def fairness_slice(df_all, test_idx, y_true, y_prob, threshold, group_col, out_csv):
    df_test = df_all.loc[test_idx].copy()
    y_pred = (y_prob >= threshold).astype(int)
    out = []
    for g, idx in df_test.groupby(group_col, observed=True).indices.items():
        idx = list(idx)
        yt = y_true.iloc[idx]
        yp = y_pred[idx]
        prec_g = precision_score(yt, yp, zero_division=0)
        rec_g  = recall_score(yt, yp, zero_division=0)
        fnr_g  = 1 - rec_g
        out.append({"group": g, "n": len(idx), "precision": prec_g, "recall": rec_g, "FNR": fnr_g})
    pd.DataFrame(out).sort_values("group").to_csv(out_csv, index=False)
    print(f"[Fairness] wrote {out_csv}")

if "AgeGroup" not in df.columns:
    df["AgeGroup"] = pd.cut(df["Age"], bins=[0,30,45,60,100], labels=["<30","30-45","45-60","60+"])

fairness_slice(df, X_test.index, y_test, y_prob_xgb, best_thr_xgb,
               "AgeGroup", "reports/fairness_xgb_age.csv")
fairness_slice(df, X_test.index, y_test, y_prob_xgb, best_thr_xgb,
               "Education_Level", "reports/fairness_xgb_education.csv")
