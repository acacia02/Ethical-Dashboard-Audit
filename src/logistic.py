import pandas as pd
from pathlib import Path
import numpy as np, json, matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import (
    classification_report, roc_auc_score, average_precision_score,
    precision_recall_curve, f1_score, confusion_matrix,
    RocCurveDisplay, PrecisionRecallDisplay, precision_score, recall_score
)
import joblib

# Load & basic prep
df = pd.read_csv("data/processed/df_model_ready.csv")
if "ID" in df.columns:
    df = df.drop(columns=["ID"])

X = df.drop(columns=["Diabetes_Diagnosed"])
y = df["Diabetes_Diagnosed"].astype(int)
print("[Class balance]", y.value_counts().to_dict())

# Remove obvious leakage features
leak_prefixes = ["Takes_Insulin", "Prediabetes_Diagnosed"]
leak_cols = [c for c in X.columns if any(c.startswith(p) for p in leak_prefixes)]
if leak_cols:
    print("[Leakage removed]", leak_cols)
    X = X.drop(columns=leak_cols)

# Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

# Train logistic pipeline
pipe = make_pipeline(
    StandardScaler(with_mean=False),
    LogisticRegression(max_iter=2000, class_weight="balanced")
)
pipe.fit(X_train, y_train)

# Baseline probs and metrics (uncalibrated)
y_prob_lr = pipe.predict_proba(X_test)[:, 1]
prec_lr, rec_lr, thr_lr = precision_recall_curve(y_test, y_prob_lr)
f1s_lr = [f1_score(y_test, (y_prob_lr >= t).astype(int)) for t in thr_lr]
i_lr = int(np.argmax(f1s_lr))
thr_uncal = float(thr_lr[i_lr])
print(f"[LR Uncal] ROC {roc_auc_score(y_test, y_prob_lr):.3f} | PR {average_precision_score(y_test, y_prob_lr):.3f} | "
      f"thr {thr_uncal:.3f} | P {prec_lr[i_lr]:.3f} R {rec_lr[i_lr]:.3f} F1 {f1s_lr[i_lr]:.3f}")

# Optional: calibrate logistic
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
cal_lr = CalibratedClassifierCV(estimator=pipe, method="isotonic", cv=cv)
cal_lr.fit(X_train, y_train)
y_prob_lr_cal = cal_lr.predict_proba(X_test)[:, 1]
prec_cal, rec_cal, thr_cal = precision_recall_curve(y_test, y_prob_lr_cal)
f1s_cal = [f1_score(y_test, (y_prob_lr_cal >= t).astype(int)) for t in thr_cal]
i_cal = int(np.argmax(f1s_cal))
thr_calib = float(thr_cal[i_cal])
print(f"[LR Cal]   ROC {roc_auc_score(y_test, y_prob_lr_cal):.3f} | PR {average_precision_score(y_test, y_prob_lr_cal):.3f} | "
      f"thr {thr_calib:.3f} | P {prec_cal[i_cal]:.3f} R {rec_cal[i_cal]:.3f} F1 {f1s_cal[i_cal]:.3f}")

# Choose which version to use in dashboard/audit
# Rule: keep the one with better F1 (or set your own rule)
use_calibrated = f1s_cal[i_cal] >= f1s_lr[i_lr]

if use_calibrated:
    FINAL_PROB = y_prob_lr_cal
    FINAL_THR  = thr_calib
    version = "logistic_calibrated"
else:
    FINAL_PROB = y_prob_lr
    FINAL_THR  = thr_uncal
    version = "logistic_uncalibrated"

print(f"[LR Selected] {version} | thr {FINAL_THR:.3f}")


# Save model and reports
Path("models").mkdir(parents=True, exist_ok=True)
joblib.dump(pipe, "models/logistic_baseline.joblib")
joblib.dump(cal_lr, "models/logistic_calibrated.joblib")

Path("reports").mkdir(exist_ok=True)

# JSON report for the SELECTED version
roc = roc_auc_score(y_test, FINAL_PROB)
pr  = average_precision_score(y_test, FINAL_PROB)
prec, rec, thr = precision_recall_curve(y_test, FINAL_PROB)
f1s = [f1_score(y_test, (FINAL_PROB >= t).astype(int)) for t in thr]
j = int(np.argmax(f1s))
best_thr = float(thr[j])
y_pred_sel = (FINAL_PROB >= best_thr).astype(int)
tn, fp, fn, tp = confusion_matrix(y_test, y_pred_sel, labels=[0,1]).ravel()
report_sel = classification_report(y_test, y_pred_sel, output_dict=True)

with open(f"reports/{version}.json", "w") as f:
    json.dump({
        "model": version,
        "roc_auc": float(roc),
        "pr_auc": float(pr),
        "threshold": best_thr,
        "precision_tuned": float(prec[j]),
        "recall_tuned": float(rec[j]),
        "f1_tuned": float(f1s[j]),
        "support": {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)},
        "chosen_operating_threshold": float(FINAL_THR)  # the one you'll actually use in the app
    }, f, indent=2)

# Curves
plt.figure(); PrecisionRecallDisplay(precision=prec, recall=rec).plot()
plt.title(f"{version} PR Curve"); plt.savefig(f"reports/{version}_pr.png", bbox_inches="tight"); plt.close()
plt.figure(); RocCurveDisplay.from_predictions(y_test, FINAL_PROB)
plt.title(f"{version} ROC Curve"); plt.savefig(f"reports/{version}_roc.png", bbox_inches="tight"); plt.close()

# Odds ratios
lr_est = pipe.named_steps["logisticregression"]
coefs = lr_est.coef_.ravel()
odds  = np.exp(coefs)
pd.DataFrame({"feature": X.columns, "beta": coefs, "odds_ratio": odds}) \
  .sort_values("odds_ratio", ascending=False) \
  .to_csv("reports/lr_feature_odds.csv", index=False)

print(f"[LR Saved] reports/{version}.json and curves")


# Fairness slices (after FINAL_PROB and FINAL_THR)
def fairness_slice(df_all, test_idx, y_true, y_prob, threshold, group_col, out_csv):
    df_test = df_all.loc[test_idx].copy()
    y_pred = (y_prob >= threshold).astype(int)
    out = []
    for g, idx in df_test.groupby(group_col).indices.items():
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

fairness_slice(df, X_test.index, y_test, FINAL_PROB, FINAL_THR,
               "AgeGroup", "reports/fairness_lr_age.csv")
fairness_slice(df, X_test.index, y_test, FINAL_PROB, FINAL_THR,
               "Education_Level", "reports/fairness_lr_education.csv")
