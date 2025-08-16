import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.metrics import (classification_report, roc_auc_score, average_precision_score, precision_recall_curve,
f1_score, confusion_matrix, classification_report, RocCurveDisplay, PrecisionRecallDisplay)
import numpy as np, json, matplotlib.pyplot as plt
import joblib
from sklearn.ensemble import RandomForestClassifier
import xgboost as xgb


# Load
df = pd.read_csv("data/processed/df_model_ready.csv")

if "ID" in df.columns:
    df = df.drop(columns=["ID"])


# train/test split
X = df.drop(columns=["Diabetes_Diagnosed"])
y = df["Diabetes_Diagnosed"]
print("[Class balance]", y.value_counts().to_dict())

# Remove obvious leakage features
leak_prefixes = ["Takes_Insulin", "Prediabetes_Diagnosed"]
leak_cols = [c for c in X.columns if any(c.startswith(p) for p in leak_prefixes)]
if leak_cols:
    # print("[Leakage removed]", leak_cols)
    X = X.drop(columns=leak_cols)



# Train/test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)


# getting rid of duplicates 
# coef_df = pd.DataFrame({"feature": X.columns, "beta": lr.coef_.ravel()})
# coef_df["odds_ratio"] = np.exp(coef_df["beta"])
# coef_df = coef_df.drop_duplicates(subset="feature")
# print(coef_df.sort_values("odds_ratio", ascending=False).head(10).to_string(index=False))
# print(coef_df.sort_values("odds_ratio").head(10).to_string(index=False))


# Pipeline: scale -> logistic
pipe = make_pipeline(
    StandardScaler(with_mean=False),  # safe for sparse-ish/dummy-heavy matrices
    LogisticRegression(max_iter=2000, class_weight="balanced", n_jobs=None) # class_weight="balanced" added to make the prediction more precise
)

pipe.fit(X_train, y_train)

# Evaluate
y_pred = pipe.predict(X_test)
y_prob = pipe.predict_proba(X_test)[:, 1]

# finding the threshold to maximize F1
prec, rec, thr = precision_recall_curve(y_test, y_prob)
f1s = [f1_score(y_test, (y_prob >= t).astype(int)) for t in thr]
best_idx = int(np.argmax(f1s))
best_thr = float(thr[best_idx])
print(f"\n[Tuned Threshold] {best_thr:.3f}")
print(f"Precision @ tuned: {prec[best_idx]:.3f}")
print(f"Recall @ tuned: {rec[best_idx]:.3f}")
print(f"F1 @ tuned: {f1s[best_idx]:.3f}")


# printing reports
print(f"[PR AUC] {average_precision_score(y_test, y_prob):.3f}")
print("\n[Classification Report]")
print(classification_report(y_test, y_pred, digits=3))

print(f"[ROC AUC] {roc_auc_score(y_test, y_prob):.3f}")

# Coefficients to Odds Ratios (for interpretation)
# feature names after pipeline
lr = pipe.named_steps["logisticregression"]
scaler = pipe.named_steps["standardscaler"]

# StandardScaler(with_mean=False) keeps column order; names = X.columns
feature_names = X.columns.tolist()

coefs = lr.coef_.ravel()
odds = np.exp(coefs)  # OR = e^beta

coef_df = pd.DataFrame({
    "feature": feature_names,
    "beta": coefs,
    "odds_ratio": odds
}).sort_values("odds_ratio", ascending=False)

print("\n[Top positive odds ratios]")
print(coef_df.head(10).to_string(index=False))

print("\n[Top negative odds ratios]")
print(coef_df.sort_values('odds_ratio').head(10).to_string(index=False))

# Save
Path("models").mkdir(parents=True, exist_ok=True)
import joblib
joblib.dump(pipe, "models/logistic_baseline.joblib")
print("\nSaved -> models/logistic_baseline.joblib")







# RANDOM FOREST
rf_model = RandomForestClassifier(
    n_estimators=400,
    random_state=42,
    class_weight="balanced_subsample"
)
rf_model.fit(X_train, y_train)

y_pred_rf = rf_model.predict(X_test)
y_prob_rf = rf_model.predict_proba(X_test)[:, 1]

print("\nRandom Forest")
print(classification_report(y_test, y_pred_rf, digits=3))
print(f"[ROC AUC] {roc_auc_score(y_test, y_prob_rf):.3f}")

# Feature importance
fi_df = pd.DataFrame({
    "feature": X.columns,
    "importance": rf_model.feature_importances_
}).sort_values("importance", ascending=False)

print("\n[Top Random Forest Features]")
print(fi_df.head(10).to_string(index=False))

# Save RF model
joblib.dump(rf_model, "models/random_forest.joblib")

print("\nSaved models to 'models/' folder.")








# XGBOOST
Path("models").mkdir(exist_ok=True)
Path("reports").mkdir(exist_ok=True)

# handle imbalance with scale_pos_weight = neg/pos
pos = int((y_train == 1).sum())
neg = int((y_train == 0).sum())
spw = max(1.0, neg / max(1, pos))  # avoid div-by-zero

xgb_model = xgb.XGBClassifier(
    n_estimators=600,
    learning_rate=0.05,
    max_depth=4,
    subsample=0.9,
    colsample_bytree=0.8,
    reg_lambda=1.0,
    objective="binary:logistic",
    eval_metric="aucpr",           # optimize for PR behavior
    scale_pos_weight=spw,
    n_jobs=-1,
    random_state=42
)

# overfitting
xgb_model.fit(
    X_train, y_train,
    eval_set=[(X_test, y_test)],
    verbose=False
)

# probabilities and global metrics
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

# save metrics and curves
with open("reports/model_xgb.json", "w") as f:
    json.dump({
        "model": "xgboost",
        "roc_auc": float(roc_xgb),
        "pr_auc": float(pr_xgb),
        "threshold": best_thr_xgb,
        "precision_tuned": float(prec[best_idx]),
        "recall_tuned": float(rec[best_idx]),
        "f1_tuned": float(f1s[best_idx]),
        "support": {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)},
        "report": report_xgb
    }, f, indent=2)

# visuals
plt.figure()
PrecisionRecallDisplay(precision=prec, recall=rec).plot()
plt.title("XGBoost PR Curve")
plt.savefig("reports/xgb_pr_curve.png", bbox_inches="tight"); plt.close()

plt.figure()
RocCurveDisplay.from_predictions(y_test, y_prob_xgb)
plt.title("XGBoost ROC Curve")
plt.savefig("reports/xgb_roc_curve.png", bbox_inches="tight"); plt.close()

# feature importance
importances = xgb_model.feature_importances_
fi_xgb = pd.DataFrame({"feature": X.columns, "importance": importances})\
           .sort_values("importance", ascending=False)
fi_xgb.to_csv("reports/xgb_feature_importance.csv", index=False)

# Persist model
xgb_model.save_model("models/xgb_model.json")

print("\nXGBoost Summary")
print(f"ROC AUC: {roc_xgb:.3f} | PR AUC: {pr_xgb:.3f} | tuned thr: {best_thr_xgb:.3f}")
print(f"Precision@tuned: {prec[best_idx]:.3f} | Recall@tuned: {rec[best_idx]:.3f} | F1@tuned: {f1s[best_idx]:.3f}")
print("Artifacts saved to ./reports and model to ./models/xgb_model.json")






# REPORTS FOR LOGISTIC REGRESSION
Path("reports").mkdir(exist_ok=True)

# Probabilities
y_prob_lr = pipe.predict_proba(X_test)[:, 1]

# Global metrics
roc_lr = roc_auc_score(y_test, y_prob_lr)
pr_lr  = average_precision_score(y_test, y_prob_lr)

# Threshold tuning (maximize F1)
prec_lr, rec_lr, thr_lr = precision_recall_curve(y_test, y_prob_lr)
f1s_lr = [f1_score(y_test, (y_prob_lr >= t).astype(int)) for t in thr_lr]
best_lr_idx = int(np.argmax(f1s_lr))
best_thr_lr = float(thr_lr[best_lr_idx])
y_pred_lr_tuned = (y_prob_lr >= best_thr_lr).astype(int)

# Confusion + report
tn, fp, fn, tp = confusion_matrix(y_test, y_pred_lr_tuned, labels=[0,1]).ravel()
report_lr = classification_report(y_test, y_pred_lr_tuned, output_dict=True)

# Save JSON
with open("reports/model_logistic.json", "w") as f:
    json.dump({
        "model": "logistic_regression",
        "roc_auc": float(roc_lr),
        "pr_auc": float(pr_lr),
        "threshold": best_thr_lr,
        "precision_tuned": float(prec_lr[best_lr_idx]),
        "recall_tuned": float(rec_lr[best_lr_idx]),
        "f1_tuned": float(f1s_lr[best_lr_idx]),
        "support": {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)},
        "report": report_lr
    }, f, indent=2)

# Curves
plt.figure()
PrecisionRecallDisplay(precision=prec_lr, recall=rec_lr).plot()
plt.title("Logistic PR Curve")
plt.savefig("reports/lr_pr_curve.png", bbox_inches="tight"); plt.close()

plt.figure()
RocCurveDisplay.from_predictions(y_test, y_prob_lr)
plt.title("Logistic ROC Curve")
plt.savefig("reports/lr_roc_curve.png", bbox_inches="tight"); plt.close()

# Odds ratios (feature interpretability)
lr_est = pipe.named_steps["logisticregression"]
coefs = lr_est.coef_.ravel()
odds  = np.exp(coefs)
pd.DataFrame({"feature": X.columns, "beta": coefs, "odds_ratio": odds}) \
  .sort_values("odds_ratio", ascending=False) \
  .to_csv("reports/lr_feature_odds.csv", index=False)

print(f"[LR] ROC {roc_lr:.3f} | PR {pr_lr:.3f} | thr {best_thr_lr:.3f} | P {prec_lr[best_lr_idx]:.3f} R {rec_lr[best_lr_idx]:.3f} F1 {f1s_lr[best_lr_idx]:.3f}")









# REPORTS FOR RANDOM FOREST
Path("reports").mkdir(exist_ok=True)

# Probabilities
y_prob_rf = rf_model.predict_proba(X_test)[:, 1]

# Global metrics
roc_rf = roc_auc_score(y_test, y_prob_rf)
pr_rf  = average_precision_score(y_test, y_prob_rf)

# Threshold tuning (maximize F1)
prec_rf, rec_rf, thr_rf = precision_recall_curve(y_test, y_prob_rf)
f1s_rf = [f1_score(y_test, (y_prob_rf >= t).astype(int)) for t in thr_rf]
best_rf_idx = int(np.argmax(f1s_rf))
best_thr_rf = float(thr_rf[best_rf_idx])
y_pred_rf_tuned = (y_prob_rf >= best_thr_rf).astype(int)

# Confusion + report
tn, fp, fn, tp = confusion_matrix(y_test, y_pred_rf_tuned, labels=[0,1]).ravel()
report_rf = classification_report(y_test, y_pred_rf_tuned, output_dict=True)

# Save JSON
with open("reports/model_rf.json", "w") as f:
    json.dump({
        "model": "random_forest",
        "roc_auc": float(roc_rf),
        "pr_auc": float(pr_rf),
        "threshold": best_thr_rf,
        "precision_tuned": float(prec_rf[best_rf_idx]),
        "recall_tuned": float(rec_rf[best_rf_idx]),
        "f1_tuned": float(f1s_rf[best_rf_idx]),
        "support": {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)},
        "report": report_rf
    }, f, indent=2)

# Curves
plt.figure()
PrecisionRecallDisplay(precision=prec_rf, recall=rec_rf).plot()
plt.title("Random Forest PR Curve")
plt.savefig("reports/rf_pr_curve.png", bbox_inches="tight"); plt.close()

plt.figure()
RocCurveDisplay.from_predictions(y_test, y_prob_rf)
plt.title("Random Forest ROC Curve")
plt.savefig("reports/rf_roc_curve.png", bbox_inches="tight"); plt.close()

# Feature importance
pd.DataFrame({"feature": X.columns, "importance": rf_model.feature_importances_}) \
  .sort_values("importance", ascending=False) \
  .to_csv("reports/rf_feature_importance.csv", index=False)

print(f"[RF] ROC {roc_rf:.3f} | PR {pr_rf:.3f} | thr {best_thr_rf:.3f} | P {prec_rf[best_rf_idx]:.3f} R {rec_rf[best_rf_idx]:.3f} F1 {f1s_rf[best_rf_idx]:.3f}")




# TABLE FOR README
rows = []
for name, path in [
    ("Logistic", "reports/model_logistic.json"),
    ("RandomForest", "reports/model_rf.json"),
    ("XGBoost", "reports/model_xgb.json"),
]:
    with open(path) as f:
        d = json.load(f)
    rows.append({
        "Model": name,
        "ROC AUC": round(d["roc_auc"], 3),
        "PR AUC": round(d["pr_auc"], 3),
        "Threshold": round(d["threshold"], 3),
        "Precision@thr": round(d["precision_tuned"], 3),
        "Recall@thr": round(d["recall_tuned"], 3),
        "F1@thr": round(d["f1_tuned"], 3),
    })
pd.DataFrame(rows).to_csv("reports/model_comparison.csv", index=False)
print("Saved reports/model_comparison.csv")
