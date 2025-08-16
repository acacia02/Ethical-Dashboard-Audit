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
