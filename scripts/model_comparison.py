import json, pandas as pd

rows = []
for name, path in [
    ("Logistic_cal", "reports/logistic_calibrated.json"),
    ("XGBoost",      "reports/xgb_uncalibrated.json"),
]:
    with open(path) as f: d = json.load(f)
    rows.append({
        "Model": name,
        "ROC_AUC": round(d["roc_auc"], 3),
        "PR_AUC": round(d["pr_auc"], 3),
        "Threshold": round(d.get("chosen_operating_threshold", d["threshold"]), 3),
        "Precision": round(d["precision_tuned"], 3),
        "Recall": round(d["recall_tuned"], 3),
        "F1": round(d["f1_tuned"], 3),
    })

pd.DataFrame(rows).to_csv("reports/model_comparison.csv", index=False)
print("Wrote reports/model_comparison.csv")
