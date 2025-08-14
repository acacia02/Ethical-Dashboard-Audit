import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.metrics import classification_report, roc_auc_score
import numpy as np
import joblib
from sklearn.ensemble import RandomForestClassifier

# Load
df = pd.read_csv("data/processed/df_model_ready.csv")

if "ID" in df.columns:
    df = df.drop(columns=["ID"])


# train/test split
X = df.drop(columns=["Diabetes_Diagnosed"])
y = df["Diabetes_Diagnosed"]

# Remove obvious leakage features
leak_prefixes = ["Takes_Insulin", "Prediabetes_Diagnosed"]
leak_cols = [c for c in X.columns if any(c.startswith(p) for p in leak_prefixes)]
if leak_cols:
    print("[Leakage removed]", leak_cols)
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
    LogisticRegression(max_iter=2000, n_jobs=None)
)

pipe.fit(X_train, y_train)

# Evaluate
y_pred = pipe.predict(X_test)
y_prob = pipe.predict_proba(X_test)[:, 1]

print("\n[Classification Report]")
print(classification_report(y_test, y_pred, digits=3))

print(f"[ROC AUC] {roc_auc_score(y_test, y_prob):.3f}")

# Coefficients -> Odds Ratios (for interpretation)
# Grab feature names after pipeline
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
