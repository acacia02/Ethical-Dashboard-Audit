import pandas as pd
from pathlib import Path

# Load the missingness-handled file
df = pd.read_csv("data/model_step_missingness.csv", index_col=0)

# define target and keep only clear Yes/No rows
target = "Diabetes_Diagnosed"
df = df[df[target].isin(["Yes", "No"])].copy()
df[target] = (df[target] == "Yes").astype(int)  # Yes=1, No=0

# columns that are categorical vs numeric
categorical = [
    "Race_Ethnicity",
    "Education_Level",
    "Marital_Status",
    "Gender",
    "Country_of_Birth",
    "Citizenship_Status",
    "Prediabetes_Diagnosed",
    "Takes_Insulin",
    # "Diabetes_Diagnosed"
]
numeric = [
    "Age",
    "Poverty_Income_Ratio",
    "Family_Size"
]

# Optional: keep only the columns we actually intend to model with (plus target)
keep = categorical + numeric + [target]
df = df[[c for c in keep if c in df.columns]].copy()

# Build X/y
X = df.drop(columns=[target])
y = df[target]

# One-hot encode categoricals
#    - drop_first=True avoids dummy trap for linear models
#    - dummy_na=False because we explicitly kept "NotAsked" as a real category
X_enc = pd.get_dummies(
    X,
    columns=[c for c in categorical if c in X.columns],
    drop_first=True,
    dummy_na=False
)

# Concatenate y back on, save the frozen model-ready dataset
out = pd.concat([X_enc, y], axis=1)

Path("data").mkdir(parents=True, exist_ok=True)
out.to_csv("data/df_model_ready.csv", index=True)

print("=== Encoding summary ===")
print(f"X_enc shape: {X_enc.shape} | y shape: {y.shape}")
print(f"Total rows: {out.shape[0]:,} | Total cols (features + target): {out.shape[1]:,}")
print("\nSample encoded columns:")
print([c for c in X_enc.columns[:12]])
print("\nSaved -> data/df_model_ready.csv")
