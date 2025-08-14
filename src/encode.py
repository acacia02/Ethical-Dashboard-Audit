import pandas as pd
from pathlib import Path

# Load the missingness-handled file
df = pd.read_csv("data/processed/model_step_missingness.csv", index_col=0)

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

# Optional: keep only the columns actually being modelled (plus target)
keep = categorical + numeric + [target]
df = df[[c for c in keep if c in df.columns]].copy()

# enforcing numeric dtypes for numeric columns
for col in numeric:
    if col in df.columns:
        df[col] = pd.to_numeric(df[col], errors="coerce")

# Build X/y
X = df.drop(columns=[target])
y = df[target]

# Map Education_Level to ordinal codes
education_order = {
    "Less than 9th grade": 1,
    "9-11th grade (and 12th grade with no diploma)": 2,
    "High school graduate/GED or equivalent": 3,
    "Some college or AA degree": 4,
    "College graduate or above": 5,
    "NotAsked": 0  # Keep missing/unknown as 0
}

if "Education_Level" in df.columns:
    df["Education_Level"] = df["Education_Level"].map(education_order)
    # Remove from categorical list so it doesn't get one-hot encoded
    categorical = [c for c in categorical if c != "Education_Level"]
    numeric.append("Education_Level")


# One-hot encode categoricals
#    - drop_first=True avoids dummy trap for linear models
#    - dummy_na=False because we explicitly kept "NotAsked" as a real category
X_encode = pd.get_dummies(
    X,
    columns=[c for c in categorical if c in X.columns],
    drop_first=True, # drop_first=True for logistic; for tree models switch to drop_first=False
    dummy_na=False
)

# Concatenate y back on, save the frozen model-ready dataset
out = pd.concat([X_encode, y], axis=1)
Path("data").mkdir(parents=True, exist_ok=True)
out.to_csv("data/df_model_ready.csv", index=True)


print(f"X_encode shape: {X_encode.shape} | y shape: {y.shape}")
print(f"Total rows: {out.shape[0]:,} | Total cols (features + target): {out.shape[1]:,}")
print("\nSample encoded columns:")
print([c for c in X_encode.columns[:12]])
print("\nSaved -> data/df_model_ready.csv")
