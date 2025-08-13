import pandas as pd

# Loading cleaned data
df = pd.read_csv("data/processed/cleaned_dataset.csv", index_col=0)

# 
cat = [
    "Race_Ethnicity", "Education_Level", "Marital_Status", "Gender",
    "Prediabetes_Diagnosed", "Citizenship_Status", "Diabetes_Diagnosed",
    "Takes_Insulin", "Country_of_Birth"
]
num = [
    "Age", "Poverty_Income_Ratio", "Family_Size"   
]

# keep NotAsked, drop Unknown
# Normalize "Not Asked" to "NotAsked"
# Turn "Unknown" into NaN so we can drop those rows
for c in cat:
    df[c] = df[c].replace({"Not Asked": "NotAsked", "Unknown": pd.NA})

# Drop any rows that have NaN in the listed categorical columns
before = len(df)
df = df.dropna(subset=cat, how="any")
print(f"Dropped {before - len(df)} rows due to 'Unknown' in categorical columns.")

# numerics: fill NaN with column median
for c in num:
    if c in df.columns:
        df[c] = df[c].fillna(df[c].median())

# Quick check
print(df.isna().sum().sort_values(ascending=False).head(10))

# Save this as the base for the next steps (encoding/scaling)
# out_path = "data/model_step_missingness.csv"
# df.to_csv(out_path, index=True)
# print(f"\nSaved -> {out_path} | Rows: {df.shape[0]:,} | Cols: {df.shape[1]:,}")

