import sqlite3, pandas as pd, pathlib

csv_path = pathlib.Path("data/processed/cleaned_dataset.csv")
db_path  = pathlib.Path("data/nhanes.db")
db_path.parent.mkdir(parents=True, exist_ok=True)

df = pd.read_csv(csv_path)
con = sqlite3.connect(db_path)
# Overwrite table if it exists so you can re-run safely
df.to_sql("nhanes", con, if_exists="replace", index=False)
con.close()
print("Wrote table nhanes to data/nhanes.db")
