from pathlib import Path
from src.data_pipeline import build_processed

raw_dir = "data/raw"
out = Path("data/processed/nhanes_clean.parquet")

df = build_processed(raw_dir)
out.parent.mkdir(parents=True, exist_ok=True)
df.to_parquet(out, index=False)
print(f"Saved {out} with shape {df.shape}")
