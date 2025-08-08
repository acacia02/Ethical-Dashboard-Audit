import numpy as np, pandas as pd


default_missing_codes = {7, 9, 77, 99, 777, 999}

def clean_categorical(s, missing_codes=default_missing_codes):
    s = s.astype("string").str.strip()
    s = s.replace({"": np.nan, ".": np.nan, "NA": np.nan, "N/A": np.nan})
    s = pd.to_numeric(s, errors="coerce")
    return s.mask(s.isin(missing_codes), np.nan)

def clean_numeric(s, missing_codes=default_missing_codes):
    s = pd.to_numeric(s, errors="coerce")
    return s.mask(s.isin(missing_codes), np.nan)

