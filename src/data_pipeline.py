from pathlib import Path
import pandas as pd
from src.clean_utils import clean_categorical, clean_numeric

diabetes_keep = [
    'SEQN',    
    'DIQ010',  # Diabetes diagnosis
    'DID040',  # Age at diagnosis
    'DIQ160',  # Prediabetes
    'DIQ050',  # Taking insulin
    'DIQ070',  # Taking diabetic pills
    'DID250',  # Doctor visits
    'DIQ275',  # A1C checked
    'DIQ280',  # A1C level
    'DIQ291',  # Target A1C
    'DIQ360',  # Eye exam
    'DIQ080'   # Eye damage
]

demographics_keep = [
    "SEQN",         # Unique respondent ID
    "RIAGENDR",     # Gender
    "RIDAGEYR",     # Age (in years)
    "RIDRETH1",     # Race/Hispanic origin
    "DMDBORN4",     # Country of birth
    "DMDCITZN",     # Citizenship status
    "DMDYRSUS",     # Years in US
    "DMDEDUC2",     # Education (adults 20+)
    "DMDMARTL",     # Marital status
    "DMDFMSIZ",     # Family size
    "INDFMPIR"      # Poverty-income ratio (gold standard for SES)
]

def load_merge_raw(raw_dir: str) -> pd.DataFrame:
    raw = Path(raw_dir)
    demo = pd.read_sas(raw / "DEMO_J.xpt", format="xport")
    diq  = pd.read_sas(raw / "DIQ_J.xpt",  format="xport")
    df   = pd.merge(diq, demo, on="SEQN", how="inner")
    return df

def rename_columns(df: pd.DataFrame) -> pd.DataFrame:
    rename = {
        # Demographics
    "SEQN": "ID",
    "RIAGENDR": "Gender",
    "RIDAGEYR": "Age",
    "RIDRETH1": "Race_Ethnicity",
    "DMDBORN4": "Country_of_Birth",
    "DMDCITZN": "Citizenship_Status",
    "DMDYRSUS": "Years_in_US",
    "DMDEDUC2": "Education_Level",
    "DMDMARTL": "Marital_Status",
    "DMDFMSIZ": "Family_Size",
    "INDFMPIR": "Poverty_Income_Ratio",
    # Diabetes
    "DIQ010": "Diabetes_Diagnosed",
    "DID040": "Age_at_Diagnosis",
    "DIQ160": "Prediabetes_Diagnosed",
    "DIQ050": "Takes_Insulin",
    "DIQ070": "Takes_Diabetes_Pills",
    "DID250": "Doctor_Visits_Last_Year",
    "DIQ275": "A1C_Checked",
    "DIQ280": "Last_A1C_Value",
    "DIQ291": "Target_A1C_Value",
    "DIQ360": "Eye_Exam_Last_Year",
    "DIQ080": "Vision_Affected_by_Diabetes"
    }
    return df.rename(columns=rename)

def clean_columns(df: pd.DataFrame) -> pd.DataFrame:
    # categorical
    if "Country_of_Birth" in df: df["Country_of_Birth"] = clean_categorical(df["Country_of_Birth"], missing_codes={77,99})
    # add others as you go: Race_Ethnicity, Education, etc.

    # numerical
    if "Age" in df: df["Age"] = clean_numeric(df["Age"])
    if "Poverty_Income_Ratio" in df:
        df["Poverty_Income_Ratio"] = clean_numeric(df["Poverty_Income_Ratio"], missing_codes={77,99})
    # add BMI, A1C, etc.
    return df

def build_processed(raw_dir: str) -> pd.DataFrame:
    df = load_merge_raw(raw_dir)
    df = rename_columns(df)
    df = clean_columns(df)
    return df
