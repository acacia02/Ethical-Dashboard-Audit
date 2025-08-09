from pathlib import Path
import pandas as pd
from src.clean_utils import clean_categorical, clean_numeric
from src import mappings


# configuring/setting threshold for dropping "missing"
reports_path = Path("reports")
reports_path.mkdir(exist_ok=True)
threshold = 0.30 # 30% missing (drop column)

must_keep = {
    "Education_Level", "Marital_Status", "Prediabetes_Diagnosed",
    "Poverty_Income_Ratio", "Age", "Citizenship_Status",
    "Diabetes_Diagnosed", "Country_of_Birth", "Takes_Insulin",
    "ID", "Gender", "Race_Ethnicity", "Family_Size"
}


def load_merge_raw(raw_dir: str) -> pd.DataFrame:
    raw = Path(raw_dir)
    demo = pd.read_sas(raw / "DEMO_J.xpt", format="xport")
    diq  = pd.read_sas(raw / "DIQ_J.xpt",  format="xport")
    df   = pd.merge(diq, demo, on="SEQN", how="inner")
    return df

diabetes_keep = [
    'SEQN', 'DIQ010', 'DID040', 'DIQ160',  
    'DIQ050', 'DIQ070', 'DID250', 'DIQ275', 
    'DIQ280', 'DIQ291', 'DIQ360', 'DIQ080'  
]

demographics_keep = [
    "RIAGENDR", "RIDAGEYR", "RIDRETH1",     
    "DMDBORN4", "DMDCITZN", "DMDYRSUS", "DMDEDUC2",     
    "DMDMARTL", "DMDFMSIZ", "INDFMPIR"      
]

keep_cols = diabetes_keep + demographics_keep

# only keeping relevant columns
def restrict_columns(df, keep_cols):
    
    columns_to_keep = []
    missing_from_raw = []

    # Going through keep list one by one
    for col in keep_cols:
        if col in df.columns:
            columns_to_keep.append(col)
        else:
            missing_from_raw.append(col)

    # Logging anything I couldn't find (so I have an audit trail)
    if len(missing_from_raw) > 0:
        with open(reports_path / "missing_in_keep_cols.txt", "w", encoding="utf-8") as f:
            f.write("Columns requested but NOT found in raw data:\n")
            for col in missing_from_raw:
                f.write(f"- {col}\n")
        print(f"[warn] Some keep_cols not found. See reports/missing_in_keep_cols.txt")

    # Subset to only whats desired
    df = df[columns_to_keep].copy()
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
    if "Gender" in df:
        df["Gender"] = clean_categorical(df["Gender"])
    if "Race_Ethnicity" in df:
        df["Race_Ethnicity"] = clean_categorical(df["Race_Ethnicity"])
    if "Country_of_Birth" in df:
        df["Country_of_Birth"] = clean_categorical(df["Country_of_Birth"], missing_codes={77,99})
    if "Citizenship_Status" in df:
        df["Citizenship_Status"] = clean_categorical(df["Citizenship_Status"], missing_codes={7,9})
    if "Years_in_US" in df:
        df["Years_in_US"] = clean_categorical(df["Years_in_US"], missing_codes={77,99})
    if "Education_Level" in df:
        df["Education_Level"] = clean_categorical(df["Education_Level"], missing_codes={7,9})
    if "Marital_Status" in df:
        df["Marital_Status"] = clean_categorical(df["Marital_Status"], missing_codes={77,99})
    if "Diabetes_Diagnosed" in df:
        df["Diabetes_Diagnosed"] = clean_categorical(df["Diabetes_Diagnosed"], missing_codes={7,9})
    if "Prediabetes_Diagnosed" in df:
        df["Prediabetes_Diagnosed"] = clean_categorical(df["Prediabetes_Diagnosed"], missing_codes={7,9})
    if "Takes_Insulin" in df:
        df["Takes_Insulin"] = clean_categorical(df["Takes_Insulin"], missing_codes={7,9})
    if "Takes_Diabetes_Pills" in df:
        df["Takes_Diabetes_Pills"] = clean_categorical(df["Takes_Diabetes_Pills"], missing_codes={7,9})
    if "A1C_Checked" in df:
        df["A1C_Checked"] = clean_categorical(df["A1C_Checked"], missing_codes={7,9})
    if "Target_A1C_Value" in df:
        df["Target_A1C_Value"] = clean_categorical(df["Target_A1C_Value"], missing_codes={77,99})
    if "Eye_Exam_Last_Year" in df:
        df["Eye_Exam_Last_Year"] = clean_categorical(df["Eye_Exam_Last_Year"], missing_codes={7,9})
    if "Vision_Affected_by_Diabetes" in df:
        df["Vision_Affected_by_Diabetes"] = clean_categorical(df["Vision_Affected_by_Diabetes"], missing_codes={7,9})

    
    # numerical
    if "Age" in df:
        df["Age"] = clean_numeric(df["Age"])
    if "Family_Size" in df:
        df["Family_Size"] = clean_numeric(df["Family_Size"], missing_codes={"."})
    if "Poverty_Income_Ratio" in df:
        df["Poverty_Income_Ratio"] = clean_numeric(df["Poverty_Income_Ratio"], missing_codes={77,99})
    if "Age_at_Diagnosis" in df:
        df["Age_at_Diagnosis"] = clean_numeric(df["Age_at_Diagnosis"], missing_codes={777,999})
    if "Doctor_Visits_Last_Year" in df:
        df["Doctor_Visits_Last_Year"] = clean_numeric(df["Doctor_Visits_Last_Year"], missing_codes={7777,9999})
    if "Last_A1C_Value" in df:
        df["Last_A1C_Value"] = clean_numeric(df["Last_A1C_Value"], missing_codes={777,999})
    return df


# investigating columns with more thatn 30% missing data
def drop_high_missing(df, threshold, must_keep):
    # save report before dropping
    missing_report = df.isna().mean().sort_values(ascending=False)
    missing_report.to_csv(reports_path / "missingness_report.csv")

    # only drop if not in must_keep
    drop_cols = [
        col for col in df.columns
        if col not in must_keep
        and missing_report[col] > threshold
    ]

    kept_over = [col for col in df.columns
                 if col in must_keep and missing_report[col] > threshold]

    if drop_cols:
        df = df.drop(columns=drop_cols)
    
    # print(f"Dropped columns: {drop_cols if drop_cols else 'None'}")
    # if kept_over:
    #     print(f"Kept (over threshold): {kept_over}")

    return df



# applying mapping for unusal or special rows
def apply_code_mappings(df: pd.DataFrame) -> pd.DataFrame:
    # Columns I do map
    categorical_maps = {
        "Race_Ethnicity": mappings.race_map,
        "Gender": mappings.gender_map,
        "Takes_Insulin": mappings.insulin_map,
        "Country_of_Birth": mappings.birth_country_map,
        "Diabetes_Diagnosed": mappings.diabetes_diagnosis_map,
        "Citizenship_Status": mappings.citizenship_map,
        "Prediabetes_Diagnosed": mappings.prediabetes_diagnosed_map,
        "Marital_Status": mappings.marital_status_map,
        "Education_Level": mappings.education_level_map,
    }

    # apply mappings 
    for col, m in categorical_maps.items():
        if col in df.columns:
            orig = df[col]
            df[col] = orig.map(m)  

            # quick print check for any codes that didn’t map
            leftover = sorted(set(orig.dropna().unique()) - set(m.keys()))
            if leftover:
                print(f"[mapping] {col}: unmapped codes found - {leftover}")
        else:
            print(f"[mapping]: column not found - {col}")

    # Numeric keeps: don’t map these
    # ID, Age, Poverty_Income_Ratio, Family_Size stay numeric
    # Add top-coded flag for age 80 = 80+
    if "Age" in df.columns:
        df["is_80_plus"] = df["Age"].eq(80)
    if "Family_Size" in df.columns:
        df["7_or_more"] = df["Family_Size"].eq(7)
    if "Poverty_Income_Ratio" in df.columns:
        df["pir_5_plus"] = df["Poverty_Income_Ratio"].eq(5)

    return df


# grouping funcitions together
def build_processed(raw_dir: str) -> pd.DataFrame:
    df = load_merge_raw(raw_dir)
    df = restrict_columns(df, keep_cols) 
    df = rename_columns(df)
    df = clean_columns(df)
    df = drop_high_missing(df, threshold=threshold, must_keep=must_keep)
    df = apply_code_mappings(df)
    return df


if __name__ == "__main__":
    raw_data_dir = "data/raw"
    out_path = "data/cleaned_dataset.csv"

    df_clean = build_processed(raw_data_dir)

    # Toggle save (flip to True when you want a file)
    save_output = False
    if save_output:
        df_clean.to_csv(out_path, index=False)
        print(f"Saved cleaned data to {out_path}")

    print(f"Rows: {df_clean.shape[0]:,} | Cols: {df_clean.shape[1]:,}")
    


