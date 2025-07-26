import pandas as pd

# loading and inspecting demographics data
demo_df = pd.read_sas('DEMO_J.xpt', format='xport')


# loading and inspecting diabetes data
diq_df = pd.read_sas("DIQ_J.xpt", format='xport')


df = pd.merge(demo_df, diq_df, on="SEQN", how="inner")
df.shape
# df.info()
# df.isna().sum().sort_values(ascending=False)

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


# merging both lists cleanly (and avoiding duplication of SEQN)
columns_to_keep = list(set(diabetes_keep + demographics_keep))

# now making the dataframe only relevant columns (columns_to_keep)
df = df[columns_to_keep]

# sorting columns to keep diabetes and demographics grouped separately
df = df[demographics_keep + diabetes_keep[1:]]


# renaming columns for clarity
rename_dict = {
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


# renaming the columns
df.rename(columns=rename_dict, inplace=True)
df.columns