Ethical Audit Dashboard – NHANES Diabetes Data

This project is part of a personal initiative to explore fairness and subgroup disparities in diabetes diagnosis and care using U.S. NHANES data.
The goal is to build a dashboard that makes it easy to see how race, gender, income, education, and other social factors might influence who gets diagnosed, treated, or monitored for diabetes.

Features:
- Model Comparison – Logistic vs. XGBoost with calibration and performance metrics
- Fairness Slices – Precision, recall, and false negative rates across age and education subgroups
- SQL Explorer – Run simple SQL queries on NHANES subsets directly in the app
- Data Preview – Quick view of the cleaned NHANES dataset used for modeling
- Insights Panel – Automatically generated summary of key fairness findings


About the Data:
This dashboard uses data derived from the NHANES 2017–2018 cycle.
The processed file is included in the repo for demo purposes so the dashboard runs smoothly on Streamlit Cloud.
In a production setting, this dataset would normally be stored externally to keep the repo lightweight.

Tech Stack:
- Python (pandas, scikit-learn, xgboost, matplotlib, seaborn)
- Streamlit (interactive dashboard + SQL explorer)
- SQL (lite queries and database creation)
- GitHub + Streamlit Cloud (deployment)

Limitations / Next Steps:
- Some fairness slices are based on small subgroups (unstable metrics)
- Education levels are currently coded numerically for modeling purposes
- Future improvement: external data hosting (cloud storage or HuggingFace Datasets) for cleaner repo size

LIVE APP: https://ethical-dashboard-audit-5gxmqnfgpfyvz8legetjcx.streamlit.app/
GITHUB REPO: You’re already here!
