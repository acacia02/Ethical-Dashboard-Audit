import streamlit as st
import pandas as pd
import json
from pathlib import Path

st.set_page_config(page_title="Ethical Audit Dashboard", layout="wide")
st.title("Ethical Audit Dashboard")
st.caption("Evaluating diabetes risk models and fairness on NHANES")

# Paths
P = Path("reports/app")
CMP_CSV = P / "model_comparison.csv"

LR_JSON  = P / "logistic_calibrated.json"
XGB_JSON = P / "xgb_uncalibrated.json"

LR_PR   = P / "lr_pr_curve.png"
XGB_PR  = P / "xgb_pr_curve.png"
LR_ROC  = P / "lr_roc_curve.png"
XGB_ROC = P / "xgb_roc_curve.png"

LR_ODDS = P / "lr_feature_odds.csv"
XGB_FI  = P / "xgb_feature_importance.csv"

FAIR = {
    ("Logistic","AgeGroup"):      P / "fairness_lr_age.csv",
    ("Logistic","Education_Level"):P / "fairness_lr_education.csv",
    ("XGBoost","AgeGroup"):       P / "fairness_xgb_age.csv",
    ("XGBoost","Education_Level"):P / "fairness_xgb_education.csv",
}

# helpers
@st.cache_data
def load_df(path): return pd.read_csv(path)

def load_json(path):
    with open(path) as f:
        return json.load(f)

def metric_block(col, title, d):
    col.subheader(title)
    col.metric("ROC AUC", round(d["roc_auc"],3))
    col.metric("PR AUC",  round(d["pr_auc"],3))
    thr = d.get("chosen_operating_threshold", d.get("threshold"))
    col.metric("Threshold", round(thr,3))
    col.metric("Precision@thr", round(d["precision_tuned"],3))
    col.metric("Recall@thr",    round(d["recall_tuned"],3))
    col.metric("F1@thr",        round(d["f1_tuned"],3))

# Layout
tab_overview, tab_perf, tab_fair, tab_features, tab_data = st.tabs(
    ["Overview", "Performance", "Fairness", "Features", "Data Preview"]
)

with tab_overview:
    st.subheader("Model comparison")
    if CMP_CSV.exists():
        cmp = load_df(CMP_CSV)
        st.dataframe(cmp, use_container_width=True)
    else:
        st.info("model_comparison.csv not found under reports/app/")

    c1, c2 = st.columns(2)
    try:
        metric_block(c1, "Logistic (calibrated)", load_json(LR_JSON))
    except Exception:
        c1.warning("Missing logistic JSON")
    try:
        metric_block(c2, "XGBoost", load_json(XGB_JSON))
    except Exception:
        c2.warning("Missing XGBoost JSON")

with tab_perf:
    st.subheader("ROC & PR Curves")
    c1, c2 = st.columns(2)
    if LR_PR.exists(): c1.image(str(LR_PR), caption="Logistic PR", use_container_width=True)
    if XGB_PR.exists(): c2.image(str(XGB_PR), caption="XGBoost PR", use_container_width=True)
    c3, c4 = st.columns(2)
    if LR_ROC.exists(): c3.image(str(LR_ROC), caption="Logistic ROC", use_container_width=True)
    if XGB_ROC.exists(): c4.image(str(XGB_ROC), caption="XGBoost ROC", use_container_width=True)

with tab_fair:
    st.subheader("Fairness slices (precision / recall / FNR by group)")
    model = st.selectbox("Model", ["Logistic","XGBoost"])
    group = st.selectbox("Group", ["AgeGroup","Education_Level"])
    path = FAIR.get((model, group))
    if path and path.exists():
        st.dataframe(load_df(path).round(3), use_container_width=True)
    else:
        st.info("Fairness CSV not found. Export it to reports/app/ and refresh.")

with tab_features:
    c1, c2 = st.columns(2)
    with c1:
        st.subheader("Logistic — Odds ratios")
        if LR_ODDS.exists():
            st.dataframe(load_df(LR_ODDS).round(3).head(25), use_container_width=True)
        else:
            st.info("lr_feature_odds.csv not found.")
    with c2:
        st.subheader("XGBoost — Feature importance")
        if XGB_FI.exists():
            st.dataframe(load_df(XGB_FI).round(6).head(25), use_container_width=True)
        else:
            st.info("xgb_feature_importance.csv not found.")

with tab_data:
    st.subheader("Quick preview of inputs")
    data_path = Path("data/processed/df_model_ready.csv")
    if data_path.exists():
        df = load_df(data_path)
        st.dataframe(df.head(), use_container_width=True)
    else:
        st.info("df_model_ready.csv not found.")


# verbal explanation
st.markdown("---")
st.header("Insights")

def safe_load_json(p):
    try:
        with open(p) as f:
            return json.load(f)
    except Exception:
        return None

def top_gaps(df, k=1, metric="recall"):
    """Return worst and best groups (+ gaps) for a metric."""
    if df is None or df.empty or metric not in df.columns:
        return None
    d = df.copy()
    d = d.dropna(subset=[metric])
    if d.empty:
        return None
    worst = d.sort_values(metric, ascending=True).head(k)
    best  = d.sort_values(metric, ascending=False).head(k)
    rng   = d[metric].max() - d[metric].min()
    return {
        "worst": worst[["group", metric, "n"]].values.tolist(),
        "best":  best[["group", metric, "n"]].values.tolist(),
        "range": float(rng)
    }

def try_read_csv(p):
    try:
        return pd.read_csv(p)
    except Exception:
        return None

# Load tuned-threshold summaries
lr_json  = safe_load_json(LR_JSON)
xgb_json = safe_load_json(XGB_JSON)

# Load fairness slices
fair_tables = {
    "Logistic / AgeGroup":        try_read_csv(FAIR.get(("Logistic", "AgeGroup"))),
    "Logistic / Education_Level": try_read_csv(FAIR.get(("Logistic", "Education_Level"))),
    "XGBoost / AgeGroup":         try_read_csv(FAIR.get(("XGBoost", "AgeGroup"))),
    "XGBoost / Education_Level":  try_read_csv(FAIR.get(("XGBoost", "Education_Level")))
}

# Build insights
insights = []

# overall trade-off at tuned thresholds
if lr_json and xgb_json:
    lr_p, lr_r, lr_f1 = lr_json["precision_tuned"], lr_json["recall_tuned"], lr_json["f1_tuned"]
    xg_p, xg_r, xg_f1 = xgb_json["precision_tuned"], xgb_json["recall_tuned"], xgb_json["f1_tuned"]
    if lr_r > xg_r:
        insights.append(f"**Logistic** has higher recall at its chosen threshold ({lr_r:.2f} vs {xg_r:.2f}), "
                        f"but precision is {lr_p:.2f} vs XGBoost’s {xg_p:.2f}.")
    if xg_r > lr_r:
        insights.append(f"**XGBoost** has higher recall at its chosen threshold ({xg_r:.2f} vs {lr_r:.2f}), "
                        f"but precision is {xg_p:.2f} vs Logistic’s {lr_p:.2f}.")
    else:
        insights.append(f"**XGBoost** and **Logistic** roughly have the same recall at their chosen thresholds ({xg_r:.2f} vs {lr_r:.2f}), "
                        f"but precision is {xg_p:.2f} vs Logistic’s {lr_p:.2f}.")
    if lr_f1 > xg_f1:
        insights.append(f"Overall **F1** is higher for Logistic ({lr_f1:.2f} vs {xg_f1:.2f}).")
    elif xg_f1 > lr_f1:
        insights.append(f"Overall **F1** is higher for XGBoost ({xg_f1:.2f} vs {lr_f1:.2f}).")
    else:
        insights.append("Overall **F1** is roughly tied between the models.")

# Fairness: biggest recall disparities by group
for label, df_ in fair_tables.items():
    g = top_gaps(df_, metric="recall")
    if not g:
        continue
    worst_txt = ", ".join([f"{grp} (recall {val:.2f}, n={int(n)})" for grp, val, n in g["worst"]])
    best_txt  = ", ".join([f"{grp} (recall {val:.2f}, n={int(n)})" for grp, val, n in g["best"]])
    insights.append(
        f"**{label}** — recall spread is **{g['range']:.2f}**. "
        f"Worst group(s): {worst_txt}. Best group(s): {best_txt}."
    )

# Flag groups with very low support
low_support_flags = []
for label, df_ in fair_tables.items():
    if df_ is None or "n" not in df_.columns: 
        continue
    few = df_.sort_values("n").head(1)
    for _, row in few.iterrows():
        low_support_flags.append(f"{label}: {row['group']} (n={int(row['n'])})")
if low_support_flags:
    insights.append("Some groups have very small sample sizes (unstable metrics): " + "; ".join(low_support_flags))

# Render
if insights:
    for line in insights:
        st.markdown(f"- {line}")
else:
    st.info("No insights yet — make sure fairness CSVs and model JSONs are present in `reports/app/`.")