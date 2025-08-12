import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import chi2_contingency
from math import sqrt

# Cramer's V for categorical correlation 
def cramers_v(x, y):
    confusion_matrix = pd.crosstab(x, y)
    chi2 = chi2_contingency(confusion_matrix)[0]
    n = confusion_matrix.sum().sum()
    phi2 = chi2 / n
    r, k = confusion_matrix.shape
    phi2corr = max(0, phi2 - ((k-1)*(r-1))/(n-1))
    rcorr = r - ((r-1)**2)/(n-1)
    kcorr = k - ((k-1)**2)/(n-1)
    return sqrt(phi2corr / min((kcorr-1), (rcorr-1)))


# running EDA
def run_eda(input_path="data/processed/cleaned_dataset.csv", output_dir="eda_outputs/plots"):
    # Create output dir
    os.makedirs(output_dir, exist_ok=True)

    # Read data
    df = pd.read_csv(input_path)
    print(f"Loaded dataset: {df.shape[0]:,} rows, {df.shape[1]:,} cols")

    # missing values ummary 
    missing_summary = df.isna().sum()
    missing_summary = missing_summary[missing_summary > 0].sort_values(ascending=False)
    if missing_summary.empty:
        print("No missing values found")
    else:
        print("\nMissing Values Summary:\n", missing_summary)
        missing_summary.to_csv(os.path.join(output_dir, "missing_values_summary.csv"))

    # Histograms for numeric variables
    numeric_cols = df.select_dtypes(include=["float", "int"]).columns
    for col in numeric_cols:
        plt.figure(figsize=(6, 4))
        sns.histplot(df[col].dropna(), kde=False, bins=30)
        plt.title(f"Distribution of {col}")
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f"hist_{col}.png"))
        plt.close()

    # count plots for categorical variables
    categorical_cols = df.select_dtypes(include=["object", "category"]).columns
    for col in categorical_cols:
        plt.figure(figsize=(6, 4))
        order = df[col].value_counts().index
        sns.countplot(y=df[col], order=order)
        plt.title(f"Count Plot: {col}")
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f"count_{col}.png"))
        plt.close()

    # numeric correlation Heatmap
    if len(numeric_cols) > 1:
        plt.figure(figsize=(10, 8))
        corr = df[numeric_cols].corr()
        sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm")
        plt.title("Numeric Feature Correlation")
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "numeric_correlation_heatmap.png"))
        plt.close()

    # categorical association heatmap (Cramer's V_
    if len(categorical_cols) > 1:
        cat_corr = pd.DataFrame(index=categorical_cols, columns=categorical_cols, dtype=float)
        for col1 in categorical_cols:
            for col2 in categorical_cols:
                if col1 == col2:
                    cat_corr.loc[col1, col2] = 1.0
                else:
                    cat_corr.loc[col1, col2] = cramers_v(df[col1], df[col2])
        plt.figure(figsize=(10, 8))
        sns.heatmap(cat_corr.astype(float), annot=True, fmt=".2f", cmap="coolwarm")
        plt.title("Categorical Feature Association (Cramer's V)")
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "categorical_association_heatmap.png"))
        plt.close()

    print(f"\nEDA completed. Plots saved. {output_dir}")


if __name__ == "__main__":
    run_eda()




