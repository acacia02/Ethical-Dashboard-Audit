import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load cleaned data
df = pd.read_csv("data/processed/cleaned_dataset.csv")

# # Basic sanity checks
# print(df.shape)           # Rows, Cols
# print(df.info())          # Dtypes + non-null counts
# print(df.head(10))        # Peek at first 10 rows
# print(df.isna().sum().sort_values(ascending=False))  # NA counts

# # Quick categorical check
# for col in df.select_dtypes(include=['object', 'category']).columns:
#     print(f"\n{col} value counts:")
#     print(df[col].value_counts(dropna=False))


# Histograms for numeric columns
numeric_cols = df.select_dtypes(include=['float64', 'int64']).columns
df[numeric_cols].hist(bins=20, figsize=(15, 8))
plt.suptitle("Numeric Distributions", fontsize=16)
plt.tight_layout()
plt.show()

# bar charts for categorical columns
categorical_cols = df.select_dtypes(include=['object', 'category', 'bool']).columns
for col in categorical_cols:
    plt.figure(figsize=(8, 4))
    sns.countplot(data=df, x=col, order=df[col].value_counts().index)
    plt.xticks(rotation=45)
    plt.title(f"{col} Distribution")
    plt.show()

# correlation heatmap for numeric columns
plt.figure(figsize=(10, 8))
sns.heatmap(df[numeric_cols].corr(), annot=True, cmap='coolwarm', fmt=".2f")
plt.title("Numeric Feature Correlation")
plt.show()