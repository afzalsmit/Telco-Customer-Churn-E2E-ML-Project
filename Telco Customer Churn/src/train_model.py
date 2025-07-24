import pandas as pd

# Load the dataset
df = pd.read_csv("WA_Fn-UseC_-Telco-Customer-Churn.csv")

# 🔍 1. Convert TotalCharges to numeric (handle blanks)
df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')  # blanks will become NaN

# 🔍 2. Drop rows where TotalCharges is NaN
df = df.dropna(subset=['TotalCharges'])

# 🔍 3. Drop customerID (not useful for training)
df.drop('customerID', axis=1, inplace=True)

# 🔍 4. Encode target column 'Churn' → 1 for Yes, 0 for No
df['Churn'] = df['Churn'].map({'Yes': 1, 'No': 0})

# 🔍 5. Encode categorical variables (except already numeric like tenure, MonthlyCharges, etc.)
df = pd.get_dummies(df, drop_first=True)

# ✅ Final clean dataset ready for training
print("Final shape:", df.shape)
print("First few columns:", df.columns[:10].tolist())
