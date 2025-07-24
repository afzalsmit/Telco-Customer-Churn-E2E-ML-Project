import pandas as pd
from sklearn.preprocessing import LabelEncoder
import os

def load_data(file_path):
    df = pd.read_csv(file_path)
    return df

def clean_data(df):
    # Drop customerID, handle missing TotalCharges
    df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
    df = df.dropna()
    df.drop(['customerID'], axis=1, inplace=True)
    return df

def encode_features(df):
    # Binary target variable
    df['Churn'] = df['Churn'].map({'Yes': 1, 'No': 0})
    
    # Encode categorical variables
    categorical_cols = df.select_dtypes(include='object').columns
    df = pd.get_dummies(df, columns=categorical_cols, drop_first=True)
    
    return df

def save_processed_data(df, output_path):
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_csv(output_path, index=False)

if __name__ == "__main__":
    raw_path = "data/raw/WA_Fn-UseC_-Telco-Customer-Churn.csv"
    processed_path = "data/processed/processed_churn.csv"

    df = load_data(raw_path)
    df = clean_data(df)
    df = encode_features(df)
    save_processed_data(df, processed_path)
    print(f"✅ Data preprocessing complete. Saved to: {processed_path}")
