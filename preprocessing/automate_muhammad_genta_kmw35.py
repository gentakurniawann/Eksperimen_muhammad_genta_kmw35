import pandas as pd
from sklearn.datasets import load_breast_cancer
import os

def load_data():
    print("Loading Breast Cancer dataset...")
    data = load_breast_cancer()
    df = pd.DataFrame(data.data, columns=data.feature_names)
    df['target'] = data.target
    return df

def preprocess_data(df):
    print("Preprocessing data...")
    # Drop rows with missing values (if any)
    df_cleaned = df.dropna()
    return df_cleaned

def save_data(df, output_path):
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_csv(output_path, index=False)
    print(f"Data saved to {output_path}")

if __name__ == "__main__":
    raw_df = load_data()
    processed_df = preprocess_data(raw_df)
    save_data(processed_df, "../breast_cancer_preprocessing/breast_cancer_processed.csv")
    print("Automation pipeline completed successfully!")
