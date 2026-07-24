import pandas as pd
import os

def load_data(filepath):
    return pd.read_csv(filepath)

def preprocess_data(df):
    return df

def save_data(df, filepath):
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    df.to_csv(filepath, index=False)

if __name__ == "__main__":
    raw_data_path = os.path.join("..", "dataset_raw", "breast_cancer.csv")
    processed_data_path = os.path.join("dataset_preprocessing", "breast_cancer_processed.csv")
    
    print(f"Loading data from {raw_data_path}...")
    df = load_data(raw_data_path)
    
    print("Preprocessing data...")
    df_processed = preprocess_data(df)
    
    print(f"Saving processed data to {processed_data_path}...")
    save_data(df_processed, processed_data_path)
    
    print("Preprocessing completed successfully!")
