import os
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
import mlflow

os.environ['MLFLOW_TRACKING_URI'] = "http://localhost:5000"

def main():
    try:
        df = pd.read_csv('../Eksperimen_SML_muhammad_genta_kmw35/iris_preprocessing/iris_processed.csv')
    except FileNotFoundError:
        print("Data not found. Please run automate_muhammad_genta_kmw35.py first.")
        return
        
    X = df.drop(columns=['target'])
    y = df['target']
    
    mlflow.sklearn.autolog()
    
    with mlflow.start_run():
        model = RandomForestClassifier(n_estimators=10, random_state=42)
        model.fit(X, y)

if __name__ == "__main__":
    main()
