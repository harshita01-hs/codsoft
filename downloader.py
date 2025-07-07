import kagglehub
import pandas as pd
import os

def download_and_load():
    # Download from Kaggle using kagglehub
    path = kagglehub.dataset_download("uciml/iris")
    print(" Dataset downloaded to:", path)

    csv_path = os.path.join(path, "Iris.csv")  # Kaggle file name is "Iris.csv"

    try:
        df = pd.read_csv(csv_path)
        print("\n First 5 rows:")
        print(df.head())
        return df
    except FileNotFoundError:
        print(" File not found in:", path)
        return None
