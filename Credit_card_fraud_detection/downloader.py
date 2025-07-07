import kagglehub
import pandas as pd
import os

def download_and_load():
    #  Download dataset using kagglehub
    path = kagglehub.dataset_download("mlg-ulb/creditcardfraud")
    print(" Dataset downloaded to:", path)

    #  Locate the CSV
    csv_path = os.path.join(path, "creditcard.csv")

    try:
        df = pd.read_csv(csv_path)
        print("\n First 5 rows of data:")
        print(df.head())
        return df
    except FileNotFoundError:
        print(" Error: creditcard.csv not found in:", path)
        return None
