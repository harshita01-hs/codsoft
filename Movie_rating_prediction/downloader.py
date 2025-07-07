import kagglehub
import pandas as pd
import os

def download_and_load():
    # Download dataset using kagglehub
    path = kagglehub.dataset_download("tmdb/tmdb-movie-metadata")
    print(" Dataset downloaded to:", path)

    csv_path = os.path.join(path, "tmdb_5000_movies.csv")  # main file

    try:
        df = pd.read_csv(csv_path)
        print("\n First 5 rows of data:")
        print(df.head())
        return df
    except FileNotFoundError:
        print(" Error: tmdb_5000_movies.csv not found.")
        return None
