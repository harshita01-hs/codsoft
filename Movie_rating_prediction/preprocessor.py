import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
import ast  # safer than eval

def preprocess(df):
    print(" Preprocessing data...")

    # Step 1: Select relevant columns and drop missing
    required_columns = ['genres', 'runtime', 'popularity', 'budget', 'revenue', 'vote_average']
    if not all(col in df.columns for col in required_columns):
        print(" Required columns not found in dataset.")
        return None

    df = df[required_columns].dropna()

    # Step 2: Safely extract first genre
    def extract_genre(genre_str):
        try:
            genres_list = ast.literal_eval(genre_str)
            return genres_list[0]['name'] if genres_list else 'Unknown'
        except:
            return 'Unknown'

    df['genre'] = df['genres'].apply(extract_genre)

    # Step 3: One-hot encode genre
    encoder = OneHotEncoder(sparse_output=False)
    genre_encoded = encoder.fit_transform(df[['genre']])
    genre_df = pd.DataFrame(genre_encoded, columns=encoder.get_feature_names_out(['genre']))
    genre_df.reset_index(drop=True, inplace=True)

    # Step 4: Prepare features and target
    numeric_features = df[['runtime', 'popularity', 'budget', 'revenue']].reset_index(drop=True)
    features = pd.concat([numeric_features, genre_df], axis=1)
    target = df['vote_average']

    # Step 5: Normalize numeric features
    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(features)

    # Step 6: Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        features_scaled, target, test_size=0.2, random_state=42
    )

    print(" Preprocessing complete. Feature shape:", features.shape)
    return X_train, X_test, y_train, y_test
