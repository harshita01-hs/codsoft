from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler

def preprocess(df):
    X = df.drop(columns=["Id", "Species"])  # Remove ID and target
    y = df["Species"]

    # Encode the target labels
    encoder = LabelEncoder()
    y_encoded = encoder.fit_transform(y)

    # Feature scaling (optional for this dataset)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
    )

    return X_train, X_test, y_train, y_test, encoder
