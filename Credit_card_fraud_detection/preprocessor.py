from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

def preprocess(df):
    X = df.drop('Class', axis=1)
    y = df['Class']

    # Normalize 'Amount' and 'Time'
    scaler = StandardScaler()
    X[['Amount', 'Time']] = scaler.fit_transform(X[['Amount', 'Time']])

    # Split into train/test sets
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )

    return X_train, X_test, y_train, y_test
