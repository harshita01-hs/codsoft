from sklearn.ensemble import RandomForestClassifier

def train_model(X_train, y_train):
    print(" Training RandomForest model...")

    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    print(" Model training completed.")
    return model
