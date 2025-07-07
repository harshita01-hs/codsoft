from sklearn.linear_model import LinearRegression

def train_model(X_train, y_train):
    print(" Training Linear Regression model...")

    model = LinearRegression()
    model.fit(X_train, y_train)

    print(" Model training complete.")
    return model
