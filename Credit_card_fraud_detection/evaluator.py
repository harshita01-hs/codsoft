from sklearn.metrics import classification_report, confusion_matrix

def evaluate(model, X_test, y_test):
    y_pred = model.predict(X_test)

    print("\n Confusion Matrix:")
    print(confusion_matrix(y_test, y_pred))

    print("\n Classification Report:")
    print(classification_report(y_test, y_pred))
