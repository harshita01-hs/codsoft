from sklearn.metrics import classification_report, accuracy_score

def evaluate(model, X_test, y_test, encoder):
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)

    print(f"\n Accuracy: {accuracy:.2f}")
    print("\n Classification Report:")
    print(classification_report(y_test, y_pred, target_names=encoder.classes_))
