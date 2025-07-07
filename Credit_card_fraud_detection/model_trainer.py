import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.utils import resample

def train_model(X_train, y_train):
    print("Balancing and training model...")

    # Combine X and y
    df_train = X_train.copy()
    df_train['Class'] = y_train

    # Handle class imbalance
    df_majority = df_train[df_train.Class == 0]
    df_minority = df_train[df_train.Class == 1]

    df_minority_upsampled = resample(
        df_minority,
        replace=True,
        n_samples=len(df_majority),
        random_state=42
    )

    df_balanced = pd.concat([df_majority, df_minority_upsampled])
    X_bal = df_balanced.drop('Class', axis=1)
    y_bal = df_balanced['Class']

    # Train logistic regression
    model = LogisticRegression(max_iter=1000)
    model.fit(X_bal, y_bal)

    print("Model training completed.")
    return model
