from downloader import download_and_load
from preprocessor import preprocess
from model_trainer import train_model
from evaluator import evaluate

# Step 1: Download and load dataset
df = download_and_load()

if df is None:
    print(" Exiting: Dataset not found.")
    exit()

# Step 2: Preprocess
X_train, X_test, y_train, y_test, encoder = preprocess(df)

# Step 3: Train model
model = train_model(X_train, y_train)

# Step 4: Evaluate
evaluate(model, X_test, y_test, encoder)
