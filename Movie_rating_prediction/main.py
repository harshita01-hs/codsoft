from downloader import download_and_load
from preprocessor import preprocess
from model_trainer import train_model
from evaluator import evaluate

# Step 1: Download and load
df = download_and_load()
if df is None:
    exit(" Exiting: Dataset not loaded.")

# Step 2: Preprocess
X_train, X_test, y_train, y_test = preprocess(df)

# Step 3: Train
model = train_model(X_train, y_train)

# Step 4: Evaluate
evaluate(model, X_test, y_test)
