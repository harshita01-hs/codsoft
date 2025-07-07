from downloader import download_and_load
from preprocessor import preprocess
from model_trainer import train_model
from evaluator import evaluate

# Step 1: Load dataset
df = download_and_load()

# Safety check
if df is None:
    print(" Exiting: Dataset not found or failed to load.")
    exit()

# Step 2: Preprocess
X_train, X_test, y_train, y_test = preprocess(df)

# Step 3: Train model
model = train_model(X_train, y_train)

# Step 4: Evaluate model
evaluate(model, X_test, y_test)
