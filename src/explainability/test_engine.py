import os
import sys

# Add the current explainability directory to Python path
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(CURRENT_DIR)

from data_loader import load_dataset, get_train_test_data


# Project root = explainability -> src -> project root
PROJECT_ROOT = os.path.abspath(
    os.path.join(CURRENT_DIR, "../..")
)

DATA_PATH = os.path.join(
    PROJECT_ROOT,
    "data",
    "processed",
    "ner_landslide_v1.csv"
)


print("Loading dataset from:")
print(DATA_PATH)


df = load_dataset(DATA_PATH)

print("\nDataset shape:")
print(df.shape)


X_train, X_test, y_train, y_test = get_train_test_data(df)


print("\nTrain shape:")
print(X_train.shape)

print("\nTest shape:")
print(X_test.shape)


print("\nFeatures:")
for feature in X_train.columns:
    print("-", feature)


print("\nLabel distribution:")
print(df["label"].value_counts())


print("\nDataset loaded successfully!")