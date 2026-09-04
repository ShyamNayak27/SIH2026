import os
import sys
import json
import joblib
import pandas as pd

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from xgboost import XGBClassifier

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score
)


# ==========================================
# IMPORT PROJECT MODULES
# ==========================================

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(CURRENT_DIR)

from data_loader import load_dataset
from config import MODEL_FEATURES


# ==========================================
# PATH CONFIGURATION
# ==========================================

PROJECT_ROOT = os.path.abspath(
    os.path.join(CURRENT_DIR, "../..")
)

DATA_PATH = os.path.join(
    PROJECT_ROOT,
    "data",
    "processed",
    "ner_landslide_v1.csv"
)

MODEL_DIR = os.path.join(
    PROJECT_ROOT,
    "models"
)

os.makedirs(MODEL_DIR, exist_ok=True)


# ==========================================
# LOAD AND SPLIT DATA
# ==========================================

print("\nLoading dataset...")

df = load_dataset(DATA_PATH)

train_df = df[df["split"] == "train"]
val_df = df[df["split"] == "val"]
test_df = df[df["split"] == "test"]


X_train = train_df[MODEL_FEATURES]
y_train = train_df["label"]

X_val = val_df[MODEL_FEATURES]
y_val = val_df["label"]

X_test = test_df[MODEL_FEATURES]
y_test = test_df["label"]


print(f"Train samples: {len(X_train)}")
print(f"Validation samples: {len(X_val)}")
print(f"Test samples: {len(X_test)}")


# ==========================================
# EVALUATION FUNCTION
# ==========================================

def evaluate_model(model, X, y):

    predictions = model.predict(X)

    probabilities = model.predict_proba(X)[:, 1]

    results = {

        "accuracy": accuracy_score(
            y,
            predictions
        ),

        "precision": precision_score(
            y,
            predictions,
            zero_division=0
        ),

        "recall": recall_score(
            y,
            predictions,
            zero_division=0
        ),

        "f1": f1_score(
            y,
            predictions,
            zero_division=0
        ),

        "roc_auc": roc_auc_score(
            y,
            probabilities
        )
    }

    return results


# ==========================================
# DEFINE MODELS
# ==========================================

models = {

    # --------------------------------------
    # LOGISTIC REGRESSION
    # --------------------------------------

    "Logistic Regression":

        Pipeline([
            (
                "scaler",
                StandardScaler()
            ),
            (
                "model",
                LogisticRegression(
                    max_iter=2000,
                    class_weight="balanced",
                    random_state=42
                )
            )
        ]),


    # --------------------------------------
    # RANDOM FOREST
    # --------------------------------------

    "Random Forest":

        RandomForestClassifier(
            n_estimators=300,
            max_depth=12,
            min_samples_split=5,
            min_samples_leaf=2,
            class_weight="balanced",
            random_state=42,
            n_jobs=-1
        ),


    # --------------------------------------
    # XGBOOST
    # --------------------------------------

    "XGBoost":

        XGBClassifier(
            n_estimators=300,
            max_depth=4,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,

            objective="binary:logistic",
            eval_metric="logloss",

            random_state=42,
            n_jobs=-1
        )
}


# ==========================================
# TRAIN + VALIDATE
# ==========================================

all_results = {}

best_model = None
best_model_name = None
best_score = -1


for name, model in models.items():

    print("\n" + "=" * 50)
    print(f"Training {name}")

    model.fit(
        X_train,
        y_train
    )


    results = evaluate_model(
        model,
        X_val,
        y_val
    )


    all_results[name] = results


    print("\nValidation Results:")

    for metric, value in results.items():

        print(
            f"{metric}: {value:.4f}"
        )


    # --------------------------------------
    # MODEL SELECTION
    # --------------------------------------
    # Primary metric = ROC-AUC
    # because this is an imbalanced
    # risk prediction problem.

    if results["roc_auc"] > best_score:

        best_score = results["roc_auc"]

        best_model = model

        best_model_name = name


# ==========================================
# BEST MODEL
# ==========================================

print("\n" + "=" * 50)

print("BEST MODEL")

print(f"Model: {best_model_name}")

print(f"Validation ROC-AUC: {best_score:.4f}")


# ==========================================
# FINAL TEST EVALUATION
# ==========================================

print("\nEvaluating best model on test data...")


test_results = evaluate_model(
    best_model,
    X_test,
    y_test
)


print("\nTEST RESULTS:")

for metric, value in test_results.items():

    print(
        f"{metric}: {value:.4f}"
    )


# ==========================================
# SAVE FINAL MODEL
# ==========================================

MODEL_PATH = os.path.join(
    MODEL_DIR,
    "landslide_model.pkl"
)


joblib.dump(
    best_model,
    MODEL_PATH
)


# ==========================================
# SAVE MODEL METADATA
# ==========================================

metadata = {

    "model_name": best_model_name,

    "features": MODEL_FEATURES,

    "validation_results": all_results,

    "test_results": test_results
}


METADATA_PATH = os.path.join(
    MODEL_DIR,
    "model_metadata.json"
)


with open(
    METADATA_PATH,
    "w"
) as f:

    json.dump(
        metadata,
        f,
        indent=4
    )


# ==========================================
# COMPLETE
# ==========================================

print("\nModel saved to:")
print(MODEL_PATH)


print("\nMetadata saved to:")
print(METADATA_PATH)


print("\nTraining pipeline completed successfully!")