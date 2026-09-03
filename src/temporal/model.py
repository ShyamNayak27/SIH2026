from pathlib import Path

import joblib
import pandas as pd

from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    precision_recall_fscore_support,
)

from .features import add_temporal_features


DATA_PATH = Path("data/processed/ner_landslide_v1.csv")
MODEL_PATH = Path("models/temporal_risk_model.joblib")


TEMPORAL_FEATURES = [
    "log_rain_1d",
    "log_rain_3d",
    "log_rain_7d",
    "log_api",
    "rain_3d_ratio_30d",
    "rainfall_acceleration",
]


def get_features(df):

    df = add_temporal_features(df)

    return df[TEMPORAL_FEATURES]


def train_model():

    df = pd.read_csv(DATA_PATH)

    train = df[df["split"] == "train"].copy()
    val = df[df["split"] == "val"].copy()
    test = df[df["split"] == "test"].copy()

    X_train = get_features(train)
    X_val = get_features(val)
    X_test = get_features(test)

    y_train = train["label"]
    y_val = val["label"]
    y_test = test["label"]

    # ---------------------------------------------------------
    # Regularized logistic regression
    # ---------------------------------------------------------

    model = Pipeline([
        ("scaler", StandardScaler()),

        ("classifier", LogisticRegression(
            C=0.25,
            max_iter=1000,
            class_weight="balanced",
            random_state=42,
        )),
    ])

    model.fit(X_train, y_train)

    # ---------------------------------------------------------
    # Evaluation
    # ---------------------------------------------------------

    def evaluate(X, y, name):

        probabilities = model.predict_proba(X)[:, 1]
        predictions = (probabilities >= 0.5).astype(int)

        auc = roc_auc_score(y, probabilities)
        ap = average_precision_score(y, probabilities)

        precision, recall, f1, _ = (
            precision_recall_fscore_support(
                y,
                predictions,
                average="binary",
                zero_division=0,
            )
        )

        print(f"\n{name}")
        print("=" * 40)
        print(f"ROC-AUC   : {auc:.4f}")
        print(f"PR-AUC    : {ap:.4f}")
        print(f"Precision : {precision:.4f}")
        print(f"Recall    : {recall:.4f}")
        print(f"F1        : {f1:.4f}")

        return probabilities

    evaluate(X_train, y_train, "TRAIN")
    evaluate(X_val, y_val, "VALIDATION")
    evaluate(X_test, y_test, "TEST")

    # ---------------------------------------------------------
    # Feature coefficients
    # ---------------------------------------------------------

    classifier = model.named_steps["classifier"]

    print("\nFEATURE COEFFICIENTS")
    print("=" * 40)

    for feature, coefficient in zip(
        TEMPORAL_FEATURES,
        classifier.coef_[0],
    ):
        print(f"{feature:30s} {coefficient:+.4f}")

    # ---------------------------------------------------------
    # Save
    # ---------------------------------------------------------

    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)

    artifact = {
        "model": model,
        "feature_names": TEMPORAL_FEATURES,
    }

    joblib.dump(artifact, MODEL_PATH)

    print(f"\nModel saved to: {MODEL_PATH}")


if __name__ == "__main__":
    train_model()