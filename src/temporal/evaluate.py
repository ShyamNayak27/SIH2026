from pathlib import Path

import joblib
import pandas as pd


DATA_PATH = Path("data/processed/ner_landslide_v1.csv")
MODEL_PATH = Path("models/temporal_risk_model.joblib")
OUTPUT_PATH = Path("data/processed/temporal_risk_predictions.csv")

# Selected using validation-set precision/recall trade-off.
# Frozen before evaluating the held-out test set.
ALERT_THRESHOLD = 47.0


def generate_predictions():

    df = pd.read_csv(DATA_PATH)

    artifact = joblib.load(MODEL_PATH)

    model = artifact["model"]
    feature_names = artifact["feature_names"]

    from .features import add_temporal_features

    features = add_temporal_features(df)
    X = features[feature_names]

    probabilities = model.predict_proba(X)[:, 1]

    output = df[
        [
            "sample_id",
            "lon",
            "lat",
            "state",
            "district",
            "event_date",
            "label",
            "split",
        ]
    ].copy()

    # ---------------------------------------------------------
    # Continuous temporal risk score
    # ---------------------------------------------------------

    output["temporal_risk"] = (
        probabilities * 100
    ).round(2)

    # ---------------------------------------------------------
    # Operational alert
    # ---------------------------------------------------------

    output["temporal_alert"] = (
        output["temporal_risk"] >= ALERT_THRESHOLD
    )

    output["alert_level"] = output[
        "temporal_alert"
    ].map({
        True: "ELEVATED",
        False: "NORMAL",
    })

    # ---------------------------------------------------------
    # Rainfall indicators
    # ---------------------------------------------------------

    output["rainfall_24h"] = df["rain_1d"]
    output["rainfall_72h"] = df["rain_3d"]
    output["rainfall_7d"] = df["rain_7d"]
    output["rainfall_30d"] = df["rain_30d"]
    output["api"] = df["api"]

    output["rainfall_acceleration"] = (
        (df["rain_3d"] / 3.0)
        / ((df["rain_30d"] / 30.0) + 1e-6)
    ).round(3)

    output["rainfall_concentration"] = (
        df["rain_3d"]
        / (df["rain_30d"] + 1e-6)
    ).round(3)

    # ---------------------------------------------------------
    # Save
    # ---------------------------------------------------------

    OUTPUT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    output.to_csv(
        OUTPUT_PATH,
        index=False,
    )

    print(f"Saved: {OUTPUT_PATH}")
    print(f"Rows: {len(output)}")
    print(f"Alert threshold: {ALERT_THRESHOLD}")

    print("\nOverall alert distribution:")
    print(output["alert_level"].value_counts())

    print("\nAlert distribution by split:")
    print(
        output.groupby(
            ["split", "alert_level"]
        ).size()
    )


if __name__ == "__main__":
    generate_predictions()