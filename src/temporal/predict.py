from pathlib import Path

import joblib
import pandas as pd

from .features import add_temporal_features


MODEL_PATH = Path("models/temporal_risk_model.joblib")


def load_model():
    artifact = joblib.load(MODEL_PATH)
    return artifact["model"], artifact["feature_names"]


def get_temporal_risk(row: pd.DataFrame) -> dict:

    model, feature_names = load_model()

    features = add_temporal_features(row)

    X = features[feature_names]

    probability = float(model.predict_proba(X)[0, 1])

    risk_score = round(probability * 100, 1)

    if risk_score >= 70:
        risk_level = "HIGH"
    elif risk_score >= 40:
        risk_level = "MODERATE"
    else:
        risk_level = "LOW"

    r1 = float(row["rain_1d"].iloc[0])
    r3 = float(row["rain_3d"].iloc[0])
    r7 = float(row["rain_7d"].iloc[0])
    r30 = float(row["rain_30d"].iloc[0])
    api = float(row["api"].iloc[0])

    concentration_1d = r1 / (r7 + 1e-6)
    concentration_3d = r3 / (r30 + 1e-6)

    acceleration = (
        (r3 / 3.0)
        / ((r30 / 30.0) + 1e-6)
    )

    drivers = []

    if r1 >= 20:
        drivers.append("High 24-hour rainfall")

    if r3 >= 60:
        drivers.append("Strong 72-hour accumulation")

    if concentration_1d >= 0.18:
        drivers.append("Rainfall concentrated in recent period")

    if acceleration >= 1.5:
        drivers.append("Recent rainfall intensity elevated")

    if api >= 120:
        drivers.append("Elevated antecedent wetness")

    if not drivers:
        drivers.append("No dominant rainfall trigger detected")

    return {
        "temporal_risk": risk_score,
        "risk_level": risk_level,

        "rainfall_24h": round(r1, 2),
        "rainfall_72h": round(r3, 2),
        "rainfall_7d": round(r7, 2),
        "rainfall_30d": round(r30, 2),

        "api": round(api, 2),

        "rainfall_concentration_24h_7d":
            round(concentration_1d, 3),

        "rainfall_concentration_72h_30d":
            round(concentration_3d, 3),

        "rainfall_acceleration":
            round(acceleration, 3),

        "drivers": drivers,
    }