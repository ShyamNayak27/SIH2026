import os
import sys
import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from src.integration.decision_engine import generate_decision
from src.explainability.explainer import (create_explainer,explain_prediction,
                                          load_model,)
from src.explainability.config import MODEL_FEATURES
from src.temporal.predict import get_temporal_risk
from src.temporal.predict import get_temporal_risk


CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(CURRENT_DIR, "../.."))


app = FastAPI(title="TerraWatch Landslide Risk API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

DATA_PATH = os.path.join(
    PROJECT_ROOT, "data", "processed", "ner_landslide_v1.csv"
)

dataframe = pd.read_csv(DATA_PATH)
model = load_model()
explainer = create_explainer(model)

DISPLAY_NAMES = {
    "rain_1d": "Recent 1-day rainfall",
    "rain_3d": "Recent 3-day rainfall",
    "rain_7d": "Recent 7-day rainfall",
    "slope_deg": "Slope angle",
    "tri": "Terrain ruggedness",
    "prof_curv": "Profile curvature",
    "plan_curv": "Plan curvature",
    "elevation": "Elevation",
    "api": "Antecedent precipitation",
}


def driver_level(relative_strength):
    if relative_strength >= 75:
        return "high"
    if relative_strength >= 40:
        return "medium"
    return "low"


@app.get("/api/risk/{sample_id}")
def get_risk(sample_id: str):
    sample = dataframe[dataframe["sample_id"] == sample_id]

    if sample.empty:
        raise HTTPException(
            status_code=404,
            detail="Unknown sample ID"
        )

    sample = sample.iloc[[0]]
    features = sample[MODEL_FEATURES]

    spatial_risk, shap_contributions = explain_prediction(
        model,
        explainer,
        features
    )

    temporal_result = get_temporal_risk(sample)
    temporal_risk = temporal_result["temporal_risk"] / 100

    decision = generate_decision(
        spatial_risk=spatial_risk,
        temporal_risk=temporal_risk,
        vision_risk=None,
        shap_contributions=shap_contributions
    )

    top_contributions = list(shap_contributions.items())[:4]
    max_impact = max(
        abs(value) for _, value in top_contributions
    ) or 1

    drivers = [
        {
            "name": DISPLAY_NAMES.get(
                feature,
                feature.replace("_", " ").title()
            ),
            "strength": round(abs(value) / max_impact * 100),
            "direction": "increases" if value > 0 else "reduces",
            "level": driver_level(abs(value) / max_impact * 100)
        }
        for feature, value in top_contributions
    ]

    return {
        "location": {
            "sample_id": sample_id,
            "state": sample.iloc[0]["state"],
            "district": sample.iloc[0]["district"],
            "latitude": sample.iloc[0]["lat"],
            "longitude": sample.iloc[0]["lon"]
        },
        "decision": decision,
        "spatial_risk_score": round(spatial_risk * 100),
        "drivers": drivers,
        "observed_conditions": {
            "rain_1d_mm": round(sample.iloc[0]["rain_1d"], 1),
            "rain_3d_mm": round(sample.iloc[0]["rain_3d"], 1),
            "antecedent_precipitation_index": round(sample.iloc[0]["api"], 1)
        },
        "temporal_analysis": temporal_result,
        "model_status": {
            "spatial": "connected",
            "temporal": "connected",
            "vision": "pending"
        }
    }