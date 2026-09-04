import os
import sys
import json
import pandas as pd


CURRENT_DIR = os.path.dirname(
    os.path.abspath(__file__)
)

PROJECT_ROOT = os.path.abspath(
    os.path.join(CURRENT_DIR, "../..")
)


# ==========================================
# IMPORT PATHS
# ==========================================

sys.path.append(CURRENT_DIR)

sys.path.append(
    os.path.join(
        PROJECT_ROOT,
        "src",
        "explainability"
    )
)


from decision_engine import generate_decision

from explainer import (
    load_model,
    create_explainer,
    explain_prediction
)

from config import MODEL_FEATURES


# ==========================================
# DATA PATH
# ==========================================

DATA_PATH = os.path.join(
    PROJECT_ROOT,
    "data",
    "processed",
    "ner_landslide_v1.csv"
)


# ==========================================
# LOAD DATA + MODEL
# ==========================================

print("\nLoading dataset...")

df = pd.read_csv(DATA_PATH)


print("Loading trained spatial model...")

model = load_model()


print("Creating SHAP explainer...")

explainer = create_explainer(model)


# ==========================================
# SELECT REAL SAMPLE
# ==========================================

test_sample = df[
    df["split"] == "test"
].iloc[[0]]

X_sample = test_sample[
    MODEL_FEATURES
]


# ==========================================
# SPATIAL PREDICTION + SHAP
# ==========================================

print("\nGenerating spatial prediction...")

spatial_risk, shap_contributions = explain_prediction(

    model,

    explainer,

    X_sample

)


print(
    f"Spatial Risk: {spatial_risk:.4f}"
)


# ==========================================
# GENERATE FINAL DECISION
# ==========================================

print("\nGenerating final decision...")

decision = generate_decision(

    spatial_risk=spatial_risk,

    temporal_risk=None,

    vision_risk=None,

    shap_contributions=shap_contributions

)


# ==========================================
# DISPLAY RESULT
# ==========================================

print("\n" + "=" * 60)

print("FINAL LANDSLIDE RISK DECISION")

print("=" * 60)


print(
    f"\nFinal Risk Score: "
    f"{decision['final_risk']:.4f}"
)


print(
    f"Base Alert Level: "
    f"{decision['base_alert_level']}"
)


print(
    f"Final Alert Level: "
    f"{decision['final_alert_level']}"
)


print(
    f"\nRecommended Action:\n"
    f"{decision['action']}"
)


if decision["escalation_reason"]:

    print(
        f"\nEscalation Reason:\n"
        f"{decision['escalation_reason']}"
    )


print("\nMODEL CONTRIBUTIONS:")

for model_name, contribution in decision[
    "model_contributions"
].items():

    print(
        f"{model_name.capitalize()}: "
        f"{contribution:.4f}"
    )


# ==========================================
# EXPLANATION
# ==========================================

if decision["explanation"]:

    explanation = decision["explanation"]

    print("\n" + "-" * 60)

    print("WHY THIS RISK LEVEL?")

    print("-" * 60)


    print(
        "\nPrimary Risk Driver:"
    )

    print(
        explanation[
            "primary_risk_driver"
        ]
    )


    print(
        "\nRisk Increasing Factors:"
    )

    for factor in explanation[
        "risk_increasing_factors"
    ]:

        print(f"- {factor}")


    print(
        "\nRisk Reducing Factors:"
    )

    for factor in explanation[
        "risk_decreasing_factors"
    ]:

        print(f"- {factor}")


    print("\nExplanation:")

    print(
        explanation["summary"]
    )


print("\n" + "=" * 60)