import os
import sys
import pandas as pd
from narrative import generate_explanation


CURRENT_DIR = os.path.dirname(
    os.path.abspath(__file__)
)

sys.path.append(CURRENT_DIR)


from explainer import (
    load_model,
    create_explainer,
    explain_prediction
)

from config import MODEL_FEATURES


# ==========================================
# PATHS
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


# ==========================================
# LOAD DATA
# ==========================================

print("\nLoading model...")

model = load_model()

print("Model loaded successfully!")


print("\nLoading dataset...")

df = pd.read_csv(DATA_PATH)

print(f"Dataset shape: {df.shape}")


# ==========================================
# SELECT SAMPLE
# ==========================================

# Pick first test sample
test_sample = df[
    df["split"] == "test"
].iloc[[0]]


X_sample = test_sample[
    MODEL_FEATURES
]


print("\nSample selected.")


# ==========================================
# CREATE EXPLAINER
# ==========================================

print("\nCreating SHAP explainer...")

explainer = create_explainer(model)

print("Explainer ready!")


# ==========================================
# EXPLAIN PREDICTION
# ==========================================

print("\nGenerating explanation...")

probability, contributions = explain_prediction(
    model,
    explainer,
    X_sample
)


# ==========================================
# OUTPUT
# ==========================================

print("\n" + "=" * 50)

print("LANDSLIDE PREDICTION EXPLANATION")

print("=" * 50)


print(
    f"\nPredicted Landslide Probability: "
    f"{probability:.4f}"
)


print("\nTop Contributing Features:")


print("\nTop Contributing Features:")

for feature, value in list(contributions.items())[:10]:

    direction = (
        "INCREASED RISK"
        if value > 0
        else "DECREASED RISK"
    )

    print(
        f"{feature:20} "
        f"{value:+.4f} "
        f"({direction})"
    )
print("\n" + "=" * 50)
print("HUMAN-READABLE EXPLANATION")
print("=" * 50)

print("HUMAN-READABLE EXPLANATION")

print("=" * 50)


explanation = generate_explanation(
    probability,
    contributions
)


print(
    "\nPrediction Probability:",
    f"{explanation['prediction_probability']}%"
)


print(
    "\nPrimary Risk-Increasing Driver:",
    explanation["primary_risk_driver"]
)

print(
    "Primary Risk-Reducing Factor:",
    explanation["primary_risk_reducer"]
)

print(
    "Strongest Overall Contributor:",
    explanation["strongest_overall_factor"]
)


print("\nFactors Increasing Risk:")

for factor in explanation[
    "risk_increasing_factors"
]:

    print(f"- {factor}")


print("\nFactors Decreasing Risk:")

for factor in explanation[
    "risk_decreasing_factors"
]:

    print(f"- {factor}")


print("\nSummary:")

print(explanation["summary"])