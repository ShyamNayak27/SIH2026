import os
import sys
import joblib
import shap


CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))

PROJECT_ROOT = os.path.abspath(
    os.path.join(CURRENT_DIR, "../..")
)

sys.path.append(CURRENT_DIR)

from config import MODEL_FEATURES


# ==========================================
# PATHS
# ==========================================

MODEL_PATH = os.path.join(
    PROJECT_ROOT,
    "models",
    "landslide_model.pkl"
)


# ==========================================
# LOAD MODEL
# ==========================================

def load_model():
    """
    Loads the trained landslide prediction model.
    """

    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(
            f"Model not found at: {MODEL_PATH}"
        )

    return joblib.load(MODEL_PATH)


# ==========================================
# CREATE SHAP EXPLAINER
# ==========================================

def create_explainer(model):
    """
    Creates a SHAP TreeExplainer for the
    trained Random Forest model.
    """

    return shap.TreeExplainer(model)


# ==========================================
# EXPLAIN PREDICTION
# ==========================================

def explain_prediction(model, explainer, sample):
    """
    Generates SHAP explanations for a single
    landslide prediction.

    Returns:
        prediction_probability
        feature_contributions
    """

    # Ensure correct feature order
    sample = sample[MODEL_FEATURES]

    # Get landslide probability (class = 1)
    probability = model.predict_proba(sample)[0][1]

    # Generate SHAP values
    shap_values = explainer.shap_values(sample)

    # ------------------------------------------
    # HANDLE DIFFERENT SHAP OUTPUT FORMATS
    # ------------------------------------------

    # Older SHAP versions:
    # [class_0_values, class_1_values]
    if isinstance(shap_values, list):

        values = shap_values[1][0]

    # Newer SHAP versions:
    # (samples, features, classes)
    elif len(shap_values.shape) == 3:

        values = shap_values[0, :, 1]

    # Standard:
    # (samples, features)
    else:

        values = shap_values[0]

    # ------------------------------------------
    # MAP FEATURES TO SHAP CONTRIBUTIONS
    # ------------------------------------------

    feature_contributions = {}

    for feature, value in zip(MODEL_FEATURES, values):

        feature_contributions[feature] = float(value)

    # Sort by absolute importance
    feature_contributions = dict(
        sorted(
            feature_contributions.items(),
            key=lambda item: abs(item[1]),
            reverse=True
        )
    )

    return probability, feature_contributions