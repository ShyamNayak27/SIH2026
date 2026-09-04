import os
import sys
from src.risk_engine.fusion import fuse_risks
from src.risk_engine.alerts import (classify_risk,get_alert_message,
                                    apply_escalation,)
from src.explainability.narrative import generate_explanation


CURRENT_DIR = os.path.dirname(
    os.path.abspath(__file__)
)

PROJECT_ROOT = os.path.abspath(
    os.path.join(CURRENT_DIR, "../..")
)
"""
Decision Engine

Combines:
- Model predictions
- Risk fusion
- Alert classification
- Escalation logic
- SHAP-based explainability

into one final landslide risk response.
"""


sys.path.append(
    os.path.join(PROJECT_ROOT, "src", "risk_engine")
)

sys.path.append(
    os.path.join(PROJECT_ROOT, "src", "explainability")
)


# ==========================================
# MAIN DECISION FUNCTION
# ==========================================

def generate_decision(
    spatial_risk=None,
    temporal_risk=None,
    vision_risk=None,
    shap_contributions=None
):
    """
    Generates a complete landslide risk decision.

    Parameters:
        spatial_risk:
            Probability from spatial ML model.

        temporal_risk:
            Probability from temporal model.

        vision_risk:
            Probability from satellite/vision model.

        shap_contributions:
            SHAP feature contributions from
            the spatial model.

    Returns:
        dict containing:
        - final risk score
        - alert level
        - action
        - model contributions
        - explainability information
    """


    # ==========================================
    # RISK FUSION
    # ==========================================

    final_risk, model_contributions = fuse_risks(

        spatial_risk=spatial_risk,

        temporal_risk=temporal_risk,

        vision_risk=vision_risk

    )


    # ==========================================
    # BASE ALERT LEVEL
    # ==========================================

    base_level = classify_risk(
        final_risk
    )


    # ==========================================
    # ESCALATION LOGIC
    # ==========================================

    final_level, escalation_reason = apply_escalation(

        base_level,

        spatial_risk=spatial_risk,

        temporal_risk=temporal_risk,

        vision_risk=vision_risk

    )


    # ==========================================
    # ALERT ACTION
    # ==========================================

    action = get_alert_message(
        final_level
    )


    # ==========================================
    # EXPLAINABILITY
    # ==========================================

    explanation = None

    if (
        spatial_risk is not None
        and shap_contributions is not None
    ):

        explanation = generate_explanation(

            probability=spatial_risk,

            contributions=shap_contributions

        )


    # ==========================================
    # FINAL RESPONSE
    # ==========================================

    decision = {

        "final_risk": round(
            final_risk,
            4
        ),

        "base_alert_level": base_level,

        "final_alert_level": final_level,

        "action": action,

        "model_contributions": {

            model: round(value, 4)

            for model, value
            in model_contributions.items()

        },

        "escalation_reason": escalation_reason,

        "explanation": explanation

    }


    return decision