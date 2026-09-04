"""
Alert Classification Module

Converts continuous landslide risk scores
into operational alert levels.
"""


RISK_THRESHOLDS = {
    "LOW": 0.25,
    "MODERATE": 0.50,
    "HIGH": 0.75,
    "SEVERE": 1.00
}


def classify_risk(final_risk):
    """
    Converts a risk score between 0 and 1
    into an alert category.

    Returns:
        str: LOW, MODERATE, HIGH, or SEVERE
    """

    if not 0 <= final_risk <= 1:
        raise ValueError(
            "Risk score must be between 0 and 1."
        )

    if final_risk <= RISK_THRESHOLDS["LOW"]:
        return "LOW"

    elif final_risk <= RISK_THRESHOLDS["MODERATE"]:
        return "MODERATE"

    elif final_risk <= RISK_THRESHOLDS["HIGH"]:
        return "HIGH"

    else:
        return "SEVERE"


def get_alert_message(risk_level):
    """
    Returns an operational message
    corresponding to the risk level.
    """

    messages = {

        "LOW":
            "Conditions appear stable. Continue routine monitoring.",

        "MODERATE":
            "Elevated risk detected. Increase monitoring frequency.",

        "HIGH":
            "High landslide risk detected. Prepare preventive response measures.",

        "SEVERE":
            "Severe landslide risk detected. Immediate emergency preparedness is recommended."
    }

    return messages.get(
        risk_level,
        "Unknown risk level."
    )
    # ==========================================
# ESCALATION LOGIC
# ==========================================


RISK_LEVEL_ORDER = {
    "LOW": 0,
    "MODERATE": 1,
    "HIGH": 2,
    "SEVERE": 3
}


def escalate_level(current_level):
    """
    Escalates a risk level by one category.
    """

    levels = [
        "LOW",
        "MODERATE",
        "HIGH",
        "SEVERE"
    ]

    current_index = levels.index(current_level)

    if current_index < len(levels) - 1:
        return levels[current_index + 1]

    return current_level


def apply_escalation(
    base_level,
    spatial_risk=None,
    temporal_risk=None,
    vision_risk=None
):
    """
    Applies conservative rule-based escalation.

    Escalation requires agreement between
    multiple independent risk signals.

    Returns:
        final_level
        escalation_reason
    """

    high_signals = 0
    severe_signals = 0

    risks = {
        "spatial": spatial_risk,
        "temporal": temporal_risk,
        "vision": vision_risk
    }

    for risk in risks.values():

        if risk is None:
            continue

        if risk >= 0.75:
            severe_signals += 1

        elif risk >= 0.50:
            high_signals += 1


    # --------------------------------------
    # RULE 1
    # Multiple severe signals
    # --------------------------------------

    if severe_signals >= 2:

        return (
            "SEVERE",
            "Multiple independent models indicate severe conditions."
        )


    # --------------------------------------
    # RULE 2
    # Multiple elevated signals
    # --------------------------------------

    total_elevated = high_signals + severe_signals

    if total_elevated >= 2 and base_level == "MODERATE":

        return (
            "HIGH",
            "Multiple independent models indicate elevated conditions."
        )


    # --------------------------------------
    # RULE 3
    # Single severe component
    # Flag but don't escalate
    # --------------------------------------

    if severe_signals == 1:

        return (
            base_level,
            "One component indicates severe conditions and requires attention."
        )


    # --------------------------------------
    # NO ESCALATION
    # --------------------------------------

    return (
        base_level,
        None
    )