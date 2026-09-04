"""
Risk Fusion Module

Combines risk scores from independent project components:
- Spatial susceptibility model
- Temporal rainfall risk model
- Remote sensing / vision model
"""


DEFAULT_WEIGHTS = {
    "spatial": 0.45,
    "temporal": 0.35,
    "vision": 0.20
}


def validate_risk(risk):
    """
    Ensures a risk score is between 0 and 1.
    """

    if risk is None:
        return None

    risk = float(risk)

    if risk < 0 or risk > 1:
        raise ValueError(
            f"Risk score must be between 0 and 1. Got: {risk}"
        )

    return risk


def fuse_risks(
    spatial_risk=None,
    temporal_risk=None,
    vision_risk=None,
    weights=None
):
    """
    Combines available risk signals into a final risk score.

    Missing model outputs are handled dynamically.
    Weights are re-normalized based on available signals.

    Returns:
        final_risk (float): Value between 0 and 1
        contributions (dict): Contribution of each model
    """

    if weights is None:
        weights = DEFAULT_WEIGHTS.copy()

    risks = {
        "spatial": validate_risk(spatial_risk),
        "temporal": validate_risk(temporal_risk),
        "vision": validate_risk(vision_risk)
    }

    # Keep only available risk signals
    available = {
        name: value
        for name, value in risks.items()
        if value is not None
    }

    if not available:
        raise ValueError(
            "At least one risk signal must be provided."
        )

    # Get corresponding weights
    active_weights = {
        name: weights[name]
        for name in available
    }

    # Normalize weights
    total_weight = sum(active_weights.values())

    normalized_weights = {
        name: weight / total_weight
        for name, weight in active_weights.items()
    }

    # Weighted fusion
    final_risk = sum(
        available[name] * normalized_weights[name]
        for name in available
    )

    # Calculate contribution
    contributions = {
        name: available[name] * normalized_weights[name]
        for name in available
    }

    return final_risk, contributions