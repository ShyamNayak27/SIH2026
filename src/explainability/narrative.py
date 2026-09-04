FEATURE_NAMES = {

    "elevation": "Elevation",

    "slope_deg": "Slope steepness",

    "aspect_sin": "Terrain orientation",

    "aspect_cos": "Terrain orientation",

    "plan_curv": "Plan curvature",

    "prof_curv": "Profile curvature",

    "tri": "Terrain ruggedness",

    "twi": "Topographic wetness",

    "relief_500m": "Local terrain relief",

    "rain_1d": "Recent 1-day rainfall",

    "rain_3d": "Recent 3-day rainfall",

    "rain_7d": "Recent 7-day rainfall",

    "rain_15d": "Recent 15-day rainfall",

    "rain_30d": "Recent 30-day rainfall",

    "api": "Antecedent precipitation",

    "rain_annual_mean": "Historical annual rainfall"
}


def get_feature_name(feature):
    """
    Converts technical feature names into
    human-readable names.
    """

    return FEATURE_NAMES.get(
        feature,
        feature.replace("_", " ").title()
    )


def generate_explanation(
    probability,
    contributions,
    top_n=5
):
    """
    Converts SHAP feature contributions
    into a human-readable explanation.
    """

    top_features = list(
        contributions.items()
    )[:top_n]

    increasing = []
    decreasing = []

    increasing_features = []
    decreasing_features = []


    # ------------------------------------------
    # SEPARATE POSITIVE AND NEGATIVE FEATURES
    # ------------------------------------------

    for feature, value in top_features:

        readable_name = get_feature_name(feature)

        if value > 0:

            increasing.append(readable_name)

            increasing_features.append(
                (feature, value)
            )

        else:

            decreasing.append(readable_name)

            decreasing_features.append(
                (feature, value)
            )


    explanation = {}


    # ------------------------------------------
    # PREDICTION PROBABILITY
    # ------------------------------------------

    explanation["prediction_probability"] = round(
        probability * 100,
        2
    )


    # ------------------------------------------
    # RISK FACTORS
    # ------------------------------------------

    explanation["risk_increasing_factors"] = increasing

    explanation["risk_decreasing_factors"] = decreasing


    # ------------------------------------------
    # PRIMARY RISK-INCREASING DRIVER
    # ------------------------------------------

    if increasing_features:

        primary_increasing = max(
            increasing_features,
            key=lambda x: x[1]
        )[0]

        explanation["primary_risk_driver"] = (
            get_feature_name(primary_increasing)
        )

    else:

        explanation["primary_risk_driver"] = None


    # ------------------------------------------
    # PRIMARY RISK-REDUCING FACTOR
    # ------------------------------------------

    if decreasing_features:

        primary_decreasing = min(
            decreasing_features,
            key=lambda x: x[1]
        )[0]

        explanation["primary_risk_reducer"] = (
            get_feature_name(primary_decreasing)
        )

    else:

        explanation["primary_risk_reducer"] = None


    # ------------------------------------------
    # STRONGEST OVERALL CONTRIBUTOR
    # ------------------------------------------

    if top_features:

        strongest_feature = top_features[0][0]

        explanation["strongest_overall_factor"] = (
            get_feature_name(strongest_feature)
        )

    else:

        explanation["strongest_overall_factor"] = None


    # ------------------------------------------
    # HUMAN-READABLE SUMMARY
    # ------------------------------------------

    summary = (
        f"The spatial model estimates a landslide probability of "
        f"{probability * 100:.1f}%. "
    )


    if increasing:

        summary += (
            "The main factors pushing the model toward "
            "higher risk are "
            + ", ".join(increasing)
            + ". "
        )


    if decreasing:

        summary += (
            "Factors pushing the model toward lower risk include "
            + ", ".join(decreasing)
            + "."
        )


    explanation["summary"] = summary


    return explanation


    # ------------------------------------------
    # RISK SUMMARY
    # ------------------------------------------

    explanation["prediction_probability"] = round(
        probability * 100,
        2
    )


    # ------------------------------------------
    # PRIMARY RISK DRIVERS
    # ------------------------------------------

    explanation["risk_increasing_factors"] = increasing

    explanation["risk_decreasing_factors"] = decreasing


    # ------------------------------------------
    # PRIMARY DRIVER
    # ------------------------------------------

    if top_features:

        primary_feature = top_features[0][0]

        explanation["primary_driver"] = get_feature_name(
            primary_feature
        )

    else:

        explanation["primary_driver"] = None


    # ------------------------------------------
    # HUMAN READABLE SUMMARY
    # ------------------------------------------

    if increasing:

        increasing_text = ", ".join(increasing)

        summary = (
            f"The predicted landslide probability is "
            f"{probability * 100:.1f}%. "
            f"The primary factors increasing risk are "
            f"{increasing_text}."
        )

    else:

        summary = (
            f"The predicted landslide probability is "
            f"{probability * 100:.1f}%. "
            f"No major risk-increasing factors were "
            f"identified among the top contributors."
        )


    if decreasing:

        decreasing_text = ", ".join(decreasing)

        summary += (
            f" Factors reducing the predicted risk include "
            f"{decreasing_text}."
        )


    explanation["summary"] = summary


    return explanation