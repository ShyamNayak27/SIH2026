# Feature configuration for landslide explainability

TERRAIN_FEATURES = [
    "elevation",
    "slope_deg",
    "aspect_sin",
    "aspect_cos",
    "plan_curv",
    "prof_curv",
    "tri",
    "twi",
    "relief_500m"
]


RAINFALL_FEATURES = [
    "rain_1d",
    "rain_3d",
    "rain_7d",
    "rain_15d",
    "rain_30d",
    "api",
    "rain_annual_mean"
]


MODEL_FEATURES = TERRAIN_FEATURES + RAINFALL_FEATURES


FEATURE_METADATA = {

    "elevation": {
        "display_name": "Elevation",
        "description": "elevation characteristics"
    },

    "slope_deg": {
        "display_name": "Slope",
        "description": "terrain slope steepness"
    },

    "aspect_sin": {
        "display_name": "Terrain Aspect (North-South)",
        "description": "terrain orientation"
    },

    "aspect_cos": {
        "display_name": "Terrain Aspect (East-West)",
        "description": "terrain orientation"
    },

    "plan_curv": {
        "display_name": "Plan Curvature",
        "description": "horizontal terrain curvature"
    },

    "prof_curv": {
        "display_name": "Profile Curvature",
        "description": "vertical terrain curvature"
    },

    "tri": {
        "display_name": "Terrain Ruggedness",
        "description": "terrain ruggedness and surface irregularity"
    },

    "twi": {
        "display_name": "Topographic Wetness Index",
        "description": "potential water accumulation in the terrain"
    },

    "relief_500m": {
        "display_name": "Local Relief",
        "description": "elevation variation within the surrounding 500 metre area"
    },

    "rain_1d": {
        "display_name": "1-Day Rainfall",
        "description": "recent rainfall accumulation during the previous day"
    },

    "rain_3d": {
        "display_name": "3-Day Rainfall",
        "description": "cumulative rainfall during the previous three days"
    },

    "rain_7d": {
        "display_name": "7-Day Rainfall",
        "description": "cumulative rainfall during the previous seven days"
    },

    "rain_15d": {
        "display_name": "15-Day Rainfall",
        "description": "cumulative rainfall during the previous fifteen days"
    },

    "rain_30d": {
        "display_name": "30-Day Rainfall",
        "description": "cumulative rainfall during the previous thirty days"
    },

    "api": {
        "display_name": "Antecedent Precipitation Index",
        "description": "accumulated influence of previous rainfall"
    },

    "rain_annual_mean": {
        "display_name": "Annual Mean Rainfall",
        "description": "long-term average rainfall characteristics"
    }
}


RISK_THRESHOLDS = {
    "LOW": 0.25,
    "MODERATE": 0.50,
    "HIGH": 0.75
}


RISK_ACTIONS = {

    "LOW": {
        "alert": "Low landslide risk detected.",
        "action": "Continue routine environmental monitoring."
    },

    "MODERATE": {
        "alert": "Moderate landslide risk detected.",
        "action": "Increase monitoring in potentially vulnerable areas."
    },

    "HIGH": {
        "alert": "High landslide risk detected.",
        "action": "Prepare precautionary measures and increase local monitoring."
    },

    "CRITICAL": {
        "alert": "Critical landslide risk detected.",
        "action": "Issue urgent warnings and activate emergency response protocols."
    }
}