import numpy as np
import pandas as pd


TEMPORAL_RAW = [
    "rain_1d",
    "rain_3d",
    "rain_7d",
    "rain_15d",
    "rain_30d",
    "api",
    "rain_annual_mean",
]


def add_temporal_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create interpretable temporal rainfall indicators.

    The original rainfall features are preserved.
    New features describe:
      - recent rainfall concentration
      - rainfall acceleration
      - accumulation
      - antecedent wetness
      - normalized rainfall intensity
    """

    df = df.copy()

    eps = 1e-6

    # ---------------------------------------------------------
    # 1. Rainfall concentration
    # ---------------------------------------------------------

    df["rain_1d_ratio_7d"] = (
        df["rain_1d"] / (df["rain_7d"] + eps)
    )

    df["rain_3d_ratio_30d"] = (
        df["rain_3d"] / (df["rain_30d"] + eps)
    )

    df["rain_7d_ratio_30d"] = (
        df["rain_7d"] / (df["rain_30d"] + eps)
    )

    # ---------------------------------------------------------
    # 2. Rainfall acceleration
    #
    # Recent 3-day daily rate compared with
    # long-term 30-day daily rate.
    # ---------------------------------------------------------

    df["rainfall_acceleration"] = (
        (df["rain_3d"] / 3.0)
        / ((df["rain_30d"] / 30.0) + eps)
    )

    # ---------------------------------------------------------
    # 3. Short-term vs medium-term intensity
    # ---------------------------------------------------------

    df["rain_1d_intensity"] = df["rain_1d"]

    df["rain_3d_daily_intensity"] = df["rain_3d"] / 3.0

    df["rain_7d_daily_intensity"] = df["rain_7d"] / 7.0

    df["rain_30d_daily_intensity"] = df["rain_30d"] / 30.0

    # ---------------------------------------------------------
    # 4. Antecedent wetness relative to recent accumulation
    # ---------------------------------------------------------

    df["api_ratio_30d"] = (
        df["api"] / (df["rain_30d"] + eps)
    )

    # ---------------------------------------------------------
    # 5. Long-term rainfall normalization
    #
    # NOTE:
    # rain_annual_mean is annual climatological rainfall,
    # so this is only a contextual normalization and should
    # not be interpreted as a true daily rainfall anomaly.
    # ---------------------------------------------------------

    df["rain_30d_fraction_annual"] = (
        df["rain_30d"]
        / (df["rain_annual_mean"] + eps)
    )

    # ---------------------------------------------------------
    # 6. Log transforms for highly skewed rainfall variables
    # ---------------------------------------------------------

    for col in [
        "rain_1d",
        "rain_3d",
        "rain_7d",
        "rain_15d",
        "rain_30d",
        "api",
    ]:
        df[f"log_{col}"] = np.log1p(
            np.clip(df[col], 0, None)
        )

    return df


def get_temporal_features(df: pd.DataFrame) -> pd.DataFrame:
    """Return only features used by the temporal model."""

    df = add_temporal_features(df)

    features = [
        # Raw rainfall
        "rain_1d",
        "rain_3d",
        "rain_7d",
        "rain_15d",
        "rain_30d",
        "api",

        # Temporal structure
        "rain_1d_ratio_7d",
        "rain_3d_ratio_30d",
        "rain_7d_ratio_30d",
        "rainfall_acceleration",

        # Intensity
        "rain_3d_daily_intensity",
        "rain_7d_daily_intensity",
        "rain_30d_daily_intensity",

        # Wetness
        "api_ratio_30d",

        # Context
        "rain_30d_fraction_annual",

        # Stabilized rainfall
        "log_rain_1d",
        "log_rain_3d",
        "log_rain_7d",
        "log_rain_15d",
        "log_rain_30d",
        "log_api",
    ]

    return df[features]