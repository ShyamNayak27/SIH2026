"""
Feature Engineering & Multicollinearity Diagnostics Module

Implements physically justified terrain feature transformations and calculates
correlation matrices and Variance Inflation Factors (VIF) to diagnose redundancy.
"""
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

from preprocessing import STATIC_SPATIAL_FEATURES


def add_engineered_features(df):
    """
    Applies non-linear and interaction transformations to static spatial features.
    """
    df_out = df.copy()
    
    # 1. Logarithmic transform for skewed variables
    if "relief_500m" in df_out.columns:
        df_out["log_relief_500m"] = np.log1p(np.maximum(0, df_out["relief_500m"]))
    if "twi" in df_out.columns:
        df_out["log_twi"] = np.log1p(np.maximum(0, df_out["twi"]))
        
    # 2. Physically justified interaction terms
    if "slope_deg" in df_out.columns and "relief_500m" in df_out.columns:
        df_out["slope_x_relief"] = df_out["slope_deg"] * df_out["relief_500m"]
    if "slope_deg" in df_out.columns and "twi" in df_out.columns:
        df_out["slope_x_twi"] = df_out["slope_deg"] * df_out["twi"]
    if "elevation" in df_out.columns and "slope_deg" in df_out.columns:
        df_out["elev_x_slope"] = df_out["elevation"] * df_out["slope_deg"]
        
    return df_out


def compute_vif(df, features):
    """
    Calculates Variance Inflation Factor (VIF) using linear correlation inverse matrix.
    """
    X = df[features].dropna().copy()
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    corr_matrix = np.corrcoef(X_scaled, rowvar=False)
    try:
        inv_corr = np.linalg.inv(corr_matrix)
        vif_vals = np.diag(inv_corr)
    except np.linalg.LinAlgError:
        vif_vals = np.full(len(features), np.nan)
        
    vif_data = pd.DataFrame({
        "feature": features,
        "VIF": vif_vals
    }).sort_values("VIF", ascending=False)
    return vif_data


def compute_correlation_matrix(df, features):
    """
    Calculates pairwise Pearson correlation matrix for feature redundancy analysis.
    """
    return df[features].corr()


if __name__ == "__main__":
    from preprocessing import load_dataset
    df = load_dataset()
    df_eng = add_engineered_features(df)
    vif_df = compute_vif(df, STATIC_SPATIAL_FEATURES)
    print("Variance Inflation Factor (VIF) for Base Static Features:")
    print(vif_df.to_string(index=False))
