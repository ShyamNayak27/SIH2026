"""
Preprocessing Module for Spatial Landslide Susceptibility Baseline

Implements leakage-safe data transformation pipelines using sklearn ColumnTransformer
and StandardScaler. All scalers/transformers are fitted strictly on training data.
"""
import os
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer

# Identity and metadata columns to exclude from training features
IDENTITY_COLS = ["sample_id", "lon", "lat", "state", "district", "event_date"]
METADATA_COLS = ["loc_accuracy_km", "label_conf", "trigger", "size", "category", 
                 "fatalities", "source", "hq_location"]
SPLIT_COLS = ["block_id", "split"]
EXCLUDE_COLS = set(IDENTITY_COLS + METADATA_COLS + SPLIT_COLS + ["label"])

# Defined Feature Sets
STATIC_SPATIAL_FEATURES = [
    "elevation", "slope_deg", "aspect_sin", "aspect_cos",
    "plan_curv", "prof_curv", "tri", "twi", "relief_500m", "rain_annual_mean"
]

TEMPORAL_TRIGGER_FEATURES = [
    "rain_1d", "rain_3d", "rain_7d", "rain_15d", "rain_30d", "api"
]

PROXIMITY_FEATURES = ["dist_road_m", "dist_major_road_m", "dist_stream_m"]


def load_dataset(path="data/processed/ner_landslide_v1.csv"):
    if not os.path.exists(path):
        raise FileNotFoundError(f"Processed dataset not found at {path}")
    df = pd.read_csv(path)
    return df


def validate_schema(df):
    """Integrity checks for dataset format and identity columns."""
    assert "label" in df.columns, "Target column 'label' missing"
    assert "block_id" in df.columns, "Spatial block column 'block_id' missing"
    assert df["sample_id"].duplicated().sum() == 0, "Duplicate sample_id detected"
    print("Schema validation passed.")


def get_feature_sets(df, exp_type="A"):
    """
    Returns feature column names for Experiment A (Static Susceptibility) 
    or Experiment B (Static + Temporal Benchmark).
    """
    present_cols = set(df.columns)
    
    # Filter out columns that are all NaN (e.g. proximity features in v1.csv)
    valid_static = [c for c in STATIC_SPATIAL_FEATURES if c in present_cols and df[c].notna().any()]
    
    if exp_type.upper() == "A":
        # Experiment A: Pure Static Susceptibility
        features = valid_static
        exp_name = "EXPERIMENT A: Static Spatial Susceptibility"
    elif exp_type.upper() == "B":
        # Experiment B: Static + Temporal Trigger Benchmark
        valid_temporal = [c for c in TEMPORAL_TRIGGER_FEATURES if c in present_cols and df[c].notna().any()]
        features = valid_static + valid_temporal
        exp_name = "EXPERIMENT B: Static + Temporal Environmental Benchmark"
    else:
        raise ValueError(f"Unknown experiment type: {exp_type}. Choose 'A' or 'B'.")
        
    return features, exp_name


def create_preprocessing_pipeline(feature_cols, scale_numeric=True):
    """
    Creates a scikit-learn Pipeline with SimpleImputer and optional StandardScaler.
    Fitted only on training folds.
    """
    steps = [("imputer", SimpleImputer(strategy="median"))]
    if scale_numeric:
        steps.append(("scaler", StandardScaler()))
        
    numeric_transformer = Pipeline(steps=steps)
    
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, feature_cols)
        ],
        remainder="drop"
    )
    return preprocessor


if __name__ == "__main__":
    df = load_dataset()
    validate_schema(df)
    feats_a, name_a = get_feature_sets(df, "A")
    feats_b, name_b = get_feature_sets(df, "B")
    print(f"{name_a}: {len(feats_a)} features -> {feats_a}")
    print(f"{name_b}: {len(feats_b)} features -> {feats_b}")
