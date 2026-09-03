"""
Inference Script for Spatial Landslide Susceptibility Baseline

Loads trained pipelines and generates probability predictions P(landslide=1 | spatial features)
for new input coordinates or test datasets.
"""
import os
import joblib
import pandas as pd
import numpy as np


def load_pipeline(model_path="models/randomforest_model.pkl"):
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found at {model_path}")
    return joblib.load(model_path)


def predict_susceptibility(df, model_path="models/randomforest_model.pkl"):
    pipeline = load_pipeline(model_path)
    probs = pipeline.predict_proba(df)[:, 1]
    df_res = df.copy()
    df_res["susceptibility_prob"] = np.round(probs, 4)
    return df_res


if __name__ == "__main__":
    import sys
    from preprocessing import load_dataset, get_feature_sets
    
    input_file = sys.argv[1] if len(sys.argv) > 1 else "data/processed/ner_landslide_v1.csv"
    model_file = sys.argv[2] if len(sys.argv) > 2 else "models/randomforest_model.pkl"
    
    if os.path.exists(input_file) and os.path.exists(model_file):
        df = load_dataset(input_file)
        features, _ = get_feature_sets(df, "A")
        res = predict_susceptibility(df[features], model_file)
        print("Sample Predictions (First 5 Rows):")
        print(res[["susceptibility_prob"]].head().to_string())
    else:
        print("Input file or model file not found.")
