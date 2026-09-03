"""
Probability Calibration Module for Spatial Landslide Susceptibility Baseline

Evaluates raw probabilities vs. Sigmoid (Platt scaling) vs. Isotonic Calibration.
Calibration methods are selected strictly using validation fold data and evaluated
on the holdout test set.
"""
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from sklearn.calibration import CalibratedClassifierCV, calibration_curve
from sklearn.metrics import brier_score_loss, log_loss

FIGURE_DIR = "figures"
REPORT_DIR = "reports"
os.makedirs(FIGURE_DIR, exist_ok=True)
os.makedirs(REPORT_DIR, exist_ok=True)


def calibrate_and_evaluate(model_pipeline, train_df, val_df, test_df, feature_cols, model_name):
    """
    Fits calibration methods on validation data and evaluates Brier score / Log Loss on test data.
    """
    X_tr = train_df[feature_cols]
    y_tr = train_df["label"].values
    X_va = val_df[feature_cols]
    y_va = val_df["label"].values
    X_te = test_df[feature_cols]
    y_te = test_df["label"].values
    
    # 1. Uncalibrated Model
    model_pipeline.fit(X_tr, y_tr)
    raw_val_prob = model_pipeline.predict_proba(X_va)[:, 1]
    raw_test_prob = model_pipeline.predict_proba(X_te)[:, 1]
    
    # 2. Sigmoid Calibration (Platt Scaling)
    sig_calibrator = CalibratedClassifierCV(model_pipeline, method="sigmoid", cv="prefit")
    sig_calibrator.fit(X_va, y_va)
    sig_test_prob = sig_calibrator.predict_proba(X_te)[:, 1]
    
    # 3. Isotonic Calibration
    iso_calibrator = CalibratedClassifierCV(model_pipeline, method="isotonic", cv="prefit")
    iso_calibrator.fit(X_va, y_va)
    iso_test_prob = iso_calibrator.predict_proba(X_te)[:, 1]
    
    eps = 1e-15
    results = [
        {
            "Model": model_name,
            "Calibration Method": "Uncalibrated",
            "Brier Score": float(brier_score_loss(y_te, raw_test_prob)),
            "Log Loss": float(log_loss(y_te, np.clip(raw_test_prob, eps, 1 - eps)))
        },
        {
            "Model": model_name,
            "Calibration Method": "Sigmoid (Platt)",
            "Brier Score": float(brier_score_loss(y_te, sig_test_prob)),
            "Log Loss": float(log_loss(y_te, np.clip(sig_test_prob, eps, 1 - eps)))
        },
        {
            "Model": model_name,
            "Calibration Method": "Isotonic",
            "Brier Score": float(brier_score_loss(y_te, iso_test_prob)),
            "Log Loss": float(log_loss(y_te, np.clip(iso_test_prob, eps, 1 - eps)))
        }
    ]
    
    prob_dict = {
        "Uncalibrated": raw_test_prob,
        "Sigmoid": sig_test_prob,
        "Isotonic": iso_test_prob
    }
    
    return results, prob_dict, y_te


def plot_calibration_curves(prob_dict, y_true, filename="calibration_curves.png"):
    plt.figure(figsize=(8, 6))
    for name, y_prob in prob_dict.items():
        prob_true, prob_pred = calibration_curve(y_true, y_prob, n_bins=10)
        plt.plot(prob_pred, prob_true, marker='o', label=name)
        
    plt.plot([0, 1], [0, 1], 'k--', label="Perfect Calibration")
    plt.xlabel("Mean Predicted Probability")
    plt.ylabel("Fraction of Positives")
    plt.title("Reliability Diagram (Calibration Curves)")
    plt.legend(loc="upper left")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURE_DIR, filename), dpi=300)
    plt.close()
