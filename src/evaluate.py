"""
Evaluation Module for Spatial Landslide Susceptibility Baseline

Computes comprehensive primary and secondary metrics, generates model comparison tables,
and plots ROC curves, Precision-Recall curves, and confusion matrices.
"""
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from sklearn.metrics import (roc_auc_score, average_precision_score, brier_score_loss,
                             log_loss, accuracy_score, balanced_accuracy_score,
                             precision_score, recall_score, f1_score, confusion_matrix,
                             roc_curve, precision_recall_curve)

FIGURE_DIR = "figures"
REPORT_DIR = "reports"
os.makedirs(FIGURE_DIR, exist_ok=True)
os.makedirs(REPORT_DIR, exist_ok=True)


def evaluate_model_performance(y_true, y_prob, threshold=0.5):
    """
    Computes primary (probabilistic) and secondary (class threshold) metrics.
    """
    # Clip probabilities to avoid log loss infinity
    eps = 1e-15
    y_prob_clipped = np.clip(y_prob, eps, 1 - eps)
    y_pred = (y_prob >= threshold).astype(int)
    
    cm = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = cm.ravel() if cm.size == 4 else (0, 0, 0, 0)
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    
    return {
        "ROC-AUC": float(roc_auc_score(y_true, y_prob)),
        "PR-AUC": float(average_precision_score(y_true, y_prob)),
        "Brier Score": float(brier_score_loss(y_true, y_prob)),
        "Log Loss": float(log_loss(y_true, y_prob_clipped)),
        "Accuracy": float(accuracy_score(y_true, y_pred)),
        "Balanced Accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "Precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "Recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "F1 Score": float(f1_score(y_true, y_pred, zero_division=0)),
        "Specificity": float(specificity)
    }


def plot_roc_curves(eval_dict, filename="roc_curves.png"):
    plt.figure(figsize=(8, 6))
    for name, (y_true, y_prob) in eval_dict.items():
        fpr, tpr, _ = roc_curve(y_true, y_prob)
        auc_val = roc_auc_score(y_true, y_prob)
        plt.plot(fpr, tpr, label=f"{name} (AUC = {auc_val:.3f})")
        
    plt.plot([0, 1], [0, 1], 'k--', label="Random Guess (AUC = 0.500)")
    plt.xlabel("False Positive Rate (1 - Specificity)")
    plt.ylabel("True Positive Rate (Recall)")
    plt.title("ROC Curves — Spatial Susceptibility Models")
    plt.legend(loc="lower right")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURE_DIR, filename), dpi=300)
    plt.close()


def plot_pr_curves(eval_dict, filename="pr_curves.png"):
    plt.figure(figsize=(8, 6))
    for name, (y_true, y_prob) in eval_dict.items():
        precision, recall, _ = precision_recall_curve(y_true, y_prob)
        ap_val = average_precision_score(y_true, y_prob)
        plt.plot(recall, precision, label=f"{name} (AP = {ap_val:.3f})")
        
    base_rate = np.mean(list(eval_dict.values())[0][0])
    plt.axhline(base_rate, color='k', linestyle='--', label=f"Base Rate ({base_rate:.3f})")
    plt.xlabel("Recall (Sensitivity)")
    plt.ylabel("Precision (Positive Predictive Value)")
    plt.title("Precision-Recall Curves — Spatial Susceptibility Models")
    plt.legend(loc="upper right")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURE_DIR, filename), dpi=300)
    plt.close()


def generate_comparison_report(metrics_list, output_filename="model_comparison.csv"):
    df_res = pd.DataFrame(metrics_list)
    df_res.to_csv(os.path.join(REPORT_DIR, output_filename), index=False)
    print(f"Model comparison saved to {os.path.join(REPORT_DIR, output_filename)}")
    return df_res
