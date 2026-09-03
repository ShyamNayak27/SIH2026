"""
Feature Importance & Model Explanation Module

Calculates Logistic Regression coefficients, Random Forest Gini feature importances,
and Permutation Importances under spatial block cross-validation.
Generates CSV reports and visual bar charts.
"""
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from sklearn.inspection import permutation_importance

FIGURE_DIR = "figures"
REPORT_DIR = "reports"
os.makedirs(FIGURE_DIR, exist_ok=True)
os.makedirs(REPORT_DIR, exist_ok=True)


def extract_lr_coefficients(lr_pipeline, feature_cols):
    model = lr_pipeline.named_steps["model"]
    coefs = model.coef_[0]
    df_coef = pd.DataFrame({
        "feature": feature_cols,
        "coefficient": coefs,
        "abs_coefficient": np.abs(coefs),
        "direction": ["Positive" if c > 0 else "Negative" for c in coefs]
    }).sort_values("abs_coefficient", ascending=False)
    return df_coef


def extract_tree_importances(tree_pipeline, feature_cols, train_df, val_df):
    model = tree_pipeline.named_steps["model"]
    prep_name = "prep" if "prep" in tree_pipeline.named_steps else "preprocessor"
    preprocessor = tree_pipeline.named_steps[prep_name]
    
    X_tr = train_df[feature_cols]
    y_tr = train_df["label"].values
    X_va = val_df[feature_cols]
    y_va = val_df["label"].values
    
    X_tr_trans = preprocessor.transform(X_tr)
    X_va_trans = preprocessor.transform(X_va)
    
    gini_imp = model.feature_importances_
    perm_imp = permutation_importance(model, X_va_trans, y_va, n_repeats=10, random_state=42)
    
    df_imp = pd.DataFrame({
        "feature": feature_cols,
        "gini_importance": gini_imp,
        "perm_importance_mean": perm_imp.importances_mean,
        "perm_importance_std": perm_imp.importances_std
    }).sort_values("gini_importance", ascending=False)
    
    return df_imp


def plot_feature_importance(df_imp, imp_col="gini_importance", title="Feature Importance", filename="feature_importance.png"):
    plt.figure(figsize=(9, 5))
    df_sorted = df_imp.sort_values(imp_col, ascending=True)
    plt.barh(df_sorted["feature"], df_sorted[imp_col], color="skyblue", edgecolor="navy")
    plt.xlabel("Importance Score")
    plt.ylabel("Feature")
    plt.title(title)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURE_DIR, filename), dpi=300)
    plt.close()
