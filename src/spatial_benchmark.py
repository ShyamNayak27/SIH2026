"""
Spatial Susceptibility Baseline & Leakage Audit Benchmark
Evaluates Logistic Regression, Random Forest, and Gradient Boosting on spatial features
under both Random Stratified CV and Spatially-Disjoint Block CV.
"""
import os
import sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import StratifiedKFold, GroupKFold
from sklearn.inspection import permutation_importance
from sklearn.metrics import (roc_auc_score, average_precision_score,
                             brier_score_loss, f1_score, precision_score, recall_score)

SEED = 42
np.random.seed(SEED)

DATA_PATH = "data/processed/ner_landslide_v1.csv"
OUT_DIR = "reports"
os.makedirs(OUT_DIR, exist_ok=True)


def load_and_engineer_data(path=DATA_PATH):
    if not os.path.exists(path):
        raise FileNotFoundError(f"Processed dataset not found at {path}")
    
    df = pd.read_csv(path)
    print(f"Loaded dataset: {df.shape[0]} rows, {df.shape[1]} columns.")
    
    base_features = [
        "elevation", "slope_deg", "aspect_sin", "aspect_cos",
        "plan_curv", "prof_curv", "tri", "twi", "relief_500m", "rain_annual_mean"
    ]
    
    feats = [f for f in base_features if f in df.columns]
    
    df_eng = df.copy()
    df_eng["slope_x_relief"] = df_eng["slope_deg"] * df_eng["relief_500m"]
    df_eng["slope_x_twi"] = df_eng["slope_deg"] * df_eng["twi"]
    df_eng["elev_x_slope"] = df_eng["elevation"] * df_eng["slope_deg"]
    df_eng["log_relief"] = np.log1p(np.maximum(0, df_eng["relief_500m"]))
    
    engineered_features = feats + ["slope_x_relief", "slope_x_twi", "elev_x_slope", "log_relief"]
    
    return df_eng, feats, engineered_features


def evaluate_predictions(y_true, y_prob, threshold=0.5):
    y_pred = (y_prob >= threshold).astype(int)
    return {
        "roc_auc": float(roc_auc_score(y_true, y_prob)),
        "pr_auc": float(average_precision_score(y_true, y_prob)),
        "brier": float(brier_score_loss(y_true, y_prob)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0))
    }


def run_cross_validation(df, features, model_name, model_fn, cv_type="random", n_splits=5):
    X = df[features].values
    y = df["label"].values
    groups = df["block_id"].values
    
    y_prob_all = np.zeros(len(df))
    metrics_per_fold = []
    
    if cv_type == "random":
        cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=SEED)
        splits = cv.split(X, y)
    elif cv_type == "spatial":
        cv = GroupKFold(n_splits=n_splits)
        splits = cv.split(X, y, groups=groups)
    else:
        raise ValueError(f"Unknown cv_type: {cv_type}")
        
    for fold, (train_idx, val_idx) in enumerate(splits):
        X_tr, y_tr = X[train_idx], y[train_idx]
        X_va, y_va = X[val_idx], y[val_idx]
        
        clf = model_fn()
        clf.fit(X_tr, y_tr)
        
        if hasattr(clf, "predict_proba"):
            probs = clf.predict_proba(X_va)[:, 1]
        else:
            probs = clf.predict(X_va)
            
        y_prob_all[val_idx] = probs
        m = evaluate_predictions(y_va, probs)
        metrics_per_fold.append(m)
        
    overall_m = evaluate_predictions(y, y_prob_all)
    mean_fold_roc = float(np.mean([m["roc_auc"] for m in metrics_per_fold]))
    std_fold_roc = float(np.std([m["roc_auc"] for m in metrics_per_fold]))
    mean_fold_pr = float(np.mean([m["pr_auc"] for m in metrics_per_fold]))
    
    res = {
        "model": model_name,
        "cv_type": cv_type,
        "num_features": len(features),
        "overall_roc_auc": overall_m["roc_auc"],
        "overall_pr_auc": overall_m["pr_auc"],
        "mean_fold_roc": mean_fold_roc,
        "std_fold_roc": std_fold_roc,
        "mean_fold_pr": mean_fold_pr,
        "brier": overall_m["brier"],
        "f1": overall_m["f1"]
    }
    return res, y_prob_all


def run_benchmark():
    df, base_feats, eng_feats = load_and_engineer_data()
    
    pos_count = int(df.label.sum())
    neg_count = int(len(df) - pos_count)
    
    print(f"Dataset stats: {len(df)} samples ({pos_count} positives, {neg_count} negatives, base rate = {pos_count/len(df):.3f})")
    print(f"Spatial blocks: {df.block_id.nunique()} unique 25km blocks.")
    
    models = {
        "Logistic Regression (L2, Balanced)": lambda: Pipeline([
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(class_weight="balanced", C=0.1, random_state=SEED, max_iter=1000))
        ]),
        "Random Forest (Balanced, max_depth=6)": lambda: RandomForestClassifier(
            n_estimators=200, max_depth=6, class_weight="balanced", random_state=SEED
        ),
        "Gradient Boosting (max_depth=4)": lambda: GradientBoostingClassifier(
            n_estimators=100, max_depth=4, learning_rate=0.05, random_state=SEED
        )
    }
    
    results = []
    prob_dict = {}
    
    for fset_name, feats in [("Base Spatial Features", base_feats), ("Engineered Features", eng_feats)]:
        print(f"\n==================================================")
        print(f" Feature Set: {fset_name} ({len(feats)} features)")
        print(f"==================================================")
        
        for m_name, m_factory in models.items():
            for cv_type in ["random", "spatial"]:
                tag = f"{m_name} | {cv_type.upper()} CV | {fset_name}"
                res, probs = run_cross_validation(df, feats, m_name, m_factory, cv_type=cv_type, n_splits=5)
                res["feature_set"] = fset_name
                results.append(res)
                prob_dict[tag] = probs
                
                print(f"  {m_name:<40} | {cv_type.upper():<7} CV | "
                      f"ROC-AUC: {res['mean_fold_roc']:.3f} ± {res['std_fold_roc']:.3f} | "
                      f"PR-AUC: {res['mean_fold_pr']:.3f}")

    res_df = pd.DataFrame(results)
    res_df.to_csv(os.path.join(OUT_DIR, "spatial_benchmark_results.csv"), index=False)
    
    # Feature Importance Analysis (Random Forest & Gradient Boosting)
    rf_clf = RandomForestClassifier(n_estimators=200, max_depth=6, class_weight="balanced", random_state=SEED)
    rf_clf.fit(df[base_feats], df["label"])
    
    gb_clf = GradientBoostingClassifier(n_estimators=100, max_depth=4, learning_rate=0.05, random_state=SEED)
    gb_clf.fit(df[base_feats], df["label"])
    
    # Permutation importance for RF
    perm_imp = permutation_importance(rf_clf, df[base_feats], df["label"], n_repeats=10, random_state=SEED)
    
    importances = pd.DataFrame({
        "feature": base_feats,
        "rf_gini_importance": rf_clf.feature_importances_,
        "gb_gini_importance": gb_clf.feature_importances_,
        "rf_perm_importance_mean": perm_imp.importances_mean,
        "rf_perm_importance_std": perm_imp.importances_std
    }).sort_values("rf_gini_importance", ascending=False)
    importances.to_csv(os.path.join(OUT_DIR, "feature_importances.csv"), index=False)
    
    print("\nFeature Importances (Base Spatial Features):")
    print(importances.to_string(index=False))
    
    # Subset Audit: High-Accuracy (<=1km) vs Low-Accuracy (>1km)
    print("\n==================================================")
    print(" Subset Audit: High-Accuracy (<=1km) vs Low-Accuracy (>1km)")
    print("==================================================")
    
    hq_positives = df[(df.label == 1) & (df.hq_location == 1)]
    lq_positives = df[(df.label == 1) & (df.hq_location == 0)]
    all_negatives = df[df.label == 0]
    
    df_hq = pd.concat([hq_positives, all_negatives], ignore_index=True)
    df_lq = pd.concat([lq_positives, all_negatives], ignore_index=True)
    
    res_hq_rf, _ = run_cross_validation(df_hq, base_feats, "RF High-Acc Subset (<=1km)", 
                                        lambda: RandomForestClassifier(n_estimators=200, max_depth=6, class_weight="balanced", random_state=SEED),
                                        cv_type="spatial", n_splits=5)
    res_lq_rf, _ = run_cross_validation(df_lq, base_feats, "RF Low-Acc Subset (>1km)", 
                                        lambda: RandomForestClassifier(n_estimators=200, max_depth=6, class_weight="balanced", random_state=SEED),
                                        cv_type="spatial", n_splits=5)
                                        
    print(f"  High-Accuracy Positives ({len(hq_positives)} pos + {len(all_negatives)} neg) | Spatial CV ROC-AUC: {res_hq_rf['mean_fold_roc']:.3f} | PR-AUC: {res_hq_rf['mean_fold_pr']:.3f}")
    print(f"  Low-Accuracy Positives  ({len(lq_positives)} pos + {len(all_negatives)} neg) | Spatial CV ROC-AUC: {res_lq_rf['mean_fold_roc']:.3f} | PR-AUC: {res_lq_rf['mean_fold_pr']:.3f}")
    
    return res_df, importances


if __name__ == "__main__":
    run_benchmark()
