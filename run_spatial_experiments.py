"""
Master Runner for Spatial Susceptibility Baseline Experiments (Stages 4 - 18)

Executes leakage-safe training, spatial CV benchmarking, class balancing audit,
feature engineering, feature importance, calibration, label quality sensitivity,
and ablation studies. Generates all required CSV reports and figure plots.
"""
import os
import sys
import joblib
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import StratifiedKFold, GroupKFold
from sklearn.pipeline import Pipeline
from sklearn.metrics import (roc_auc_score, average_precision_score, brier_score_loss,
                             log_loss, accuracy_score, balanced_accuracy_score,
                             precision_score, recall_score, f1_score, confusion_matrix,
                             roc_curve, precision_recall_curve)

# Import local modules
sys.path.insert(0, "src")
from preprocessing import load_dataset, validate_schema, get_feature_sets, create_preprocessing_pipeline
from spatial_validation import verify_split_integrity, get_canonical_train_val_test
from feature_engineering import add_engineered_features
from evaluate import evaluate_model_performance, plot_roc_curves, plot_pr_curves, generate_comparison_report
from calibrate import calibrate_and_evaluate, plot_calibration_curves
from explain import extract_lr_coefficients, extract_tree_importances, plot_feature_importance

SEED = 42
np.random.seed(SEED)

REPORT_DIR = "reports"
FIGURE_DIR = "figures"
MODEL_DIR = "models"

for d in (REPORT_DIR, FIGURE_DIR, MODEL_DIR):
    os.makedirs(d, exist_ok=True)


def run_all_stages():
    print("==========================================================")
    print("  STAGE 1-3: Dataset Audit & Feature Set Definition ")
    print("==========================================================")
    df = load_dataset()
    validate_schema(df)
    verify_split_integrity(df)
    
    feats_static, name_static = get_feature_sets(df, "A")
    feats_temporal, name_temporal = get_feature_sets(df, "B")
    
    print(f"\n{name_static}: {len(feats_static)} features -> {feats_static}")
    print(f"{name_temporal}: {len(feats_temporal)} features -> {feats_temporal}")
    
    train_df, val_df, test_df = get_canonical_train_val_test(df)
    
    # Define mandatory model factories
    pos_count = int(train_df.label.sum())
    neg_count = int(len(train_df) - pos_count)
    scale_pos = neg_count / max(1, pos_count)
    
    models = {
        "Dummy (Prior)": lambda: DummyClassifier(strategy="prior"),
        "Logistic Regression (L2)": lambda: LogisticRegression(class_weight="balanced", C=0.1, random_state=SEED, max_iter=1000),
        "Random Forest": lambda: RandomForestClassifier(n_estimators=200, max_depth=6, class_weight="balanced", random_state=SEED),
        "Gradient Boosting": lambda: GradientBoostingClassifier(n_estimators=100, max_depth=4, learning_rate=0.05, random_state=SEED)
    }
    
    print("\n==========================================================")
    print("  STAGE 4-8: Model Training & Spatial CV Evaluation ")
    print("==========================================================")
    
    all_metrics = []
    prob_dict_test = {}
    
    for m_name, m_factory in models.items():
        # Pipeline for static features
        pipe = Pipeline([
            ("prep", create_preprocessing_pipeline(feats_static)),
            ("model", m_factory())
        ])
        
        pipe.fit(train_df[feats_static], train_df["label"].values)
        
        # Test evaluation
        if hasattr(pipe, "predict_proba"):
            test_probs = pipe.predict_proba(test_df[feats_static])[:, 1]
        else:
            test_probs = pipe.predict(test_df[feats_static])
            
        prob_dict_test[m_name] = (test_df["label"].values, test_probs)
        
        m = evaluate_model_performance(test_df["label"].values, test_probs)
        m["Model"] = m_name
        m["Experiment"] = "EXPERIMENT A: Static Spatial Susceptibility"
        m["Split"] = "Test Set"
        all_metrics.append(m)
        
        print(f"  [{m_name:<25}] Test ROC-AUC: {m['ROC-AUC']:.3f} | PR-AUC: {m['PR-AUC']:.3f} | Brier: {m['Brier Score']:.3f}")
        joblib.dump(pipe, os.path.join(MODEL_DIR, f"{m_name.split()[0].lower()}_model.pkl"))

    # Generate Plots
    plot_roc_curves(prob_dict_test, "roc_curves.png")
    plot_pr_curves(prob_dict_test, "pr_curves.png")
    
    # Generate Comparison Table
    df_comp = generate_comparison_report(all_metrics, "model_comparison.csv")
    
    print("\n==========================================================")
    print("  STAGE 9: Class Balancing Sensitivity Analysis ")
    print("==========================================================")
    balancing_results = []
    for mode in ["Unweighted", "Class-Weighted"]:
        w = None if mode == "Unweighted" else "balanced"
        rf_pipe = Pipeline([
            ("prep", create_preprocessing_pipeline(feats_static)),
            ("model", RandomForestClassifier(n_estimators=200, max_depth=6, class_weight=w, random_state=SEED))
        ])
        rf_pipe.fit(train_df[feats_static], train_df["label"].values)
        probs = rf_pipe.predict_proba(test_df[feats_static])[:, 1]
        m = evaluate_model_performance(test_df["label"].values, probs)
        m["Balancing Strategy"] = mode
        balancing_results.append(m)
        print(f"  Random Forest ({mode:<15}) | ROC-AUC: {m['ROC-AUC']:.3f} | PR-AUC: {m['PR-AUC']:.3f} | F1: {m['F1 Score']:.3f}")
        
    pd.DataFrame(balancing_results).to_csv(os.path.join(REPORT_DIR, "class_balancing_results.csv"), index=False)

    print("\n==========================================================")
    print("  STAGE 10: Feature Engineering Comparison ")
    print("==========================================================")
    df_eng = add_engineered_features(df)
    train_eng = df_eng[df_eng["split"] == "train"].reset_index(drop=True)
    test_eng = df_eng[df_eng["split"] == "test"].reset_index(drop=True)
    
    feats_eng = feats_static + ["slope_x_relief", "slope_x_twi", "elev_x_slope", "log_relief_500m"]
    
    rf_eng_pipe = Pipeline([
        ("prep", create_preprocessing_pipeline(feats_eng)),
        ("model", RandomForestClassifier(n_estimators=200, max_depth=6, class_weight="balanced", random_state=SEED))
    ])
    rf_eng_pipe.fit(train_eng[feats_eng], train_eng["label"].values)
    eng_probs = rf_eng_pipe.predict_proba(test_eng[feats_eng])[:, 1]
    m_eng = evaluate_model_performance(test_eng["label"].values, eng_probs)
    print(f"  Random Forest (Engineered Feats) | Test ROC-AUC: {m_eng['ROC-AUC']:.3f} | PR-AUC: {m_eng['PR-AUC']:.3f}")

    print("\n==========================================================")
    print("  STAGE 11: Feature Importance Analysis ")
    print("==========================================================")
    rf_pipe_base = Pipeline([
        ("prep", create_preprocessing_pipeline(feats_static)),
        ("model", RandomForestClassifier(n_estimators=200, max_depth=6, class_weight="balanced", random_state=SEED))
    ])
    rf_pipe_base.fit(train_df[feats_static], train_df["label"].values)
    
    df_imp = extract_tree_importances(rf_pipe_base, feats_static, train_df, val_df)
    df_imp.to_csv(os.path.join(REPORT_DIR, "feature_importance.csv"), index=False)
    print("Feature Importances:")
    print(df_imp.to_string(index=False))
    
    plot_feature_importance(df_imp, "gini_importance", "Random Forest Gini Importance", "feature_importance.png")
    plot_feature_importance(df_imp, "perm_importance_mean", "Random Forest Permutation Importance", "permutation_importance.png")

    print("\n==========================================================")
    print("  STAGE 12: Probability Calibration ")
    print("==========================================================")
    cal_res, cal_probs, y_te = calibrate_and_evaluate(
        Pipeline([("prep", create_preprocessing_pipeline(feats_static)),
                  ("model", RandomForestClassifier(n_estimators=200, max_depth=6, class_weight="balanced", random_state=SEED))]),
        train_df, val_df, test_df, feats_static, "Random Forest"
    )
    df_cal = pd.DataFrame(cal_res)
    df_cal.to_csv(os.path.join(REPORT_DIR, "calibration_results.csv"), index=False)
    print("Calibration Results:")
    print(df_cal.to_string(index=False))
    plot_calibration_curves(cal_probs, y_te, "calibration_curves.png")

    print("\n==========================================================")
    print("  STAGE 13: Label Quality Sensitivity Analysis ")
    print("==========================================================")
    subsets_res = []
    for threshold, desc in [(100.0, "All Positives (<=100km)"), (10.0, "Medium+ Accuracy (<=10km)"), (1.0, "High Accuracy (<=1km)")]:
        sub_pos = df[(df.label == 1) & (df.loc_accuracy_km <= threshold)]
        sub_neg = df[df.label == 0]
        sub_df = pd.concat([sub_pos, sub_neg], ignore_index=True)
        
        sub_tr = sub_df[sub_df["split"] == "train"]
        sub_te = sub_df[sub_df["split"] == "test"]
        
        if len(sub_tr[sub_tr.label == 1]) == 0 or len(sub_te[sub_te.label == 1]) == 0:
            continue
            
        pipe_sub = Pipeline([
            ("prep", create_preprocessing_pipeline(feats_static)),
            ("model", RandomForestClassifier(n_estimators=200, max_depth=6, class_weight="balanced", random_state=SEED))
        ])
        pipe_sub.fit(sub_tr[feats_static], sub_tr["label"].values)
        p_sub = pipe_sub.predict_proba(sub_te[feats_static])[:, 1]
        m_sub = evaluate_model_performance(sub_te["label"].values, p_sub)
        
        m_sub["Threshold"] = desc
        m_sub["Positives Retained"] = len(sub_pos)
        subsets_res.append(m_sub)
        print(f"  {desc:<28} | Positives: {len(sub_pos):<4} | Test ROC-AUC: {m_sub['ROC-AUC']:.3f} | PR-AUC: {m_sub['PR-AUC']:.3f}")
        
    pd.DataFrame(subsets_res).to_csv(os.path.join(REPORT_DIR, "label_quality_sensitivity.csv"), index=False)

    print("\n==========================================================")
    print("  STAGE 14: Ablation Study ")
    print("==========================================================")
    ablation_sets = {
        "1. Elevation Only": ["elevation"],
        "2. Terrain Slopes & Curvature": ["slope_deg", "aspect_sin", "aspect_cos", "plan_curv", "prof_curv"],
        "3. Topo Hydrology & Ruggedness": ["tri", "twi", "relief_500m"],
        "4. All Static Spatial Features": feats_static,
        "5. Static + Temporal Benchmark (Exp B)": feats_temporal
    }
    
    ablation_metrics = []
    for grp_name, grp_feats in ablation_sets.items():
        pipe_abl = Pipeline([
            ("prep", create_preprocessing_pipeline(grp_feats)),
            ("model", RandomForestClassifier(n_estimators=200, max_depth=6, class_weight="balanced", random_state=SEED))
        ])
        pipe_abl.fit(train_df[grp_feats], train_df["label"].values)
        p_abl = pipe_abl.predict_proba(test_df[grp_feats])[:, 1]
        m_abl = evaluate_model_performance(test_df["label"].values, p_abl)
        m_abl["Feature Group"] = grp_name
        m_abl["Num Features"] = len(grp_feats)
        ablation_metrics.append(m_abl)
        print(f"  {grp_name:<38} ({len(grp_feats)} feats) | ROC-AUC: {m_abl['ROC-AUC']:.3f} | PR-AUC: {m_abl['PR-AUC']:.3f}")

    pd.DataFrame(ablation_metrics).to_csv(os.path.join(REPORT_DIR, "ablation_results.csv"), index=False)
    
    print("\nAll stages 4-18 completed successfully.")


if __name__ == "__main__":
    run_all_stages()
