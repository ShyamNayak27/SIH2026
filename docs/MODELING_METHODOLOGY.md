# Spatial Susceptibility Baseline Modeling Methodology

**Project:** Problem Statement 26001 — AI-Based Early Warning and Landslide Risk Monitoring System in NER  
**Target Output:** $P(\text{landslide} = 1 \mid \text{spatial/environmental characteristics})$  
**Framework:** Research Baseline & Inventory Limitation Assessment

---

## 1. Experimental Framing & Objectives

The goal of this experiment is to establish a rigorous, leakage-safe spatial baseline using the existing data foundation (`data/processed/ner_landslide_v1.csv`).

Crucially, **this is NOT a production model development task**. The objective is to measure what spatial signal can be reliably extracted from the current inventory and conditioning factors, and to quantify model sensitivity to spatial validation and label quality.

---

## 2. Experimental Separation

To maintain scientific rigor, two distinct experiments were established:

### EXPERIMENT A: Primary Static Spatial Susceptibility
Features restricted to persistent environmental and terrain characteristics:
* `elevation`, `slope_deg`, `aspect_sin`, `aspect_cos`, `plan_curv`, `prof_curv`, `tri`, `twi`, `relief_500m`, `rain_annual_mean`.
* **Explicit Exclusion**: Short-term triggering rainfall (`rain_1d` .. `rain_30d`, `api`) is strictly excluded from Experiment A.

### EXPERIMENT B: Static + Temporal Environmental Benchmark
Features combining static spatial variables with daily antecedent rainfall totals:
* Base Static Features + `rain_1d`, `rain_3d`, `rain_7d`, `rain_15d`, `rain_30d`, `api`.
* **Note**: Experiment B serves as a comparative benchmark and is NOT presented as the static susceptibility map.

---

## 3. Mandatory Model Architectures & Pipelines

All models are trained using scikit-learn `Pipeline` objects with `ColumnTransformer`, `SimpleImputer` (median), and `StandardScaler`. Scalers and imputers are fitted strictly on training folds to eliminate data leakage.

1. **DummyClassifier (`strategy="prior"`)**: Predicts positive class proportion ($\approx 0.331$). Serves as the naive probability benchmark ($\text{ROC-AUC} = 0.500, \text{PR-AUC} = 0.297$).
2. **Logistic Regression (`L2, class_weight="balanced"`, $C=0.1$)**: Standardized linear baseline.
3. **Random Forest (`n_estimators=200, max_depth=6, class_weight="balanced"`)**: Non-linear tree ensemble baseline.
4. **Gradient Boosting (`n_estimators=100, max_depth=4, lr=0.05`)**: Non-linear boosted tree ensemble.

---

## 4. Leakage-Safe Spatial Validation

* **Spatial Block CV**: 5-fold cross-validation grouped by 25 km $\times$ 25 km spatial blocks (`block_id`).
* **Canonical Holdout Test Set**: The 15% canonical test set (`split == "test"`, 172 samples) is reserved exclusively for final single-pass evaluation and is never used during preprocessing, scaling, hyperparameter tuning, or calibration selection.
* **Integrity Gate**: Split verification confirms 0 blocks and 0 duplicate coordinates cross split boundaries.

---

## 5. Calibration & Class Balancing Protocols

* **Class Balancing**: Evaluated unweighted vs. class-weighted (`class_weight="balanced"`) pipelines. Resampling algorithms (e.g. SMOTE) were rejected to avoid generating unphysical synthetic terrain samples in steep hill topography.
* **Calibration Protocol**: Sigmoid (Platt scaling) and Isotonic calibration models are fitted on validation fold probabilities (`val_df`) and evaluated on the holdout test set (`test_df`). Selection is based on Brier Score and Log Loss minimization.
