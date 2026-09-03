# Final Research Report: Spatial Susceptibility Baseline & Inventory Limitation Benchmark

**Project:** SIH Problem Statement 26001 — AI-Based Early Warning and Landslide Risk Monitoring System in NER  
**Target:** Ministry of Development of North Eastern Region (MDoNER)  
**Deliverable:** Spatial Landslide Susceptibility Baseline & Inventory Quality Assessment

---

## Executive Summary

This report establishes the canonical benchmark for the **Static Spatial Landslide Susceptibility Model** ($P(\text{landslide} = 1 \mid \text{spatial features})$) on the existing dataset (`data/processed/ner_landslide_v1.csv`).

The central research objective was **NOT to produce a high-accuracy production model**, but rather to answer:
> *"What spatial predictive signal can be reliably extracted from our currently available landslide inventory and spatial conditioning data?"*

### Key Findings:
1. **Severe Inventory Location Error**: 86.8% of NASA GLC landslide events have location errors of 5 km to 50 km (media reports pinned to town centroids). Micro-topographic features at 90 m resolution (`slope_deg`, `twi`, `tri`) exhibit zero univariate discrimination (**slope AUC = 0.507**).
2. **Performance Collapse on High-Accuracy Data**: Evaluating Random Forest under Spatial Block CV on the subset of **high-accuracy positives ($\le 1\text{ km}$ error, 45 events)** causes test PR-AUC to **collapse from 0.382 to 0.073** (and ROC-AUC from 0.629 to 0.543).
3. **Regional Fingerprinting Leakage**: Tree models (RF, Gradient Boosting) derive **55.5% of decision weight** from macro regional traits (`rain_annual_mean`, `relief_500m`, `elevation`), while physical slope gradient (`slope_deg`) contributes only **7.6%**. The trees are learning media reporting town locations, not slope stability physics.
4. **Spatial Block CV Drop**: Random 5-Fold CV inflates PR-AUC by 13.5%–16.2% due to spatial autocorrelation. Spatially disjoint 25 km block validation reveals true holdout performance.

---

## 1. Summary of Benchmark Experiments

### A. Model Comparison (Canonical 15% Holdout Test Set)

| Model Architecture | Feature Set | Calibration | ROC-AUC | PR-AUC | Accuracy | F1 Score | Brier Loss | Log Loss |
|---|---|---|---|---|---|---|---|---|
| **Dummy (Prior)** | Base Spatial (10) | None | 0.500 | 0.297 | 0.703 | 0.000 | 0.210 | 0.612 |
| **Logistic Regression** | Base Spatial (10) | Standardized | 0.505 | 0.289 | 0.494 | 0.495 | 0.252 | 0.697 |
| **Random Forest** | Base Spatial (10) | Uncalibrated | 0.629 | 0.382 | 0.669 | 0.280 | 0.214 | 0.620 |
| **Random Forest** | Base Spatial (10) | **Sigmoid (Platt)** | **0.629** | **0.382** | **0.680** | **0.320** | **0.211** | **0.611** |
| **Gradient Boosting** | Base Spatial (10) | Uncalibrated | **0.664** | **0.403** | 0.709 | 0.344 | 0.209 | 0.605 |

---

### B. Class Balancing Sensitivity

| Strategy | Model | ROC-AUC | PR-AUC | Recall | Precision | F1 Score |
|---|---|---|---|---|---|---|
| **Unweighted** | Random Forest | 0.614 | 0.379 | 0.039 | 0.333 | 0.069 |
| **Class-Weighted** | Random Forest | **0.629** | **0.382** | **0.235** | **0.353** | **0.280** |

* **Finding**: Unweighted training on imbalanced data (33.1% positive rate) causes tree models to collapse positive recall to 3.9%. Class-weighted loss functions (`class_weight="balanced"`) are essential to maintain balanced prediction thresholds.

---

### C. Feature Importance Analysis

| Feature Name | RF Gini Importance | Permutation Importance (Mean ± Std) | Role in Model |
|---|---|---|---|
| `rain_annual_mean` | **27.85%** | **0.0780 ± 0.0181** | Regional Climatology Fingerprint |
| `relief_500m` | **15.84%** | **0.0113 ± 0.0140** | Regional Relief Fingerprint |
| `elevation` | **11.83%** | **0.0293 ± 0.0144** | Regional Elevation Fingerprint |
| `tri` | 9.11% | 0.0380 ± 0.0119 | Terrain Ruggedness |
| `twi` | 8.58% | 0.0153 ± 0.0090 | Topographic Wetness |
| `slope_deg` | **7.63%** | **0.0227 ± 0.0104** | Physical Slope Gradient (Rank 6) |

---

### D. Label Quality Sensitivity Audit (The Core Finding)

| Subset Threshold | Positives Retained | Test ROC-AUC | Test PR-AUC | Brier Loss | Interpretation |
|---|---|---|---|---|---|
| **All Events ($\le 100\text{ km}$)** | 341 | 0.629 | 0.382 | 0.214 | Full catalog (dominated by town centroids) |
| **Medium+ Accuracy ($\le 10\text{ km}$)** | 210 | 0.668 | 0.288 | 0.218 | Intermediate catalog |
| **High Accuracy ($\le 1\text{ km}$)** | **45** | **0.543** | **0.073** | **0.212** | **Performance collapses to random guess** |

---

## 2. Answers to Core Research Questions (Phase 23)

### A. Does the current dataset support useful spatial susceptibility modelling?
**NO.** At 90 m resolution, the NASA GLC catalog has a median location error of 10–25 km. Micro-terrain features (`slope_deg`, `twi`, `tri`) at recorded town coordinates are random noise relative to actual failure scars.

### B. How well do Logistic Regression, Random Forest, and Gradient Boosting perform?
* **Logistic Regression**: Fails completely ($\text{ROC-AUC} = 0.505$), correctly reflecting the lack of linear spatial signal.
* **Random Forest & Gradient Boosting**: Achieve apparent ROC-AUCs of 0.63–0.66, but this performance is driven by regional location fingerprinting (`rain_annual_mean`, `relief_500m`, `elevation`) rather than slope physics.

### C. Does spatial validation substantially change reported performance?
**YES.** Random 5-Fold CV overestimates PR-AUC by **13.5% to 16.2%** compared to spatially disjoint 25 km block validation due to spatial autocorrelation.

### D. How much does label quality affect performance?
**DEVASTATINGLY.** When low-accuracy town centroid points are filtered out, PR-AUC collapses from **0.382 to 0.073** (and ROC-AUC from 0.629 to 0.543).

### E. How much does class balancing affect performance?
Unweighted training causes recall to collapse to 3.9%. `class_weight="balanced"` restores recall to ~24% without degrading ROC-AUC.

### F. Which environmental feature groups contain the strongest signal?
Macro regional features (`rain_annual_mean`, `relief_500m`, `elevation`) contribute 55.5% of model importance. Micro-terrain slope and curvature contribute under 7.6%.

### G. Are resulting probabilities calibrated?
Uncalibrated Random Forest has a Brier score of 0.2141. **Sigmoid (Platt) calibration** improves Brier score to **0.2112** and Log Loss to **0.6109**.

### H. What are the major limitations?
Coarse location errors (10–25 km), missing road proximity (`dist_road_m` is all NaN), missing geology/lithology/soil layers, and spatial reporting bias toward populated towns.

### I. What additional data would most improve the susceptibility model?
1. **Surveyed Landslide Polygons** from GSI Bhukosh / NGDR with sub-30m coordinate precision.
2. **Complete OSM Road Coverage** across all 16 tiles to enable `dist_road_m`.
3. **Lithology & Structural Fault Layers** from GSI.

### J. How should this baseline interface with the separate temporal model?
The static spatial susceptibility layer ($P(\text{landslide} \mid \text{terrain})$) must be kept strictly separate from the temporal early warning model ($P(\text{landslide on day } t \mid \text{rain}_t)$). The temporal model should operate at **District $\times$ Day** resolution using daily IMD rainfall dynamics.
