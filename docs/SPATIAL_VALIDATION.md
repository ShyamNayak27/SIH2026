# Spatial Validation Methodology & Leakage Prevention Audit

**Scope:** Problem Statement 26001 — AI-Based Early Warning and Landslide Risk Monitoring System in NER  
**Module:** `src/spatial_validation.py`  
**Audited Dataset:** `data/processed/ner_landslide_v1.csv`

---

## 1. Executive Summary & Core Rationale

In geospatial landslide susceptibility modeling, standard random train/test splits lead to severe **spatial autocorrelation leakage**. Landslides naturally cluster along specific geological structures, valleys, and roads. If observations within a few hundred meters of each other are split randomly into train and test sets, the model effectively memorizes local spatial noise and reports artificially high metrics (e.g. 0.90+ ROC-AUC) that completely collapse in production on unobserved hillsides.

To ensure scientifically valid evaluation, all spatial susceptibility experiments in this project use **Spatially-Disjoint Block Cross-Validation** (`block_id`).

---

## 2. Spatial Block Construction Methodology

1. **Metric Projection**: Coordinates (`lon`, `lat` in EPSG:4326) are converted to metric distances in UTM Zone 46N ($x, y$ in km) using equirectangular conversion at central latitude $25.5^\circ\text{N}$ ($kx = 111320 \cdot \cos(25.5^\circ)$, $ky = 110540$).
2. **Block Tiling**: The study area is partitioned into **$25\text{ km} \times 25\text{ km}$ spatial blocks**:
   $$\text{bx} = \lfloor (x - x_{\min}) / 25.0 \rfloor, \quad \text{by} = \lfloor (y - y_{\min}) / 25.0 \rfloor, \quad \text{block\_id} = \text{bx} \cdot 10000 + \text{by}$$
3. **Disjoint Assignment**: Entire 25 km blocks are assigned atomically to `train`, `val`, or `test` sets. No block is ever split across sets.

---

## 3. Split Integrity Verification Checks

`src/spatial_validation.py` enforces 4 automated integrity checks before model training:

1. **Block Disjointness**: Verifies that no `block_id` appears in more than one split set ($\text{leaking\_blocks} = 0$).
2. **Coordinate Disjointness**: Verifies that duplicate lat/lon coordinates (e.g., recurring historical events at identical town coordinates on different dates) are contained within the same spatial block and split set ($\text{leaking\_coords} = 0$).
3. **Preprocessing Leakage Prevention**: Scalers (`StandardScaler`) and imputers (`SimpleImputer`) are fitted strictly on training data folds (`X_train`), never on full dataset arrays.
4. **Test Set Holdout**: The canonical 15% test set (`split == "test"`) is reserved strictly for final single-pass evaluation and is never used for hyperparameter tuning or threshold selection.

---

## 4. Benchmark Validation Schemes

Every model architecture in this benchmark is evaluated under two parallel validation strategies:

* **Scheme 1: Stratified Random 5-Fold CV (`random`)**: Standard random sample splitting. Measures apparent performance under local spatial correlation.
* **Scheme 2: Spatial Block 5-Fold GroupKFold (`spatial`)**: 5-fold cross-validation grouped by `block_id`. Measures true generalization ability to geographically held-out 25 km terrain blocks.

---

## 5. Verification Results on `ner_landslide_v1.csv`

Running `python src/spatial_validation.py` yields the following verified statistics:

```
--- Split Integrity Report (split) ---
Total rows: 1,029
Unique 25 km spatial blocks: 356
Blocks crossing multiple splits: 0
Coordinates crossing multiple splits: 0
Split integrity verified successfully. Zero spatial block leakage.

Canonical Train: 721 rows (243 pos, base rate = 0.337)
Canonical Val:   136 rows (46 pos, base rate = 0.338)
Canonical Test:  172 rows (51 pos, base rate = 0.297)
```
