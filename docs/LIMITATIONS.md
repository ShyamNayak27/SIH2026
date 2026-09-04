# Limitations of Current Data Foundation & Spatial Modeling Framework

**Project:** Problem Statement 26001 — AI-Based Early Warning and Landslide Risk Monitoring System in NER  
**Audited Dataset:** `ner_landslide_v1.csv`

---

## 1. Primary Inventory Limitations

### A. Severe Location Uncertainty (Media Report Centroids)
* Out of 341 positive landslide events, **only 4 events (1.2%) are exact ($\le 100\text{ m}$)** and **41 events (12.0%) have 1 km accuracy**.
* Over **86.8% of events have location errors of 5 km to 50 km**. Coordinates represent town centers, highway posts, or district offices where news reports originated, NOT actual landslide failure scars.
* **Empirical Impact**: At 10 km location error, 90 m DEM terrain features (`slope_deg`, `twi`, `tri`, curvature) reflect random noise. Univariate slope gradient AUC is **0.507** (pure random guess).

### B. High-Accuracy Performance Collapse
* When evaluating Random Forest on the subset of **high-accuracy positives ($\le 1\text{ km}$ error, 45 events)**, test PR-AUC **collapses from 0.382 to 0.073**, and ROC-AUC drops to **0.543**.
* This proves that apparent tree model performance on the full dataset is driven by spatial reporting bias around town centroids rather than real slope stability physics.

---

## 2. Negative Sample (Pseudo-Absence) Limitations

### A. Positive-Unlabelled (PU) Noise
* Background negative samples (688 points) are randomly drawn candidates outside a 500 m buffer around known positives.
* They are **unlabeled background samples**, not confirmed stable slopes. A steep, wet candidate site may represent an unrecorded landslide.

### B. Artificial Metadata Inversion
* Negatives are synthetically assigned `loc_accuracy_km = 0.05` and `label_conf = 1.0`, while real positives have low confidence (`label_conf = 0.45` median).
* Including metadata fields in model features creates severe target leakage.

---

## 3. Spatial Conditioning & Feature Gaps

### A. Omitted Proximity Predictors
* Road and river proximity (`dist_road_m`, `dist_major_road_m`, `dist_stream_m`) are **omitted (all NaN)** due to 11 unretrieved OSM Overpass tiles north of lat 25.7°N. Toe-excavation by hill roads is a primary failure driver in NER.

### B. Missing Geological & Environmental Layers
* Current dataset lacks:
  1. **Lithology / Rock Strength** (GSI Bhukosh)
  2. **Distance to Faults** (GSI Structural Layer)
  3. **Land Use / Land Cover** (ESA WorldCover 10 m)
  4. **Soil Texture & Depth** (SoilGrids 250 m)
  5. **Pre-Event Vegetation / NDVI** (Sentinel-2)

---

## 4. Modeling Limitations

1. **Regional Spatial Fingerprinting**: Tree models (RF, Gradient Boosting) derive over **55% of total feature importance** from regional location traits (`rain_annual_mean`, `relief_500m`, `elevation`), while physical slope gradient (`slope_deg`) contributes under 8%.
2. **Spatial Autocorrelation Sensitivity**: Random CV overestimates PR-AUC by 13.5%–16.2% compared to spatially disjoint 25 km block validation.
