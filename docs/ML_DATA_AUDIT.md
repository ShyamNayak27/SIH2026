# Complete ML & Data Foundation Audit: NER Landslide Project

**Scope:** Problem Statement 26001 — AI-Based Early Warning and Landslide Risk Monitoring System in North-Eastern Region (NER)  
**Target Focus:** Machine Learning Landslide Susceptibility & Temporal Trigger Component  
**Audit Date:** September 2026  
**Audited Dataset:** `data/processed/ner_landslide_v1.csv` (1,029 rows × 33 columns)

---

## Executive Summary

This document presents a comprehensive scientific and methodological audit of the existing data foundation, dataset construction pipeline, feature schemas, and spatial-temporal partitioning in the repository.

The headline finding is critical: **The current processed dataset (`ner_landslide_v1.csv`) CANNOT support a high-resolution (90 m / 30 m) static terrain susceptibility model in its present form.** 

While the dataset is structurally clean (14/14 code integrity gates pass), the underlying positive landslide inventory derived from the NASA Global Landslide Catalog (GLC) suffers from severe spatial coarse-grained location uncertainty (median location error of **10 km to 25 km**). Consequently, micro-topographic features derived from a 90 m DEM (such as `slope_deg`, `tri`, `twi`, `plan_curv`) exhibit zero univariate discrimination between landslide events and background terrain (**slope AUC = 0.507**). High tree-based model benchmark scores (e.g., XGBoost/RF AUC of 0.749 on terrain) are **artifacts of spatial fingerprinting/reporting bias**, where models learn regional location centroids rather than physical slope stability physics.

Conversely, coarse-grained daily rainfall features from IMD ($0.25^\circ \approx 28\text{ km}$) are spatial-resolution compatible with the location errors and show a genuine physical triggering signal (**3-day antecedent rainfall AUC = 0.623**).

---

## 1. Inventory of Repository Datasets & File Structure

### A. Raw Data Layer (`data/raw/`)
* **`Global_Landslide_Catalog_Export_rows.csv`** (8.48 MB, 11,041 global records): NASA Global Landslide Catalog (GLC). Contains media-reported global events (2007–2017) with dates, coordinates, location accuracy estimates (`location_accuracy`), trigger categories, and fatalities.
* **Copernicus DEM Rasters** (`Copernicus_DSM_COG_30_N*_E*_DEM.tif`, 41 COG tiles, ~200 MB): Copernicus GLO-90 DSM elevation tiles at 3 arc-second (~90 m grid pixel, 30 m source). Coverage spans lat 21°N–29°N, lon 87°E–97°E. (Note: Patchy spatial coverage with tile gaps).
* **IMD Daily Rainfall NetCDF** (`ind2006_rfp25.nc` to `ind2017_rfp25.nc` + `IMD_rain_0.25_2015.nc`, 13 files, ~305 MB): IMD $0.25^\circ \times 0.25^\circ$ (~28 km) daily gridded rainfall across India for 2006–2017. Note: `IMD_rain_0.25_2015.nc` is an exact file duplicate of `ind2015_rfp25.nc`.
* **OSM Road Geometries** (`roadpts_r*c*.csv` × 16 tiles, `ner_roads_band*.json`, `ner-osm-free.shp.zip`): OpenStreetMap road vertex extractions. Note: Only 5 of 16 tiles were successfully retrieved via Overpass API (covering lat 21.3°N–25.7°N), leaving northern NER states (lat 25.7°N–29.6°N) uncovered.
* **SRTM DEM Archives** (`srtm_54_07.zip` to `srtm_56_08.zip`, 6 files, ~233 MB): Legacy 90 m SRTM DEM tiles; superseded by Copernicus GLO-90.
* **Admin Boundaries** (`ne_admin1.geojson`, `india_districts.geojson`): Natural Earth 10 m state boundaries for the 8 NER states and GADM district boundaries.

### B. Processed & Interim Data Layer (`data/processed/`, `data/interim/`)
* **`data/processed/ner_landslide_v1.csv`** (248.5 KB, 1,029 rows × 33 columns): **The canonical modeling deliverable**. Synthesizes 341 positive landslide events and 688 background negative samples across 8 NER states (2007–2017), joined with DEM terrain attributes, IMD rainfall dynamics, admin units, and spatial split blocks.
* `data/processed/MOCK_ner_landslide_v0.csv`: Day-1 synthetic placeholder dataset (superseded by v1).

### C. GIS Vector Output Layer (`gis/`)
* **`gis/ner_landslide_events_v1.geojson`**: 341 point features representing positive landslide events with associated dates, rainfall totals, accuracy radii, and triggers.
* **`gis/ner_risk_grid_v1.geojson`**: 877 polygon grid cells ($0.05^\circ \times 0.05^\circ$) with aggregated event counts, mean terrain/rainfall metrics, and 4 risk severity tiers (Low, Moderate, High, Very High).

---

## 2. Construction Methodology of `ner_landslide_v1.csv`

The data processing pipeline is executed via `run_pipeline.py` across 6 modular stages:

```
NASA GLC CSV -> [src/p01_inventory.py] -> Positives (341)
Candidate Grid (0.02 deg) ---------------> Candidates (28k+)
                                              |
Copernicus GLO-90 -> [src/p02_dem.py] -------> Query Terrain
OSM Overpass ------> [src/p04_proximity.py] -> Proximity (Skipped due to gap)
                                              |
[src/p05_build.py] -> Admin Join + Neg Sampling + 25km Spatial Block Split
                                              |
IMD NetCDF 2006-17 -> [src/p03_rainfall.py] -> Antecedent Rainfall Windows
                                              |
[src/p06_finalize.py] -> Schema Order + 14 Integrity Checks -> ner_landslide_v1.csv
```

### A. Positive Sample Selection (`p01_inventory.py`)
1. Filters NASA GLC records to NER bounding box ($87.5^\circ\text{E} \le \text{lon} \le 97.5^\circ\text{E}$, $21.5^\circ\text{N} \le \text{lat} \le 29.6^\circ\text{N}$) and valid event dates (2007–2017).
2. Filters out non-rainfall triggers (`earthquake`, `volcano`, `dam_embankment_collapse`, `construction`, `mining`, `freeze_thaw`, `snowfall_snowmelt`, `no_apparent_trigger`, `other`, `unknown`).
3. Maps NASA `location_accuracy` text descriptions to explicit error radii (`loc_accuracy_km`) and confidence weights (`label_conf`):
   * `exact`: 0.1 km (conf = 1.00) — **4 events**
   * `1km`: 1.0 km (conf = 0.85) — **41 events**
   * `5km`: 5.0 km (conf = 0.60) — **101 events**
   * `10km`: 10.0 km (conf = 0.45) — **81 events**
   * `25km`: 25.0 km (conf = 0.30) — **79 events**
   * `50km`: 50.0 km (conf = 0.20) — **31 events**
   * `100km`: 100.0 km (conf = 0.10) — **4 events**
4. Restricts observations strictly to India (8 NER states: *Arunachal Pradesh, Assam, Manipur, Meghalaya, Mizoram, Nagaland, Sikkim, Tripura*). (Drops ~110 events in Bangladesh/Myanmar/Nepal/Bhutan and 78 Darjeeling events in West Bengal). Total positive events: **341**.

### B. Negative Sample Generation (`p05_build.py`)
1. Generates a regular candidate grid at $0.02^\circ \times 0.02^\circ$ resolution (~2.2 km grid cell spacing) across NER.
2. Filters candidates to points falling inside the 8 NER state boundaries and on valid DEM terrain pixels.
3. Excludes candidate points within a **500 m spatial buffer** of any positive landslide coordinate.
4. Samples candidates at a 2:1 ratio relative to positives (**688 negatives**).
5. **Slope Floor Strategy (`MIN_NEG_SLOPE = 0.0`)**: In early pipeline drafts (`sampling.py`), negatives were filtered to $\text{slope} \ge 8^\circ$ to prevent the trivial "flat floodplain is safe" shortcut. However, inside the 8 NER states, steep terrain candidate points were heavily drawn from ultra-steep Arunachal Pradesh ridges, making negatives *steeper* on average than media-reported positives ($22.5^\circ$ vs $13.3^\circ$). This inverted the slope-safety signal. Thus, the slope floor was removed, reverting to unconstrained random background sampling over NER land.
6. **Synthetic Event Date Imputation**: Each negative sample is assigned a random `event_date` sampled from the empirical date distribution of positive events. This ensures negative samples capture true monsoon daily rainfall dynamics rather than artificial dry days.
7. Negatives are assigned nominal metadata: `label_conf = 1.0`, `loc_accuracy_km = 0.05`, `trigger = "none"`, `fatalities = 0`.

### C. Terrain Feature Extraction (`p02_dem.py`)
* Mosaics 30 m Copernicus GLO-90 COG tiles on-the-fly into $3^\circ \times 3^\circ$ overlapping spatial blocks with a $0.05^\circ$ boundary buffer to eliminate border seams.
* Computes 9 terrain variables considering true latitude-dependent cell metric spacing ($dx = \text{cell} \times 111320 \times \cos(\text{lat})$, $dy = \text{cell} \times 110540$).

### D. Antecedent Rainfall Calculation (`p03_rainfall.py`)
* Loads 12 years (2006–2017) of IMD daily $0.25^\circ$ gridded NetCDF files.
* For every sample coordinate and `event_date`, computes strictly backwards-looking temporal accumulations (inclusive of event day):
  * `rain_1d`, `rain_3d`, `rain_7d`, `rain_15d`, `rain_30d` (mm)
  * Antecedent Precipitation Index: $\text{API} = \sum_{k=0}^{29} P_{t-k} \cdot 0.9^k$
  * `rain_annual_mean`: Climatological mean annual rainfall at the IMD grid cell over 2006–2017.

### E. Spatial Block Partitioning & Admin Join (`p05_build.py`)
* Coordinates are projected and partitioned into **$25\text{ km} \times 25\text{ km}$ spatial blocks** (`block_id`).
* Entire spatial blocks are randomly assigned to `train` (70%), `val` (15%), or `test` (15%) splits under random seed 42.

---

## 3. Dataset Schema & Feature Identification

One row of `ner_landslide_v1.csv` represents a **single spatio-temporal observation point in North-East India** defined by coordinates (`lon`, `lat`), administrative unit (`state`, `district`), and a calendar date (`event_date`), paired with 16 environmental/rainfall features and data quality flags.

### Target Variables
* **`label`** (Integer: `1` = landslide event, `0` = sampled background non-event): The primary binary classification target.
* **`label_conf`** (Float [0.05, 1.00]): Confidence weight derived from location precision.

### Feature Mapping Table

| Feature Name | Category | Theoretical Role | Source & Resolution | Observed Univariate AUC | Notes / Assessment |
|---|---|---|---|---|---|
| `elevation` | Static Terrain | Topographic height (m) | Copernicus GLO-90 (90 m) | 0.558 | Pos median 902 m vs Neg median 648 m. Weak signal. |
| `slope_deg` | Static Terrain | Slope gradient ($^\circ$) | Copernicus GLO-90 (90 m) | **0.507** | **Zero predictive power**. Pos median $13.4^\circ$ vs Neg median $13.5^\circ$. |
| `aspect_sin` | Static Terrain | Directional orientation ($\sin$) | Copernicus GLO-90 (90 m) | ~0.50 | Circular transformation of aspect angle. |
| `aspect_cos` | Static Terrain | Directional orientation ($\cos$) | Copernicus GLO-90 (90 m) | ~0.50 | Circular transformation of aspect angle. |
| `plan_curv` | Static Terrain | Planform curvature (flow divergence) | Copernicus GLO-90 (90 m) | ~0.50 | Micro-topography measure; random noise at 10 km error. |
| `prof_curv` | Static Terrain | Profile curvature (flow acceleration) | Copernicus GLO-90 (90 m) | ~0.50 | Micro-topography measure; random noise at 10 km error. |
| `tri` | Static Terrain | Terrain Ruggedness Index | Copernicus GLO-90 (90 m) | 0.506 | Micro-ruggedness; uninformative at town centroid. |
| `twi` | Static Terrain | Topographic Wetness Index | Copernicus GLO-90 (90 m) | 0.478 | Hydrological accumulation proxy; pure noise. |
| `relief_500m` | Static Terrain | Local elevation range in 500 m window | Copernicus GLO-90 (90 m) | 0.535 | Pos median 244 m vs Neg median 228 m. Minimal signal. |
| `rain_1d` | Dynamic Temporal | Event-day rainfall total (mm) | IMD Daily ($0.25^\circ \approx 28\text{ km}$) | 0.611 | Pos median 16.2 mm vs Neg median 8.5 mm. Physical trigger. |
| `rain_3d` | Dynamic Temporal | 3-day antecedent total (mm) | IMD Daily ($0.25^\circ \approx 28\text{ km}$) | **0.623** | **Strongest physical trigger**. Pos 49.3 mm vs Neg 30.0 mm. |
| `rain_7d` | Dynamic Temporal | 7-day antecedent total (mm) | IMD Daily ($0.25^\circ \approx 28\text{ km}$) | ~0.60 | Medium-term antecedent accumulation. |
| `rain_15d` | Dynamic Temporal | 15-day antecedent total (mm) | IMD Daily ($0.25^\circ \approx 28\text{ km}$) | ~0.59 | Long-term soil saturation accumulation. |
| `rain_30d` | Dynamic Temporal | 30-day antecedent total (mm) | IMD Daily ($0.25^\circ \approx 28\text{ km}$) | ~0.58 | Seasonal monsoon background. |
| `api` | Dynamic Temporal | Antecedent Precipitation Index | IMD Daily ($0.25^\circ \approx 28\text{ km}$) | 0.576 | Exponential decay ($\gamma=0.9$) moisture memory index. |
| `rain_annual_mean` | Static Climatological | Mean annual rainfall (mm) | IMD Daily ($0.25^\circ \approx 28\text{ km}$) | ~0.54 | Regional climatological baseline. |
| `dist_road_m` | Static Proximity | Distance to nearest road (m) | OpenStreetMap Vector | **N/A** | **OMITTED in v1** due to 11 missing OSM tiles north of 25.7°N. |
| `dist_major_road_m`| Static Proximity | Distance to highway (m) | OpenStreetMap Vector | **N/A** | **OMITTED in v1** due to missing OSM tiles. |
| `dist_stream_m` | Static Proximity | Distance to nearest river (m) | OpenStreetMap Vector | **N/A** | **OMITTED in v1** due to missing OSM tiles. |

### Metadata & Auxiliary Columns
`sample_id`, `lon`, `lat`, `state`, `district`, `event_date`, `trigger`, `size`, `category`, `fatalities`, `source`, `hq_location`, `block_id`, `split`.

---

## 4. In-Depth Scientific & Methodological Audit

### A. Spatial Resolution Mismatch & Location Error (The Central Flaw)
* **The Physics vs Data Discrepancy**: Landslides occur on specific steep slope faces (typically $20^\circ - 45^\circ$) covering areas of tens to hundreds of square meters. Slope stability physics calculated at 90 m resolution requires sub-100 m coordinate accuracy.
* **The Inventory Reality**: Out of 341 positive events, **only 4 events (1.2%) are "exact"** ($\le 100\text{ m}$) and **41 events (12.0%) have 1 km accuracy**. Over **86.8% of events have location errors of 5 km, 10 km, 25 km, or 50 km**.
* **Source Mechanism**: NASA GLC points are constructed from media reports (e.g., *"Landslide hits highway near Kohima"*). Web scripters pin the event to the nearest town centroid or district headquarter.
* **Impact on Static Terrain Modeling**: At a 10 km displacement, the recorded point (a valley town or district office) shares zero terrain characteristics with the real slope scar. Hence, `slope_deg` AUC is **0.507** (identical to flipping a fair coin). **High-resolution terrain features in this dataset are effectively random noise.**

### B. Reporting Bias & Spatial Fingerprinting Leakage
* **The Apparent Paradox**: Benchmarking a Gradient Boosted Decision Tree (XGBoost/RF) on terrain features alone yields a test AUC of **0.749**, even though every single constituent terrain feature has a univariate AUC of $\approx 0.50$.
* **Mechanism of Leakage**: Tree models do not learn slope physics here. Instead, decision trees combine multivariable terrain signatures (`elevation` + `relief_500m` + `twi` + `aspect`) to construct an implicit spatial look-up table ("fingerprint") identifying specific hilly districts where news reporting is active.
* **Reporting Bias**: News reports cluster near populated towns and major highways. The model learns to detect **populated reporting clusters**, NOT slope instability. Quoting a 0.749 AUC as "susceptibility performance" is scientifically invalid.

### C. Conflation of Static Susceptibility vs. Temporal Trigger Modeling
* Landslide hazard modeling consists of two distinct components:
  1. **Static Susceptibility Map**: $P(\text{failure} \mid \text{slope, geology, soil, landcover})$, constant over time.
  2. **Dynamic Early Warning System**: $P(\text{failure on day } t \mid \text{antecedent rain } P_{t}, \text{susceptibility})$.
* `ner_landslide_v1.csv` combines static terrain and dynamic rainfall into a single table. Because static terrain features are corrupted by location noise, any model trained on this combined table will rely almost entirely on rainfall dynamics (`rain_3d`), failing completely as a spatial susceptibility map.

### D. Negative Sampling Deficiencies & PU Learning Dynamics
* **Unlabeled Background Negatives**: Negatives are randomly sampled candidate grid points outside a 500 m buffer of positive coordinates. They are unconfirmed (Positive-Unlabelled problem). A steep negative point on a heavy monsoon day may represent an unreported landslide.
* **Artificial Quality Inversion Leakage**:
  * Negatives are assigned `loc_accuracy_km = 0.05` and `label_conf = 1.00`.
  * Positives have median `loc_accuracy_km = 10.0` and `label_conf = 0.45`.
  * If metadata columns (`loc_accuracy_km`, `label_conf`, `hq_location`) are inadvertently included in training, or if loss functions weight samples by `label_conf` without care, the model will learn that high confidence implies non-landslide status.

### E. Missing Essential Predictors
The v1 dataset completely lacks several standard static susceptibility drivers due to portal/credential constraints:
* **Lithology & Geology**: Rock types, weathering resistance (GSI Bhukosh).
* **Structural Lineaments**: Distance to geological faults (`dist_fault_m`).
* **Land Cover / Vegetation**: ESA WorldCover 10 m / Sentinel-2 NDVI.
* **Soil Attributes**: Soil depth, clay content (SoilGrids 250 m).
* **Anthropogenic Disturbance**: Distance to roads (`dist_road_m`) — omitted in v1 due to OSM tile gaps.

### F. Spatial Partitioning & Split Leakage Audit
* **Block Disjointness**: Split assignment (`train`/`val`/`test`) is grouped by 25 km $\times$ 25 km spatial blocks (`block_id`). 
* **Leakage Verification**: Integrity gate check #11 confirms **0 blocks span multiple splits**. Nearby points within 25 km stay within the same split, preventing spatial auto-correlation leakage between train and test.
* **Class Imbalance Across Splits**:
  * `train`: 721 rows (243 pos / 478 neg, base rate = 33.7%)
  * `val`: 136 rows (46 pos / 90 neg, base rate = 33.8%)
  * `test`: 172 rows (51 pos / 121 neg, base rate = 29.7%)

---

## 5. Model Suitability Assessment

| Model Architecture | Can it be trained on current `v1.csv`? | Expected Performance & Validity | Recommendation |
|---|---|---|---|
| **Logistic Regression** | **Yes** (Runs cleanly) | **Test AUC ~ 0.60 (Rainfall-driven)**. Cannot fit complex spatial fingerprints, so terrain weights collapse near zero, correctly reflecting lack of spatial signal. | **Recommended as linear baseline**, but only when restricted to rainfall features or coarse admin units. |
| **Random Forest** | **Yes** (Runs cleanly) | **Test AUC ~ 0.73 (Overfitted)**. Learns high-dimensional spatial fingerprints of town centroids from noisy 90 m terrain. High AUC is misleading. | **Not recommended on 90 m terrain** until inventory location accuracy is improved or aggregated. |
| **XGBoost / Gradient Boosting** | **Yes** (Runs cleanly) | **Test AUC ~ 0.75 (Overfitted)**. Severely exploits terrain combinations to memorize district reporting bias. | **Not recommended on 90 m terrain**. Highly prone to spatial shortcut learning on this inventory. |

---

## Concise Summary & Direct Answers

### A. What Data Is Available
1. **Landslide Inventory**: NASA GLC global catalogue filtered to NER (341 positive rainfall-triggered events, 2007–2017). Median location accuracy is 10–25 km (only 45 events $\le 1\text{ km}$).
2. **DEM Terrain**: Copernicus GLO-90 DSM (30 m pixel / 90 m grid, 41 tiles). 9 terrain metrics extracted.
3. **Rainfall**: IMD daily gridded rainfall NetCDF ($0.25^\circ \approx 28\text{ km}$, 2006–2017). 7 antecedent metrics extracted.
4. **Negatives**: 688 sampled background points on NER land with synthetic dates.
5. **Admin**: Natural Earth 8 NER states + GADM India district boundaries.

### B. What Can Be Used for Static Susceptibility
* **At 90 m pixel resolution: NOTHING from the current open inventory.** The recorded points are 10–25 km off the actual failure scars, making 90 m DEM slope/TRI/TWI features pure random noise (slope AUC = 0.507).
* **At coarse District / $0.25^\circ$ cell resolution**: Mean relief, mean elevation, and climatological annual rainfall (`rain_annual_mean`) can serve as regional static susceptibility proxies.

### C. What Belongs to Temporal Modelling
* **All daily IMD rainfall accumulations**: `rain_1d`, `rain_3d` (strongest trigger, AUC = 0.623), `rain_7d`, `rain_15d`, `rain_30d`, and `api` (Antecedent Precipitation Index).
* Temporal models must predict **District $\times$ Day** or **$0.25^\circ\text{ Grid} \times \text{Day}$** landslide probability given daily rainfall forecasts.

### D. Major Data-Quality Problems
1. **Severe Location Uncertainty**: 86.8% of positive events have 5–50 km location errors.
2. **Reporting Bias**: Landslides are recorded at towns/highways, causing models to fingerprint population centroids rather than slope physics.
3. **Missing Road Proximity**: `dist_road_m` is completely missing due to 11 unretrieved OSM tiles north of 25.7°N.
4. **Confidence Inversion**: Negatives are assigned synthetic 50 m accuracy (`loc_accuracy_km = 0.05`) and `label_conf = 1.0`, while real positives have low confidence.

### E. Potential Leakage Problems
1. **Spatial Fingerprinting Leakage**: Tree models (RF/XGBoost) memorize multi-variable terrain signatures to identify reporting towns, inflating test AUC to 0.749 despite zero slope signal.
2. **Metadata Leakage**: Including `loc_accuracy_km`, `label_conf`, or `hq_location` in feature sets will cause direct target leakage.

### F. What Should be Changed Before Model Training
1. **Shift Modeling Resolution**: Abandon 90 m pixel-level static susceptibility on NASA GLC data. Train models at **District $\times$ Day** or **$0.25^\circ \times \text{Day}$** scale.
2. **Integrate GSI Bhukosh Inventory**: Obtain surveyed landslide polygons from Geological Survey of India (Bhukosh/NGDR), which have meter-level spatial precision.
3. **Complete OSM Road Coverage**: Fetch remaining 11 OSM tiles to enable `dist_road_m`.
4. **Separate Model Pipelines**: Build a **Dynamic Rainfall Threshold Model** (temporal trigger) completely separate from the **Static Susceptibility Layer**.

### G. Whether Current Data is Sufficient for LR, Random Forest, and XGBoost
* **For a 90 m Terrain Susceptibility Model: NO.** None of the algorithms (LR, RF, XGBoost) can extract real slope physics from 10 km misplaced labels. RF and XGBoost will yield deceptively high AUCs (~0.75) by memorizing regional location fingerprints.
* **For a Regional Temporal Early-Warning Model (Rainfall-driven at District scale): YES.** LR, RF, and XGBoost can all be validly trained using `rain_1d` through `rain_30d` and `api` to predict rainfall-triggered landslide risk windows.
