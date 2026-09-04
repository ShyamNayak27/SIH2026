Yep — **this README is now stale** relative to what we've actually built.

The biggest problem is this line:

> “Read it before writing any modelling code.”

We've now got a working spatial model, temporal model, fusion engine, FastAPI API, and dashboard. So the README should evolve from **“Milestone 1 — Data Foundation”** into the current **TerraWatch / Landslide Risk Intelligence prototype** documentation.

I would **not delete the old data-foundation findings** though. They are valuable because they explain why the modelling approach was constrained.

## Replace the current README with this

```markdown
# TerraWatch — Landslide Risk Intelligence

An interpretable landslide risk assessment system for the Northeast Region
of India (NER), combining static terrain susceptibility with rainfall-driven
temporal risk.

> **Current status:** Working spatial + temporal risk intelligence prototype.
> Remote-sensing/vision integration is in progress.

---

## Overview

Landslide risk is influenced by both **where the terrain is susceptible** and
**when environmental conditions become hazardous**.

TerraWatch combines these two signals:

- **Spatial model** — estimates terrain-based landslide susceptibility using
  terrain and rainfall features.
- **Temporal model** — estimates rainfall-driven risk from recent and
  antecedent precipitation patterns.
- **Risk fusion engine** — combines available model outputs into a single
  normalized risk score.
- **Explainability layer** — identifies the features contributing most to the
  spatial model prediction.
- **FastAPI backend** — exposes the risk assessment pipeline through an API.
- **Interactive dashboard** — allows users to inspect risk across selected
  NER locations.

The system is designed as a **decision-support prototype**, not as a
production autonomous warning system.

---

## System Architecture

```text
                    NER LANDSLIDE DATA
                           │
             ┌─────────────┴─────────────┐
             │                           │
             ▼                           ▼
      SPATIAL MODEL               TEMPORAL MODEL
    Random Forest RF          Logistic Regression
             │                           │
             │ Spatial Risk              │ Temporal Risk
             └─────────────┬─────────────┘
                           ▼
                    RISK FUSION ENGINE
                           │
                           ▼
                     DECISION ENGINE
                           │
                  ┌────────┴────────┐
                  ▼                 ▼
             Explainability      Alert Level
               / SHAP              + Action
                  │                 │
                  └────────┬────────┘
                           ▼
                       FASTAPI
                           │
                           ▼
                    WEB DASHBOARD

       Remote-sensing / vision model
                 ── IN PROGRESS ──
```

When the vision model becomes available, the fusion engine is designed to
incorporate it as an additional risk component.

---

# 1. Data Foundation

The project is built around a landslide inventory covering the Northeast
Region of India.

### Dataset

- **1,029 samples**
- **341 landslide events**
- **688 sampled non-landslide locations**
- **8 NER states**
- Event period: **2007–2017**
- Terrain features derived from a **90 m DEM**
- Rainfall features derived from **IMD daily rainfall data**
- Spatial block identifier for leakage-aware evaluation

The current modelling dataset is:

```text
data/processed/ner_landslide_v1.csv
```

### Core feature groups

#### Terrain

- elevation
- slope
- aspect
- plan curvature
- profile curvature
- terrain ruggedness
- topographic wetness index
- relief

#### Rainfall / dynamic conditions

- 1-day rainfall
- 3-day rainfall
- 7-day rainfall
- 15-day rainfall
- 30-day rainfall
- antecedent precipitation index
- annual mean rainfall

---

# 2. Spatial Risk Model

The spatial component uses a **Random Forest classifier** trained on terrain
and rainfall features.

The model produces a class-1 probability that is used as the spatial
landslide risk signal.

### Held-out test performance

| Metric | Spatial Random Forest |
|---|---:|
| Accuracy | 67.44% |
| Precision | 42.42% |
| Recall | 27.45% |
| F1 | 33.33% |
| **ROC-AUC** | **0.6971** |

The held-out ROC-AUC indicates moderate discrimination, but the relatively
low recall means the current spatial model should **not** be treated as a
standalone warning system.

### Explainability

The spatial model is accompanied by SHAP-based feature attribution.

The dashboard exposes the strongest contributing factors for each selected
location, including examples such as:

- Terrain ruggedness
- Local terrain relief
- Profile curvature
- Elevation
- Recent rainfall
- Historical annual rainfall

The dashboard distinguishes between factors that push the model toward higher
or lower risk.

---

# 3. Temporal Rainfall Risk Model

The temporal component models rainfall-driven landslide risk independently
from the static spatial model.

It uses a Logistic Regression pipeline with standardized rainfall-derived
features:

```text
log_rain_1d
log_rain_3d
log_rain_7d
log_api
rain_3d_ratio_30d
rainfall_acceleration
```

The temporal model produces a risk score on a **0–100 scale**.

### Held-out test performance

| Metric | Temporal Logistic Regression |
|---|---:|
| Precision | 46.34% |
| Recall | 74.51% |
| F1 | 57.14% |
| PR-AUC | 0.5092 |
| **ROC-AUC** | **0.7300** |

The relatively high recall is useful for a rainfall-driven screening component,
where identifying potentially hazardous conditions is important.

### Temporal signals

The system derives interpretable rainfall signals including:

- High 24-hour rainfall
- Strong 72-hour accumulation
- Recent rainfall concentration
- Recent rainfall intensity
- Elevated antecedent wetness
- Rainfall acceleration

---

# 4. Risk Fusion

Spatial and temporal models provide complementary information:

```text
Spatial risk  → WHERE susceptibility exists
Temporal risk → WHEN conditions are becoming hazardous
```

The fusion engine combines the available model outputs.

Configured weights:

```text
Spatial  = 45%
Temporal = 35%
Vision   = 20%
```

When a model is unavailable, its weight is removed and the remaining weights
are normalized.

For the current spatial + temporal prototype:

```text
Spatial contribution  = 45 / (45 + 35) = 56.25%
Temporal contribution = 35 / (45 + 35) = 43.75%
```

The resulting score is converted into an alert level:

```text
LOW
MODERATE
HIGH
```

The decision engine also produces an action recommendation, for example:

```text
HIGH
High landslide risk detected. Prepare preventive response measures.
```

or:

```text
MODERATE
Elevated risk detected. Increase monitoring frequency.
```

---

# 5. Explainability

TerraWatch does not only return a risk number.

For each assessment, the API provides:

- Spatial model probability
- Spatial risk score
- Temporal risk score
- Final fused risk
- Risk-increasing factors
- Risk-decreasing factors
- Primary risk driver
- Primary risk reducer
- Recommended action

This allows the user to understand:

> **What is the risk?**

and

> **Why did the system produce this assessment?**

---

# 6. FastAPI Backend

The risk engine is exposed through FastAPI.

### Start the API

From the repository root:

```bash
uvicorn src.api.server:app --reload --port 8000
```

API documentation is available at:

```text
http://127.0.0.1:8000/docs
```

### Risk endpoint

```text
GET /api/risk/{sample_id}
```

Example:

```bash
curl http://127.0.0.1:8000/api/risk/5832ea6720e739c3
```

The response contains the location, model outputs, fused decision,
explainability information, observed rainfall conditions, and model status.

---

# 7. Interactive Dashboard

The frontend provides an interactive risk intelligence dashboard.

### Start the frontend

```bash
npm install
npm run dev
```

The dashboard runs locally at:

```text
http://localhost:5173/
```

### Current dashboard capabilities

Users can select locations and inspect:

- Current fused risk
- Risk level
- Coordinates
- 24-hour rainfall
- 72-hour rainfall
- 7-day rainfall
- 30-day rainfall
- Temporal rainfall risk
- Temporal risk drivers
- Spatial model risk
- SHAP-based model factors
- Recommended action
- Model connectivity status

### Currently supported demonstration locations

- North Sikkim, Sikkim
- West Kameng, Arunachal Pradesh
- North Cachar Hills, Assam
- Ukhrul, Manipur
- East Khasi Hills, Meghalaya
- Lawngtlai, Mizoram
- Kohima, Nagaland
- South Tripura, Tripura

---

# 8. Example Assessments

The current API produces differentiated assessments rather than a fixed
hard-coded score.

| Location | Spatial | Temporal | Final | Alert |
|---|---:|---:|---:|---|
| North Sikkim | 61 | 55 | **58** | HIGH |
| West Kameng | 82 | 46 | **66** | HIGH |
| Lawngtlai | 45 | 47 | **46** | MODERATE |
| North Cachar Hills | 28 | 43 | **34** | MODERATE |

The underlying drivers also change between locations.

For example, West Kameng is strongly influenced by local terrain relief and
terrain ruggedness, while Lawngtlai has substantially lower spatial risk.

---

# 9. Remote-Sensing / Vision Component

The repository contains the initial remote-sensing feature extraction work.

The vision module currently includes:

```text
src/models/vision/
├── feasibility_check.py
├── extract_cv_features.py
├── locations.csv
├── locations_all.csv
└── __init__.py
```

The extractor is designed to obtain remote-sensing features such as:

- Sentinel-2 NDVI
- SRTM elevation
- SRTM slope
- ESA WorldCover land-cover information

### Current status

**Vision model integration is pending.**

No fabricated vision risk score is used by the current dashboard.

The API explicitly reports:

```json
"model_status": {
    "spatial": "connected",
    "temporal": "connected",
    "vision": "pending"
}
```

This keeps the current risk assessment transparent about which model
components are actually active.

---

# 10. Validation and Methodological Notes

The project initially established a data-quality and spatial-validation
foundation before model development.

Important considerations include:

- Explicit train / validation / test splits
- Spatial block identifiers
- Leakage-aware evaluation
- Separate assessment of terrain and rainfall signals
- Held-out test evaluation
- Feature-level explainability

The initial data findings also showed that individual terrain variables should
not be interpreted as independent evidence of strong physical predictive
power. Therefore, model performance is reported using held-out evaluation
rather than relying on individual feature correlations.

The earlier exploratory benchmark is retained in:

```text
reports/spatial_benchmark_results.csv
```

and the feature-importance analysis is retained in:

```text
reports/feature_importances.csv
```

---

# 11. Project Structure

```text
├── app.js
├── index.html
├── styles.css
├── package.json
│
├── data/
│   ├── raw/
│   ├── interim/
│   └── processed/
│       ├── ner_landslide_v1.csv
│       └── temporal_risk_predictions.csv
│
├── models/
│   ├── landslide_model.pkl
│   └── temporal_risk_model.joblib
│
├── reports/
│   ├── feature_importances.csv
│   └── spatial_benchmark_results.csv
│
├── src/
│   ├── api/
│   │   └── server.py
│   ├── explainability/
│   ├── integration/
│   │   └── decision_engine.py
│   ├── risk_engine/
│   │   └── fusion.py
│   ├── temporal/
│   │   ├── features.py
│   │   ├── model.py
│   │   ├── predict.py
│   │   └── evaluate.py
│   └── models/
│       └── vision/
│
├── gis/
│
└── docs/
    ├── FINDINGS.md
    ├── PROVENANCE.md
    └── DATA_SOURCES.md
```

---

# 12. Local Development

### Backend

```bash
source .venv/bin/activate

uvicorn src.api.server:app --reload --port 8000
```

### Frontend

In another terminal:

```bash
npm install
npm run dev
```

Then open:

```text
http://localhost:5173/
```

Both services must be running for the dashboard to receive live risk
assessments.

---

# 13. Limitations

The current system is a **research / hackathon prototype** and should not be
used as an operational landslide warning system.

Current limitations include:

1. The spatial model has moderate held-out discrimination and relatively low
   recall.
2. The training inventory is limited in size and geographic coverage.
3. Current rainfall features are based on the available historical dataset;
   the dashboard does not claim to provide a live meteorological forecast.
4. Remote-sensing / vision integration is not yet connected to the fused risk.
5. Risk scores represent model-based decision support rather than deterministic
   predictions of landslide occurrence.
6. Additional ground-truth inventories and higher-resolution terrain data are
   required for production-grade deployment.

---

# 14. Future Work

### Short term

- Complete remote-sensing feature extraction
- Train and validate the vision model
- Connect the vision component to the fusion engine
- Improve spatial recall and calibration
- Expand validation across additional geographic blocks

### Longer term

- Integrate live rainfall feeds
- Introduce higher-resolution terrain data
- Incorporate verified GSI landslide inventories
- Add temporal forecasting rather than retrospective rainfall assessment
- Deploy the API and dashboard for field testing
- Add alert history and spatial monitoring across the full NER

---

# 15. Current Milestone

### Milestone 1 — Data Foundation

Completed:

- NER landslide inventory
- Terrain feature generation
- Rainfall feature generation
- Spatial-block dataset construction
- Data provenance and findings

### Milestone 2 — Risk Intelligence Prototype

Completed:

- Spatial Random Forest
- Temporal rainfall model
- Risk fusion
- Decision engine
- SHAP-based explanations
- FastAPI backend
- Interactive dashboard
- Multi-location risk assessment

### Milestone 3 — Multimodal Risk Intelligence

In progress:

- Remote-sensing / vision model
- Full three-model fusion
- Higher-resolution spatial modelling
- Expanded validation

---

## Key Takeaway

TerraWatch separates two questions that a single static susceptibility model
cannot answer well:

> **Where is the terrain vulnerable?**

and

> **Are rainfall conditions currently increasing concern?**

By combining spatial susceptibility, temporal rainfall signals, explainability,
and an actionable decision layer, the prototype provides a clearer and more
transparent risk assessment workflow.

**Spatial + Temporal + Explainability → Actionable Risk Intelligence**
```
