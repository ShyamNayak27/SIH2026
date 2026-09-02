# NER landslide platform — data sources, ranked by "can we actually get it in 4 days"

Scope: 8 states, bbox 87.5–97.5°E, 21.5–29.6°N. Canonical CRS EPSG:4326; all
metric work in EPSG:32646 (UTM 46N).

## Tier 1 — get these on day 1, everything else is optional

| Layer | Source | Access | Res | Notes |
|---|---|---|---|---|
| **Landslide inventory (primary)** | GSI Bhukosh landslide layer | Portal, toposheet-wise shapefile export, free, no login for public layers | point + polygon | ~110k events nationally; the same inventory the 2025 national susceptibility paper used. Bhukosh is flaky — NGDR portal is the fallback mirror. |
| **Landslide inventory (backup)** | NASA Global Landslide Catalog / COOLR | Direct CSV, `data.nasa.gov` legacy export + the COOLR viewer export | point | Global, media-derived, coarse location accuracy. Use as *supplementary positives with `label_conf=0.4`*, never as the only inventory. |
| **DEM** | Copernicus GLO-30 | OpenTopography Global DEM API (**free key needed**, 200 calls/day academic) or AWS `copernicus-dem-30m` public bucket | 30 m | Preferred over SRTM: newer, fewer voids in NE hills. `SRTM_GL1` / `AW3D30` / `NASADEM` are drop-in alternates on the same API. |
| **Rainfall (historical)** | IMD Pune gridded rainfall, 0.25°, daily, 1901–present | `imdpune.gov.in/cmpg/Griddata` NetCDF/binary, free; `imdlib` Python package wraps it | 0.25° | The defensible, judge-proof Indian source. Use for antecedent rainfall features + climatology. |
| **Rainfall (real-time)** | IMD merged satellite–gauge (GPM) daily, 0.25° | `imdpune.gov.in/cmpg/Realtimedata/gpm` | 0.25° | This is what the *live* system polls. Same grid as the training data — no domain shift. |
| **Roads** | OpenStreetMap (Geofabrik NE India extract) | Free download | vector | `dist_road_m` is the hill-cutting proxy and, in NER, one of the strongest features. Also gives you road-connectivity status for the dashboard. |
| **Admin boundaries** | LGD / Survey of India district + village | Free | vector | Needed for "which villages are cut off" and for district-wise alerting. |

## Tier 2 — add if time allows

| Layer | Source | Notes |
|---|---|---|
| Lithology, faults, geomorphology | GSI Bhukosh / NGDR seamless geoscientific map | Strong static predictors; shapefile export is toposheet-wise and tedious. |
| Land cover | ESA WorldCover 10 m (free, AWS) | Resample to 30 m majority class. |
| Soil moisture | SMAP L4 (NASA Earthdata login) or ERA5-Land (CDS login) | ERA5-Land is easier and hourly; SMAP is the better physical variable. |
| NDVI | Sentinel-2 L2A via Copernicus Data Space or Earth Engine | Pre-event median; vegetation loss is also a *detection* signal. |
| Soil texture | SoilGrids 250 m (free) or NBSS-LUP | SoilGrids needs no login. |
| Landslide Atlas of India (NRSC/ISRO, 2023) | PDF on nrsc.gov.in | Not machine-readable, but it is the citation that makes the risk framing credible in the pitch. Bhuvan also hosts an event-based inventory viewer. |

## For Neil's CV question (can we do imagery in 4 days?)

Don't try to train a segmentation model on raw Sentinel-2 in 4 days. Two
realistic options:

1. **Landslide4Sense** — a ready benchmark: 3,799 labelled Sentinel-2 + DEM
   patches (128×128, 14 bands), public, with baseline U-Net code on GitHub.
   Fine-tuning this is a 1-day job and gives a working "detect landslide scars
   from imagery" demo. This is the recommended path.
2. Skip pixel-level CV entirely; use Sentinel-2 only to derive NDVI as a
   *feature* in the tabular model. Cheap, defensible, no GPU.

The citizen-upload feature (crack photos) is a **separate, much smaller** CV
problem — a binary "is this a crack / slope failure / blocked road" classifier
on a few hundred scraped + hand-labelled images. That is the one worth building
fresh, because it's the demo judges will actually click on.

## Known traps

- **Bhukosh downloads are toposheet-wise.** Budget 2–3 hours of clicking, or
  script it. Have the NASA COOLR CSV as the same-day fallback so nobody is
  blocked.
- **Landslide date accuracy.** Most GSI polygons have no reliable date. Points
  without a date can only be used for *susceptibility* (static), not for
  *rainfall-triggered prediction* (temporal). Split the inventory into these two
  populations explicitly — this is the question Rhea needs answered on day 1,
  and the answer determines whether we can promise a prediction horizon at all.
- **Random train/test splits leak.** Landslides cluster; a random split puts a
  point 200 m from its own neighbour in the test set and you get a fake AUC of
  0.98. Split by spatial blocks (`sampling.py`).
- **Uniform negative sampling is worse than useless.** Negatives drawn over all
  of NER land on the Brahmaputra floodplain; the model learns "flat = safe" and
  cannot rank one hillside against another. Sample negatives from slope ≥ 8°
  outside a 500 m buffer around positives.
- **Never compute slope in degrees-CRS.** Reproject to UTM 46N first.

## Sources
- GSI Bhukosh portal — https://bhukosh.gsi.gov.in/Bhukosh/Public
- NASA COOLR — https://gpm.nasa.gov/applications/landslides/coolr
- Global Landslide Catalog export — https://catalog.data.gov/dataset/global-landslide-catalog-export
- OpenTopography global DEM API — https://opentopography.org/developers
- IMD gridded rainfall — https://www.imdpune.gov.in/cmpg/Griddata/Rainfall_25_NetCDF.html
- IMD real-time merged GPM rainfall — https://imdpune.gov.in/cmpg/Realtimedata/gpm/Rain_Download.html
- Landslide Atlas of India (NRSC/ISRO) — https://www.nrsc.gov.in/nrscnew/resources_atlas_landslide.php
- Bhuvan event-based landslide inventory — https://bhuvan-app1.nrsc.gov.in/disaster/usrtasks/landslide/landslide.php
- Landslide4Sense benchmark — https://arxiv.org/abs/2206.00515 · https://github.com/iarai/Landslide4Sense-2022
- National-scale landslide susceptibility of India (Sci Rep 2025) — https://www.nature.com/articles/s41598-025-33446-0
