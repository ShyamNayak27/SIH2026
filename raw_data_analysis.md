# Raw Data Folder Analysis — NER Landslide Project

**Total size:** ~1.25 GB across **89 files**

---

## Overview of Data Sources

| Layer | Files | Format | Size | Source |
|---|---|---|---|---|
| Landslide Inventory (NER) | `ner_landslide_v1.csv` | CSV | 248 KB | NER-specific, derived from NASA GLC |
| Landslide Inventory (Global) | `Global_Landslide_Catalog_Export_rows.csv` | CSV | 8.1 MB | NASA Global Landslide Catalog |
| Rainfall (IMD Daily) | `ind2006–2017_rfp25.nc` × 13 files | NetCDF | ~305 MB | IMD 0.25° gridded daily rainfall |
| DEM Tiles (Copernicus) | `Copernicus_DSM_COG_30_N*_E*_DEM.tif` × 41 tiles | GeoTIFF (COG) | ~200 MB | Copernicus GLO-90 (30 m) |
| OSM Road Points | `roadpts_r{row}c{col}.csv` × 16 files | CSV | ~100 MB | OpenStreetMap Overpass vertices |
| OSM Road Geometries | `ner_roads_band*.json` × 6 files | GeoJSON | ~140 MB | OpenStreetMap |
| OSM Roads (full geom) | `roads_r0c0.json` | GeoJSON | 31 MB | OpenStreetMap |
| OSM Shapefile | `ner-osm-free.shp.zip` | Zipped Shapefile | 289 MB | OSM full extract |
| SRTM DEM (legacy) | `srtm_5{4,5,6}_0{7,8}.zip` × 6 files | Zipped SRTM | ~233 MB | SRTM 90 m (older, superseded by Copernicus) |
| Real Landslide (zipped) | `ner-landslide-v1-real-data.zip` | ZIP | 165 KB | NER real data bundle |
| Road Raster | `roadpts_r0c.tif` | GeoTIFF | 513 KB | Road proximity raster |
| Road band metadata | `ner_roads_band0,1,4.json` | JSON | < 1 KB each | Band metadata (mostly empty) |

---

## 1. Landslide Inventory

### `ner_landslide_v1.csv` — **1,029 records** (project's primary labeled dataset)

**Schema (32 columns):**

| Column | Type | Description |
|---|---|---|
| `sample_id` | string | Unique hex ID |
| `lon`, `lat` | float | WGS84 coordinates |
| `state`, `district` | string | Admin units |
| `event_date` | date | Landslide occurrence date |
| `elevation` | float | metres (from DEM) |
| `slope_deg` | float | degrees |
| `aspect_sin`, `aspect_cos` | float | Circular-encoded aspect |
| `plan_curv`, `prof_curv` | float | Curvature metrics |
| `tri` | float | Terrain Ruggedness Index |
| `twi` | float | Topographic Wetness Index |
| `relief_500m` | float | Local relief in 500 m radius |
| `rain_1d`, `rain_3d`, `rain_7d`, `rain_15d`, `rain_30d` | float | Multi-day rainfall accumulations (mm) |
| `api` | float | Antecedent Precipitation Index |
| `rain_annual_mean` | float | Climatological mean (mm) |
| `label` | 0/1 | Landslide (1) / Non-event (0) |
| `label_conf` | float | Confidence score (0–1) |
| `loc_accuracy_km` | float | Location accuracy radius |
| `trigger` | string | e.g., `downpour`, `rain` |
| `size` | string | `small`, `medium`, `large` |
| `category` | string | `landslide`, `mudslide`, etc. |
| `fatalities` | int | Death count |
| `source` | string | Data origin |
| `hq_location` | 0/1 | Whether point is HQ-assigned |
| `block_id` | int | Spatial block for CV |
| `split` | string | `train` / `test` |

> [!IMPORTANT]
> This is the **gold standard labeled dataset** — 1,029 rows covering NER states (Sikkim, Manipur, etc.) from 2006–2017. The `split` column and `block_id` indicate spatial cross-validation is already designed in.

### `Global_Landslide_Catalog_Export_rows.csv` — **11,041 records** (upstream global source)

**Schema (31 columns):** Standard NASA GLC fields — `event_id`, `event_date`, `landslide_category`, `landslide_trigger`, `landslide_size`, `fatality_count`, `longitude`, `latitude`, `country_code`, etc.

> [!NOTE]
> The NER-specific CSV is a filtered + feature-engineered subset of this global catalog, merged with IMD rainfall and DEM terrain features.

---

## 2. Rainfall Data (IMD)

**Files:** `IMD_rain_0.25_2015.nc` + `ind2006_rfp25.nc` … `ind2017_rfp25.nc` (13 NetCDF files)  
**Size per file:** ~24.2 MB | **Total:** ~305 MB  
**Resolution:** 0.25° × 0.25° grid (~28 km), daily  
**Coverage:** 2006–2017 (12 years), full India grid

> [!NOTE]
> `IMD_rain_0.25_2015.nc` overlaps with `ind2015_rfp25.nc` — likely the same data under a different naming convention. Verify before processing to avoid duplication.

---

## 3. Digital Elevation Model (Copernicus GLO-90)

**Files:** 41 GeoTIFF tiles (COG format)  
**Naming:** `Copernicus_DSM_COG_30_N{lat}_00_E{lon}_00_DEM.tif`  
**Resolution:** 30 m (COG = Cloud-Optimised GeoTIFF, efficient for tiled reads)  
**Size per tile:** 630 KB – 6 MB (varies by terrain complexity)  
**Geographic coverage:**

| Latitude band | Longitude range |
|---|---|
| N21 | E087–E090 |
| N22 | E088, E092 |
| N23 | E088, E091–E094 |
| N24 | E088, E091–E094 |
| N25 | E088–E094 |
| N26 | E088, E091–E095 |
| N27 | E088, E090–E096 |
| N28 | E094–E096 |
| N29 | E097 |

> [!WARNING]
> Coverage is **patchy** — several expected tiles are missing (e.g., N21 E091–E097, N22 E089–E091). Confirm whether these are ocean/flat areas (no need) or actual gaps before running the DEM pipeline.

**Also present:** 6 older **SRTM 90m** zipped tiles (srtm_54–56 × 07–08). These appear to be legacy — Copernicus COG tiles supersede them for the NER pipeline.

---

## 4. OSM Road Data

### Road Points (sampling grid)
**Files:** 16 CSVs — `roadpts_r{0..3}c{0..3}.csv`  
**Schema:** `lon, lat, cls` (road class: `tertiary`, etc.)  
**Rows per file:** ~100K–376K | **Total:** ~1.5M road vertices  
**Purpose:** Generating non-event background samples at road locations.

> [!CAUTION]
> Coverage is **incomplete** — README explicitly states: *"covers roughly lat 21.3–25.7, not the full 21.5–29.6."* The `dist_road_m` feature column is omitted in the current pipeline as a result. Completing the Overpass tile download is needed for v1.1.

### Road Geometries
- `ner_roads_band2.json` (54 MB), `ner_roads_band3.json` (70 MB), `ner_roads_band5.json` (16 MB) — contain actual GeoJSON geometry
- `ner_roads_band0,1,4.json` — 695 bytes each (metadata/index only, essentially empty)
- `roads_r0c0.json` — 31 MB full road geometry for one tile

### OSM Shapefile
- `ner-osm-free.shp.zip` — 289 MB, full OSM extract as shapefile

---

## 5. Auxiliary / Misc

| File | Notes |
|---|---|
| `ner-landslide-v1-real-data.zip` | Zipped bundle of the real labeled data |
| `roadpts_r0c.tif` | 513 KB raster — likely road proximity distance raster for r0c tile |

---

## Summary: Data Quality Observations

| Issue | Severity | Detail |
|---|---|---|
| OSM road coverage gap (lat 25.7–29.6) | 🟡 Medium | `dist_road_m` feature missing in current dataset |
| Copernicus tile gaps | 🟡 Medium | Some lat/lon combos absent — verify if legitimate |
| Duplicate rainfall file | 🟢 Low | `IMD_rain_0.25_2015.nc` likely duplicates `ind2015_rfp25.nc` |
| SRTM tiles (legacy) | 🟢 Low | Superseded by Copernicus; safe to archive/remove |
| `ner_roads_band0,1,4.json` near-empty | 🟢 Low | 695 bytes — may just be band index metadata |
| `label_conf` varies widely | 🟡 Medium | Some events have `loc_accuracy_km` = 25 km — consider filtering |

---

## Recommended Next Steps

1. **Verify the 2015 rainfall duplicate** — compare `IMD_rain_0.25_2015.nc` vs `ind2015_rfp25.nc`
2. **Complete OSM road tiles** for lat 25.7–29.6 (Arunachal, Nagaland, Mizoram) to get `dist_road_m` in v1.1
3. **Audit Copernicus gaps** — confirm missing tiles are ocean/plains and not actual NER terrain
4. **Archive SRTM zips** — they are 233 MB of legacy data not used by current pipeline
5. **Low-confidence samples** — consider a `label_conf >= 0.5` filter before model training
