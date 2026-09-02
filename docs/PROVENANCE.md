# Provenance

Retrieved 2026-09-02. Every input was downloaded from the
URL shown, with no manual editing. Hashes let you prove you have the same inputs.

## Inputs

| files | layer | sha256 (first 16) | bytes | source |
|---|---|---|---|---|
| `Global_Landslide_Catalog_Export_rows.csv` | NASA Global Landslide Catalog | `2c4898899dd4f373` | 8,479,717 | https://data.nasa.gov/docs/legacy/Global_Landslide_Catalog_Export/Global_Landslide_Catalog_Export_rows.csv |
| `*.nc` (12) | IMD 0.25 deg daily gridded rainfall, 2006-2017 | `ec13ce26fe914f79` (set) | 305,390,988 | https://www.imdpune.gov.in/cmpg/Griddata/RF25/ind<YEAR>_rfp25.nc |
| `*.tif` (37) | Copernicus GLO-90 DEM | `0429d6cdf1dda898` (set) | 203,088,695 | https://copernicus-dem-90m.s3.amazonaws.com/<tile>/<tile>.tif |
| `*.csv` (5) | OpenStreetMap road vertices via Overpass | `e997eb1a33f09f45` (set) | 31,343,612 | https://overpass-api.de/api/interpreter |
| `ne_admin1.geojson` | Natural Earth 10m states | `22d0e3ad85eb3e27` | 40,726,851 | https://raw.githubusercontent.com/nvkelso/natural-earth-vector/master/geojson/ne_10m_admin_1_states_provinces.geojson |
| `india_districts.geojson` | GADM-derived India districts | `89ae710eb9dd947a` | 34,510,725 | https://raw.githubusercontent.com/geohacker/india/master/district/india_district.geojson |

## Outputs

| file | sha256 (first 16) | bytes |
|---|---|---|
| `data/processed/ner_landslide_v1.csv` | `b0a1199dd4199789` | 248,497 |
| `gis/ner_landslide_events_v1.geojson` | `351507984b58ef34` | 149,953 |
| `gis/ner_risk_grid_v1.geojson` | `fc1e5eb3e2064fdd` | 381,532 |

## Licences

- NASA Global Landslide Catalog — cite Kirschbaum et al. 2010 (Nat. Hazards
  52:561) and Kirschbaum, Stanley & Zhou 2015 (Geomorphology).
- IMD gridded rainfall — cite Pai et al. 2014, MAUSAM 65(1):1-18. **Do not
  redistribute the NetCDF**; it stays out of git and out of any public repo.
- Copernicus GLO-90 DEM — free and open. Credit "Copernicus DEM, (c) DLR/ESA".
- OpenStreetMap — ODbL. Credit "(c) OpenStreetMap contributors".
- Natural Earth — public domain. GADM-derived districts — non-commercial use.

## Not obtained, and the exact blocker

| layer | blocker |
|---|---|
| Copernicus GLO-30 / SRTM 30 m | OpenTopography API returns 401 without a key; the key requires an account |
| GSI Bhukosh landslide polygons, lithology, faults | JavaScript portal, toposheet-by-toposheet export, not scriptable |
| ESA WorldCover land cover | 3-degree tiles exceed the transfer ceiling on the link used |
| Sentinel-2 NDVI | needs a Copernicus Data Space login |
| SMAP / ERA5-Land soil moisture | needs an Earthdata or CDS login |
| OSM roads north of ~25.7 N | Overpass returned 504 under load; 5 of 16 tiles retrieved |

