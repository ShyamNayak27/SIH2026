# data/raw — expected layout

Nothing in here is committed (see `.gitignore`). The files are large, and IMD
asks that its gridded rainfall not be redistributed. Keep them in shared Drive
and rebuild locally; `docs/PROVENANCE.md` carries the SHA-256 of every file so
you can prove you have the same inputs.

```
data/raw/
├── inventory/
│   └── Global_Landslide_Catalog_Export_rows.csv     8.1 MB   NASA GLC
├── rainfall/
│   └── ind{2006..2017}_rfp25.nc                     292 MB   IMD 0.25° daily, 12 files
├── dem/
│   └── Copernicus_DSM_COG_30_N{lat}_00_E{lon}_00_DEM.tif
│                                                    194 MB   GLO-90, 37 tiles
├── osm/
│   └── roadpts_r{row}c{col}.csv                      30 MB   Overpass vertices, lon,lat,cls
└── admin/
    ├── ne_admin1.geojson                             39 MB   Natural Earth 10 m states
    └── india_districts.geojson                       33 MB   GADM-derived districts
```

## Filling it

On the machine that did the downloading, from the repo root:

```powershell
powershell -ExecutionPolicy Bypass -File scripts\organise_raw.ps1
```

That moves the loose downloads out of your Downloads folder into the tree above
and fetches the two admin layers. It moves rather than copies, and it will not
overwrite a file that is already in place.

Anywhere else, fetch them from the URLs in `docs/PROVENANCE.md`.

## Re-fetching individual layers

| layer | how |
|---|---|
| inventory | direct download, no login |
| rainfall | `https://www.imdpune.gov.in/cmpg/Griddata/RF25/ind<YEAR>_rfp25.nc` |
| dem | `https://copernicus-dem-90m.s3.amazonaws.com/<tile>/<tile>.tif`, tiles N21–N29 × E087–E097 |
| osm | Overpass, `way["highway"~"^(motorway\|trunk\|primary\|secondary\|tertiary)$"](bbox); out geom;` reduced to unique vertices |
| admin | raw.githubusercontent.com, URLs in PROVENANCE.md |

## Coverage note

The OSM extract is **incomplete** — it covers roughly lat 21.3–25.7, not the
full 21.5–29.6. `p04_proximity.py` detects this and omits the distance columns
rather than emitting confidently wrong values, so the pipeline runs either way.
Finish the remaining Overpass tiles to get `dist_road_m` in v1.1.
