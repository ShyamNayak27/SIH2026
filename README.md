# NER Landslide Early-Warning — Data Foundation (Milestone 1, Shyam)

## What is in here

```
src/schema.py            THE CONTRACT. 35 columns. Change it here first or nothing else works.
src/terrain_features.py  DEM -> slope/aspect/curvature/TRI/TWI/relief, sampled at points.
src/sampling.py          Negative sampling + spatial-block train/val/test split.
src/validate.py          12 data-integrity checks. Wire into CI as a blocking gate.
src/make_mock.py         Generates the synthetic stand-in dataset (below).
docs/DATA_SOURCES.md     Where every layer comes from, how to get it, and the traps.
data/processed/MOCK_ner_landslide_v0.csv   12,000 rows x 35 cols  <- SYNTHETIC
gis/MOCK_ner_risk_grid.geojson             1,004 x 0.1deg cells, 4 severity classes
gis/MOCK_ner_landslide_points.geojson      923 event points
```

## Read this before you use the CSV

**`MOCK_ner_landslide_v0.csv` is simulated. Not one row is a real landslide.**

Its only purpose is to unblock parallel work on day 1: it has the exact column
names, dtypes, physical ranges and split structure that the real
`ner_landslide_v1.csv` will have. Every script written against the mock should
run against the real file with no changes. Delete the mock the moment v1 lands.

Sanity check on the mock (test split, spatial blocks held out):
`logreg AUC 0.752 / AP 0.260`, `HistGB AUC 0.716 / AP 0.229`, base rate 8.2%.
It is learnable but not trivial — which is the point. **Do not tune against
these numbers**; they are properties of the simulator, not of NER.

## Quickstart

```bash
pip install -r requirements.txt
python src/make_mock.py                                  # regenerate the mock
python src/validate.py data/processed/MOCK_ner_landslide_v0.csv
```

## The real build, in order

1. `docs/DATA_SOURCES.md` → pull GSI Bhukosh inventory + Copernicus GLO-30 DEM
   + OSM roads for the NER bbox into `data/raw/`.
2. Normalise the inventory to `sample_id, lon, lat, event_date, label_conf`.
   Split it into *dated* and *undated* populations — this is the input Rhea
   needs to decide whether a prediction horizon is honest.
3. `python src/terrain_features.py data/raw/ner_dem_30m.tif data/interim/points.csv`
4. Build the candidate grid, then `sampling.sample_negatives(...)` at 1:2.
5. Join rainfall (IMD 0.25° daily) as `rain_{1,3,7,15,30}d` + `api` for each
   positive's `event_date`; for negatives, draw a random monsoon date in the
   same year so the rainfall distribution isn't a giveaway.
6. `assign_blocks` → `split_by_block` → write `data/processed/ner_landslide_v1.csv`
7. `python src/validate.py data/processed/ner_landslide_v1.csv` — must be 12/12.

## Repo / integrity rules for this milestone

- `data/raw/` and `data/processed/` are **gitignored**. Ship a `manifest.csv`
  of `filename, sha256, rows, source_url, retrieved_at` instead, and put the
  actual files in shared Drive. Nobody commits a 400 MB GeoTIFF.
- Any change to `schema.py` is a PR, not a push to main.
- `validate.py` runs in CI on every PR and blocks the merge on failure.
- Branch per owner: `data/shyam`, `model/charu-shourya`, `temporal/rhea`,
  `cv/neil`, `xai/stuti`. Main stays runnable at all times.
- Record the exact source URL and retrieval date for every layer. For a
  government-facing pitch, unciteable data is worse than no data.
