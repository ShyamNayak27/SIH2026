"""
extract_cv_features.py
-----------------------
Owner: #4 Remote Sensing / Vision

This is the "real" version of the feasibility check — takes any list of
(name, lat, lon) locations from src/models/vision/locations.csv and
returns NDVI, elevation, slope, and land-cover class for each one, then
writes the result to data/external/satellite_features.csv so #2 can merge
it into the shared feature table for the XGBoost model.

USAGE:
  1. Fill in src/models/vision/locations.csv with real historical
     landslide points (from GSI Bhukosh / NASA COOLR) — needs at minimum
     columns: name, lat, lon. Add a `label` column (1 = landslide site,
     0 = non-landslide control point) if #2 wants it for training —
     the script preserves any extra columns as-is.
  2. Run from the project root:
       python3 src/models/vision/extract_cv_features.py
  3. Output lands in data/external/satellite_features.csv

Requires ee.Authenticate() to already have run once (same as
feasibility_check.py) — this reuses the same cached credentials.
"""

import ee
import pandas as pd
import sys
from pathlib import Path

# ---------------------------------------------------------------------------
# CONFIG
# ---------------------------------------------------------------------------
EE_PROJECT_ID = "landslide-prediction123"

LOCATIONS_CSV = Path("SIH2026/src/models/vision/locations.csv")
OUTPUT_CSV = Path("data/external/satellite_features.csv")
BUFFER_METERS = 1000  # ~1km radius sample region around each point

# ---------------------------------------------------------------------------
# AUTH / INIT
# ---------------------------------------------------------------------------
try:
    ee.Initialize(project=EE_PROJECT_ID)
except Exception:
    ee.Authenticate()
    ee.Initialize(project=EE_PROJECT_ID)


# ---------------------------------------------------------------------------
# PER-LOCATION EXTRACTORS (same logic as feasibility_check.py, proven working)
# ---------------------------------------------------------------------------
def get_ndvi(point_geom, start="2024-01-01", end="2024-12-31"):
    s2 = (
        ee.ImageCollection("COPERNICUS/S2_SR_HARMONIZED")
        .filterBounds(point_geom)
        .filterDate(start, end)
        .filter(ee.Filter.lt("CLOUDY_PIXEL_PERCENTAGE", 20))
    )
    if s2.size().getInfo() == 0:
        return None
    median = s2.median()
    ndvi = median.normalizedDifference(["B8", "B4"]).rename("NDVI")
    return ndvi.reduceRegion(
        reducer=ee.Reducer.mean(), geometry=point_geom, scale=10, maxPixels=1e9
    ).get("NDVI").getInfo()


def get_terrain(point_geom):
    dem = ee.Image("USGS/SRTMGL1_003")
    elevation = dem.reduceRegion(
        reducer=ee.Reducer.mean(), geometry=point_geom, scale=30, maxPixels=1e9
    ).get("elevation").getInfo()
    slope_img = ee.Terrain.slope(dem)
    slope = slope_img.reduceRegion(
        reducer=ee.Reducer.mean(), geometry=point_geom, scale=30, maxPixels=1e9
    ).get("slope").getInfo()
    return elevation, slope


def get_landcover(point_geom):
    lc = ee.ImageCollection("ESA/WorldCover/v200").first()
    return lc.reduceRegion(
        reducer=ee.Reducer.mode(), geometry=point_geom, scale=10, maxPixels=1e9
    ).get("Map").getInfo()


# ---------------------------------------------------------------------------
# MAIN EXTRACTION FUNCTION
# ---------------------------------------------------------------------------
def extract_features(locations: pd.DataFrame) -> pd.DataFrame:
    """
    locations: DataFrame with at least [name, lat, lon] columns (any extra
               columns, e.g. `label`, are preserved and passed through).
    returns: same DataFrame with added columns:
             ndvi, elevation_m, slope_deg, landcover_class
    """
    results = []
    total = len(locations)

    for i, row in locations.iterrows():
        name = row.get("name", f"point_{i}")
        lat, lon = row["lat"], row["lon"]
        print(f"[{i + 1}/{total}] {name} ({lat}, {lon})...")

        point = ee.Geometry.Point([lon, lat])
        region = point.buffer(BUFFER_METERS)

        out = row.to_dict()

        try:
            out["ndvi"] = get_ndvi(region)
        except Exception as e:
            print(f"  NDVI error: {e}")
            out["ndvi"] = None

        try:
            elevation, slope = get_terrain(region)
            out["elevation_m"] = elevation
            out["slope_deg"] = slope
        except Exception as e:
            print(f"  terrain error: {e}")
            out["elevation_m"] = None
            out["slope_deg"] = None

        try:
            out["landcover_class"] = get_landcover(region)
        except Exception as e:
            print(f"  land-cover error: {e}")
            out["landcover_class"] = None

        results.append(out)

    return pd.DataFrame(results)


# ---------------------------------------------------------------------------
# CLI ENTRY POINT
# ---------------------------------------------------------------------------
def main():
    if not LOCATIONS_CSV.exists():
        print(f"ERROR: {LOCATIONS_CSV} not found.")
        print("Create it with columns: name,lat,lon (add more real points before running this).")
        sys.exit(1)

    locations = pd.read_csv(LOCATIONS_CSV)

    if locations.empty or "lat" not in locations.columns or "lon" not in locations.columns:
        print(f"ERROR: {LOCATIONS_CSV} is empty or missing lat/lon columns.")
        print("Fill in real coordinates from GSI Bhukosh / NASA COOLR first.")
        sys.exit(1)

    print(f"Extracting features for {len(locations)} locations...")
    features = extract_features(locations)

    OUTPUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    features.to_csv(OUTPUT_CSV, index=False)

    n_missing_ndvi = features["ndvi"].isna().sum()
    n_missing_terrain = features["elevation_m"].isna().sum()
    n_missing_lc = features["landcover_class"].isna().sum()

    print(f"\nDone. Wrote {len(features)} rows to {OUTPUT_CSV}")
    print(f"Missing: ndvi={n_missing_ndvi}, terrain={n_missing_terrain}, landcover={n_missing_lc}")
    if n_missing_ndvi + n_missing_terrain + n_missing_lc > 0:
        print("Some rows had gaps — check those coordinates individually before handing off to #2.")
    print("\nHand data/external/satellite_features.csv to #2 to merge into the shared feature table.")


if __name__ == "__main__":
    main()