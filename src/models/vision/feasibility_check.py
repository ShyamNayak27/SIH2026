"""
feasibility_check.py
---------------------
Day 1 feasibility test for the Remote Sensing / Vision workstream (#4).

Goal: for a handful of landslide-prone test locations, check whether we can
cleanly pull NDVI (vegetation), elevation/slope (terrain), and land-cover
class from Google Earth Engine per coordinate. That single yes/no answer
decides your Level 2/3 scope for the rest of the hackathon:

  - Clean, fast pulls  -> fold NDVI/slope/land-cover in as extra columns
                          for the XGBoost model (Level 2/3 "engineered CV
                          features"). This is the realistic target.
  - Messy / slow / gappy -> drop satellite features entirely, spend your
                          time on #6 (dashboard) instead. Don't let this
                          eat Day 2.

HOW TO RUN (pick one):

  Option A - Google Colab (recommended, easiest auth):
    1. Go to https://colab.research.google.com, new notebook.
    2. Paste this whole file into a cell and run it.
    3. First run will prompt a one-time browser sign-in + Earth Engine
       project selection. Follow the printed link.
    4. If you don't have an Earth Engine project yet, register a free one
       first at https://code.earthengine.google.com/register (instant for
       non-commercial/research use - hackathon qualifies).

  Option B - Locally:
    pip install earthengine-api
    python feasibility_check.py
    (same one-time browser auth prompt happens locally)

WHAT TO DO WITH THE OUTPUT:
  This writes feasibility_results.csv. Open it, check the "status" column
  for each location/layer. Then fill in the verdict block in problem.md
  section 11 with 2-3 sentences: can we get this reliably, yes or no.
"""

import ee
import csv

# ---------------------------------------------------------------------------
# 1. AUTH / INIT
#python3 src/models/vision/feasibility_check.py
# ---------------------------------------------------------------------------
# Replace with your own GCP project ID registered for Earth Engine.
# (https://code.earthengine.google.com/register -- free, takes a minute)
EE_PROJECT_ID = "landslide-prediction123"

try:
    ee.Initialize(project=EE_PROJECT_ID)
except Exception:
    ee.Authenticate()  # opens a browser window, one-time
    ee.Initialize(project=EE_PROJECT_ID)

# ---------------------------------------------------------------------------
# 2. TEST LOCATIONS
# ---------------------------------------------------------------------------

#4/1ATsMZqAdIVCTMqtdeyGSDCz0sFy_zuCY649TMQl_6lBQfQVZtqwNR5MOIbQ

# These are APPROXIMATE town/district-level coordinates for well-known
# landslide-prone regions in India -- placeholders to prove the pipeline
# works. Swap these out for real, precise points from GSI Bhukosh or
# NASA's Global Landslide Catalog (COOLR) once #1/#5 have sourced them.
TEST_LOCATIONS = [
    {"name": "Wayanad, Kerala",       "lat": 11.6854, "lon": 76.1320},
    {"name": "Idukki, Kerala",        "lat": 9.8497,  "lon": 77.0965},
    {"name": "Chamoli, Uttarakhand",  "lat": 30.4000, "lon": 79.3200},
    {"name": "Mandi, Himachal Pradesh","lat": 31.7080, "lon": 76.9318},
    {"name": "Darjeeling, WB",        "lat": 27.0410, "lon": 88.2660},
    {"name": "Nilgiris, Tamil Nadu",  "lat": 11.4064, "lon": 76.6932},
]

BUFFER_METERS = 1000  # ~1km radius sample region around each point

# ---------------------------------------------------------------------------
# 3. PULL LAYERS PER LOCATION
# ---------------------------------------------------------------------------
def get_ndvi(point_geom, start="2024-01-01", end="2024-12-31"):
    """Median NDVI from cloud-filtered Sentinel-2 over the last year."""
    s2 = (
        ee.ImageCollection("COPERNICUS/S2_SR_HARMONIZED")
        .filterBounds(point_geom)
        .filterDate(start, end)
        .filter(ee.Filter.lt("CLOUDY_PIXEL_PERCENTAGE", 20))
    )
    count = s2.size().getInfo()
    if count == 0:
        return None, 0
    median = s2.median()
    ndvi = median.normalizedDifference(["B8", "B4"]).rename("NDVI")
    val = ndvi.reduceRegion(
        reducer=ee.Reducer.mean(), geometry=point_geom, scale=10, maxPixels=1e9
    ).get("NDVI").getInfo()
    return val, count


def get_terrain(point_geom):
    """Elevation + slope from SRTM 30m DEM."""
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
    """Dominant land-cover class from ESA WorldCover (10m, 2021)."""
    lc = ee.ImageCollection("ESA/WorldCover/v200").first()
    mode = lc.reduceRegion(
        reducer=ee.Reducer.mode(), geometry=point_geom, scale=10, maxPixels=1e9
    ).get("Map").getInfo()
    return mode


# ---------------------------------------------------------------------------
# 4. RUN + WRITE RESULTS
# ---------------------------------------------------------------------------
def main():
    rows = []
    for loc in TEST_LOCATIONS:
        print(f"Checking {loc['name']}...")
        point = ee.Geometry.Point([loc["lon"], loc["lat"]])
        region = point.buffer(BUFFER_METERS)

        row = {"name": loc["name"], "lat": loc["lat"], "lon": loc["lon"]}

        try:
            ndvi, img_count = get_ndvi(region)
            row["ndvi"] = ndvi
            row["s2_image_count"] = img_count
            row["ndvi_status"] = "OK" if ndvi is not None else "NO CLOUD-FREE IMAGES"
        except Exception as e:
            row["ndvi_status"] = f"ERROR: {e}"

        try:
            elevation, slope = get_terrain(region)
            row["elevation_m"] = elevation
            row["slope_deg"] = slope
            row["terrain_status"] = "OK"
        except Exception as e:
            row["terrain_status"] = f"ERROR: {e}"

        try:
            lc_class = get_landcover(region)
            row["landcover_class"] = lc_class
            row["landcover_status"] = "OK"
        except Exception as e:
            row["landcover_status"] = f"ERROR: {e}"

        rows.append(row)
        print(f"  -> {row}")

    fieldnames = [
        "name", "lat", "lon",
        "ndvi", "s2_image_count", "ndvi_status",
        "elevation_m", "slope_deg", "terrain_status",
        "landcover_class", "landcover_status",
    ]
    with open("feasibility_results.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)

    print("\nDone. See feasibility_results.csv")
    print("Verdict checklist:")
    print("  - Did every location get an NDVI value (not NO CLOUD-FREE IMAGES)?")
    print("  - Do slope/elevation values look sane for known hilly regions?")
    print("  - Did land-cover classes come back with no errors?")
    print("If yes across the board: proceed with satellite features as extra")
    print("tabular columns. If patchy: drop vision, redirect to #6.")


if __name__ == "__main__":
    main()
