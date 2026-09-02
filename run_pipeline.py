"""
Whole build, in order. Run from the repo root:  python run_pipeline.py

Expects data/raw/ populated -- see data/raw/README.md, or run
scripts/organise_raw.ps1 to file the downloads from your Downloads folder.
"""
import subprocess
import sys

import os

import numpy as np
import pandas as pd

sys.path.insert(0, "src")

# git does not track empty directories, so a fresh clone has no data/interim.
for d in ("data/interim", "data/processed", "gis",
          "data/raw/inventory", "data/raw/rainfall", "data/raw/dem",
          "data/raw/osm", "data/raw/admin"):
    os.makedirs(d, exist_ok=True)


def sh(*cmd):
    print(f"\n=== {' '.join(cmd)}")
    if subprocess.call([sys.executable, *cmd]) != 0:
        sys.exit(f"failed: {cmd}")


# 1. inventory -> positives
sh("src/p01_inventory.py")

# 2. query points = positives (India) + a 0.02 deg candidate grid for negatives
p = pd.read_csv("data/interim/positives_all.csv")
p = p[p.country == "India"].copy()
p["kind"] = "positive"
LO, LA = np.meshgrid(np.arange(87.5, 97.5, 0.02), np.arange(21.5, 29.6, 0.02))
g = pd.DataFrame({"lon": LO.ravel().round(5), "lat": LA.ravel().round(5)})
g["kind"] = "candidate"
pd.concat([p[["lon", "lat", "kind"]], g], ignore_index=True) \
  .to_csv("data/interim/query_points.csv", index=False)
print(f"query points: {len(p) + len(g)}")

# 3. terrain, then proximity (proximity self-skips if OSM coverage is partial)
sh("src/p02_dem.py")
sh("src/p04_proximity.py")

# 4. negatives, admin join, spatial-block split
sh("src/p05_build.py")

# 5. antecedent rainfall for every sample, positives and negatives alike
from p03_rainfall import load_cube, antecedent           # noqa: E402
cube = load_cube()
s = pd.read_csv("data/interim/samples_no_rain.csv")
s = s.drop(columns=[c for c in s.columns if c.startswith("rain_") or c == "api"])
r = antecedent(cube, s)
pd.concat([s.reset_index(drop=True), r.reset_index(drop=True)], axis=1) \
  .to_csv("data/interim/samples_rain.csv", index=False)
print("rainfall joined")

# 6. schema order, 14 integrity checks, GIS layers
sh("src/p06_finalize.py")
