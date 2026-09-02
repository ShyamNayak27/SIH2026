"""
Step 4 — proximity features: distance to the nearest road and the nearest
watercourse, from OSM geometry pulled through Overpass.

dist_road_m is the hill-cutting proxy. In the North East it is one of the
strongest single predictors there is: the road cut removes the toe support of
the slope above it, and the spoil is tipped onto the slope below. It is also
the feature that makes the output actionable, because a high-risk cell next to
a highway is a road-closure decision, not just a colour on a map.

Distances are computed on a local equirectangular projection centred on the
region. Over 10 degrees of longitude the worst-case scale error is under 1%,
which is far below the positional accuracy of the inventory itself.
"""
import glob
import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

LAT0 = 25.5          # NER centre latitude
KX = 111320.0 * np.cos(np.radians(LAT0))
KY = 110540.0


def to_m(lon, lat):
    return np.column_stack([np.asarray(lon) * KX, np.asarray(lat) * KY])


def load_points(pattern):
    files = sorted(glob.glob(pattern))
    if not files:
        return None
    d = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)
    d = d.drop_duplicates(["lon", "lat"])
    print(f"  {pattern}: {len(files)} files, {len(d)} unique vertices")
    return d


def nearest_distance(samples, targets):
    tree = cKDTree(to_m(targets.lon.values, targets.lat.values))
    d, _ = tree.query(to_m(samples.lon.values, samples.lat.values), k=1)
    return d.astype("float32")


COVERAGE_TOL = 0.25          # degrees of slack at the edge


def covers(samples, targets):
    """
    A nearest-neighbour distance is only meaningful where the target layer
    actually exists. With a partial OSM extract the query still returns a
    number -- it just returns the distance to the nearest road in the tiles you
    happen to have, which was 133 km at the median on the first run. That is
    not a missing value, it is a confident wrong one, and it would have gone
    into the model looking perfectly healthy.

    So: refuse to emit the column at all unless the target layer spans the
    sample extent.
    """
    s = samples
    t = targets
    ok = (t.lon.min() <= s.lon.min() + COVERAGE_TOL and
          t.lon.max() >= s.lon.max() - COVERAGE_TOL and
          t.lat.min() <= s.lat.min() + COVERAGE_TOL and
          t.lat.max() >= s.lat.max() - COVERAGE_TOL)
    if not ok:
        print(f"  COVERAGE GAP: targets span lon "
              f"{t.lon.min():.2f}..{t.lon.max():.2f}, lat "
              f"{t.lat.min():.2f}..{t.lat.max():.2f}; samples need lon "
              f"{s.lon.min():.2f}..{s.lon.max():.2f}, lat "
              f"{s.lat.min():.2f}..{s.lat.max():.2f}")
        print("  -> column omitted rather than filled with wrong distances")
    return ok


if __name__ == "__main__":
    import sys
    pts = pd.read_csv(sys.argv[1] if len(sys.argv) > 1 else "data/interim/query_terrain.csv")

    roads = load_points("data/raw/osm/roadpts_*.csv")
    if roads is not None and covers(pts, roads):
        pts["dist_road_m"] = nearest_distance(pts, roads)
        # distance to a MAJOR road only (NH/SH equivalents) -- the connectivity
        # layer the dashboard prioritises
        major = roads[roads.cls.isin(["motorway", "trunk", "primary"])]
        if len(major):
            pts["dist_major_road_m"] = nearest_distance(pts, major)

    water = load_points("data/raw/osm/waterpts_*.csv")
    if water is not None and covers(pts, water):
        pts["dist_stream_m"] = nearest_distance(pts, water)

    dst = sys.argv[2] if len(sys.argv) > 2 else "data/interim/query_prox.csv"
    pts.to_csv(dst, index=False)
    cols = [c for c in ["dist_road_m", "dist_major_road_m", "dist_stream_m"]
            if c in pts]
    if cols:
        print(pts[cols].describe().round(0).to_string())
    else:
        print("  no proximity columns emitted (see coverage gap above)")
    print(f"wrote {dst}: {pts.shape}")
