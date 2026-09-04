"""
Step 4 — proximity features: distance to the nearest road, major road, and stream.

Supports both:
1. Geofabrik complete shapefile extracts (gis_osm_roads_free_1.shp, gis_osm_waterways_free_1.shp)
2. Overpass CSV point dumps (roadpts_*.csv, waterpts_*.csv)

When Geofabrik shapefiles are present under data/raw/osm/, 100% of NER coordinates
are covered, populating dist_road_m, dist_major_road_m, and dist_stream_m.
"""
import glob
import os
import numpy as np
import pandas as pd
from scipy.spatial import cKDTree
import geopandas as gpd

LAT0 = 25.5          # NER centre latitude
KX = 111320.0 * np.cos(np.radians(LAT0))
KY = 110540.0


def to_m(lon, lat):
    return np.column_stack([np.asarray(lon) * KX, np.asarray(lat) * KY])


def load_shapefile_vertices(shp_path, class_col=None, allowed_classes=None):
    if not os.path.exists(shp_path):
        return None
    print(f"  loading shapefile {os.path.basename(shp_path)} ...")
    gdf = gpd.read_file(shp_path)
    if class_col and allowed_classes and class_col in gdf.columns:
        gdf = gdf[gdf[class_col].isin(allowed_classes)]
    
    # Extract line/multiline coordinate vertices
    coords = []
    classes = []
    for row in gdf.itertuples():
        cls_val = getattr(row, class_col) if class_col and hasattr(row, class_col) else "road"
        geom = row.geometry
        if geom is None:
            continue
        if geom.geom_type == 'LineString':
            pts = list(geom.coords)
            coords.extend(pts)
            classes.extend([cls_val] * len(pts))
        elif geom.geom_type == 'MultiLineString':
            for line in geom.geoms:
                pts = list(line.coords)
                coords.extend(pts)
                classes.extend([cls_val] * len(pts))
        elif geom.geom_type == 'Point':
            coords.append((geom.x, geom.y))
            classes.append(cls_val)

    if not coords:
        return None
    df = pd.DataFrame(coords, columns=["lon", "lat"])
    df["cls"] = classes
    df = df.drop_duplicates(["lon", "lat"]).reset_index(drop=True)
    print(f"  -> extracted {len(df)} unique vertices from shapefile")
    return df


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


def process_proximity(pts):
    # Try Geofabrik shapefiles first
    road_shp = "data/raw/osm/gis_osm_roads_free_1.shp"
    water_shp = "data/raw/osm/gis_osm_waterways_free_1.shp"

    roads = load_shapefile_vertices(road_shp, class_col="fclass")
    if roads is None:
        roads = load_points("data/raw/osm/roadpts_*.csv")

    if roads is not None and covers(pts, roads):
        pts["dist_road_m"] = nearest_distance(pts, roads)
        major = roads[roads.cls.isin(["motorway", "trunk", "primary"])]
        if len(major):
            pts["dist_major_road_m"] = nearest_distance(pts, major)
        print("  -> dist_road_m and dist_major_road_m populated.")

    water = load_shapefile_vertices(water_shp, class_col="fclass")
    if water is None:
        water = load_points("data/raw/osm/waterpts_*.csv")

    if water is not None and covers(pts, water):
        pts["dist_stream_m"] = nearest_distance(pts, water)
        print("  -> dist_stream_m populated.")

    return pts


if __name__ == "__main__":
    import sys
    src_file = sys.argv[1] if len(sys.argv) > 1 else "data/interim/query_terrain.csv"
    dst_file = sys.argv[2] if len(sys.argv) > 2 else "data/interim/query_prox.csv"

    if os.path.exists(src_file):
        pts = pd.read_csv(src_file)
        pts = process_proximity(pts)
        pts.to_csv(dst_file, index=False)
        cols = [c for c in ["dist_road_m", "dist_major_road_m", "dist_stream_m"] if c in pts]
        if cols:
            print(pts[cols].describe().round(0).to_string())
        print(f"wrote {dst_file}: {pts.shape}")
    else:
        print(f"Input file {src_file} not found; skipping standalone execution.")
