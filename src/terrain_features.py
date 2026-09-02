"""
DEM -> terrain features, sampled at point locations.

Input : a mosaicked 30 m DEM GeoTIFF covering NER_BBOX (see docs/DATA_SOURCES.md
        for how to fetch Copernicus GLO-30 tiles).
Output: a DataFrame of the TERRAIN columns in schema.py, one row per point.

Run:  python src/terrain_features.py data/raw/ner_dem_30m.tif data/interim/points.csv

Everything metric (slope, curvature, distances) is computed after reprojecting
the DEM to CRS_METRIC. Computing slope on degrees gives you a ~3x error that
varies with latitude and it will quietly ruin the model.
"""
import sys
import numpy as np
import pandas as pd
import rasterio
from rasterio.warp import calculate_default_transform, reproject, Resampling
from scipy import ndimage

from schema import CRS_METRIC, NODATA


def reproject_to_metric(src_path, dst_path):
    with rasterio.open(src_path) as src:
        transform, w, h = calculate_default_transform(
            src.crs, CRS_METRIC, src.width, src.height, *src.bounds, resolution=30)
        meta = src.meta.copy()
        meta.update(crs=CRS_METRIC, transform=transform, width=w, height=h,
                    dtype="float32", nodata=NODATA)
        with rasterio.open(dst_path, "w", **meta) as dst:
            reproject(rasterio.band(src, 1), rasterio.band(dst, 1),
                      src_transform=src.transform, src_crs=src.crs,
                      dst_transform=transform, dst_crs=CRS_METRIC,
                      resampling=Resampling.bilinear)
    return dst_path


def _grad(z, cell):
    """Horn 3x3 partial derivatives."""
    kx = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], float) / (8 * cell)
    ky = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], float) / (8 * cell)
    return ndimage.convolve(z, kx), ndimage.convolve(z, ky)


def terrain_stack(dem_path):
    """Return dict of 2-D float32 arrays + the rasterio profile."""
    with rasterio.open(dem_path) as src:
        z = src.read(1).astype("float64")
        prof = src.profile
        cell = abs(src.transform.a)
        z[z == src.nodata] = np.nan

    z = np.where(np.isnan(z), ndimage.median_filter(np.nan_to_num(z), 3), z)
    dx, dy = _grad(z, cell)
    slope = np.degrees(np.arctan(np.hypot(dx, dy)))
    aspect = np.degrees(np.arctan2(-dy, dx)) % 360

    dxx, _ = _grad(dx, cell)
    _, dyy = _grad(dy, cell)
    _, dxy = _grad(dx, cell)
    p = dx ** 2 + dy ** 2
    q = p + 1
    with np.errstate(divide="ignore", invalid="ignore"):
        prof_c = (dxx * dx ** 2 + 2 * dxy * dx * dy + dyy * dy ** 2) / (p * q ** 1.5)
        plan_c = (dxx * dy ** 2 - 2 * dxy * dx * dy + dyy * dx ** 2) / (p ** 1.5)

    # TRI: mean absolute difference to the 8 neighbours
    tri = np.zeros_like(z)
    for sx in (-1, 0, 1):
        for sy in (-1, 0, 1):
            if sx or sy:
                tri += np.abs(z - np.roll(np.roll(z, sx, 0), sy, 1))
    tri /= 8.0

    # TWI with a cheap flow-accumulation proxy (upslope area ~ D-inf is overkill
    # for a 4-day build; swap in richdem/pysheds if you have the time)
    acc = ndimage.uniform_filter(np.maximum(0, -prof_c), 15) * cell ** 2 + cell ** 2
    twi = np.log(acc / np.maximum(np.tan(np.radians(np.maximum(slope, 0.1))), 1e-3))

    r = 500 // int(cell)
    relief = (ndimage.maximum_filter(z, 2 * r + 1) -
              ndimage.minimum_filter(z, 2 * r + 1))

    return dict(elevation=z, slope_deg=slope,
                aspect_sin=np.sin(np.radians(aspect)),
                aspect_cos=np.cos(np.radians(aspect)),
                plan_curv=np.nan_to_num(plan_c), prof_curv=np.nan_to_num(prof_c),
                tri=tri, twi=twi, relief_500m=relief), prof


def sample_points(stack, prof, lons, lats):
    """Bilinear-free nearest sample of every layer at the given lon/lat."""
    import rasterio.warp
    xs, ys = rasterio.warp.transform("EPSG:4326", prof["crs"], list(lons), list(lats))
    inv = ~prof["transform"]
    cols, rows = zip(*[inv * (x, y) for x, y in zip(xs, ys)])
    rows = np.clip(np.array(rows, int), 0, prof["height"] - 1)
    cols = np.clip(np.array(cols, int), 0, prof["width"] - 1)
    return pd.DataFrame({k: v[rows, cols].astype("float32") for k, v in stack.items()})


if __name__ == "__main__":
    dem, pts = sys.argv[1], sys.argv[2]
    metric = dem.replace(".tif", "_utm46n.tif")
    reproject_to_metric(dem, metric)
    stack, prof = terrain_stack(metric)
    p = pd.read_csv(pts)
    out = pd.concat([p.reset_index(drop=True),
                     sample_points(stack, prof, p.lon, p.lat)], axis=1)
    out.to_csv("data/interim/points_terrain.csv", index=False)
    print(f"wrote {len(out)} rows x {out.shape[1]} cols")
