"""
Step 2 — terrain features from Copernicus GLO-90 (1 degree COG tiles).

Why blocks and not tiles: a 3x3 Horn gradient, a 500 m relief window and a
smoothed convergence term all read neighbouring pixels. Computing them tile by
tile puts a discontinuity along all 99 tile borders, and those seams show up in
the risk map as straight lines at whole degrees. So we mosaic on the fly into
overlapping 3-degree blocks with a 0.05 degree buffer, compute inside the
buffered block, and keep only the points in the interior.

Metric correctness without reprojecting: the grid is geographic, so east-west
cell size shrinks with latitude. We use the true metre spacing per row --
dx = cell * 111320 * cos(lat), dy = cell * 110540 -- inside the Horn gradient.

Output: interim/query_terrain.csv
"""
import glob
import os
import sys
import zipfile

import numpy as np
import pandas as pd
import rasterio
from rasterio.merge import merge
from scipy import ndimage

DEG2M_LAT = 110540.0
DEG2M_LON = 111320.0
BLOCK = 3.0          # degrees
BUF = 0.05           # degrees of overlap, ~60 pixels at 3 arcsec

COLS = ["elevation", "slope_deg", "aspect_sin", "aspect_cos", "plan_curv",
        "prof_curv", "tri", "twi", "relief_500m"]


def collect_tifs(dem_dir="data/raw/dem"):
    for z in sorted(glob.glob("data/raw/dem/*.zip")):
        try:
            with zipfile.ZipFile(z) as f:
                for n in f.namelist():
                    if n.endswith(".tif"):
                        f.extract(n, dem_dir)
        except zipfile.BadZipFile:
            print(f"  skipping unreadable {z}")
    tifs = sorted(glob.glob(os.path.join(dem_dir, "*.tif")))
    print(f"{len(tifs)} DEM tiles")
    return tifs


def features(z, transform, lat0):
    """z: float32 2-D. transform: affine. Returns dict of float32 arrays."""
    h, w = z.shape
    cell = abs(transform.a)
    lats = transform.f + (np.arange(h) + 0.5) * transform.e
    dy = np.float32(cell * DEG2M_LAT)
    dx = (cell * DEG2M_LON * np.cos(np.radians(lats))).astype("float32")[:, None]

    kx = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], "float32") / 8.0
    ky = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], "float32") / 8.0
    gx = ndimage.convolve(z, kx) / dx
    gy = ndimage.convolve(z, ky) / dy

    slope = np.degrees(np.arctan(np.hypot(gx, gy))).astype("float32")
    aspect = (np.degrees(np.arctan2(-gy, gx)) % 360).astype("float32")

    gxx = ndimage.convolve(gx, kx) / dx
    gyy = ndimage.convolve(gy, ky) / dy
    gxy = ndimage.convolve(gx, ky) / dy
    p = gx * gx + gy * gy
    q = p + 1.0
    with np.errstate(divide="ignore", invalid="ignore"):
        prof = (gxx * gx * gx + 2 * gxy * gx * gy + gyy * gy * gy) / (p * q ** 1.5)
        plan = (gxx * gy * gy - 2 * gxy * gx * gy + gyy * gx * gx) / p ** 1.5
    prof = np.nan_to_num(prof, nan=0, posinf=0, neginf=0).astype("float32")
    plan = np.nan_to_num(plan, nan=0, posinf=0, neginf=0).astype("float32")
    del gx, gy, gxx, gyy, gxy, p, q

    tri = np.zeros_like(z)
    for sx, sy in [(-1, -1), (-1, 0), (-1, 1), (0, -1),
                   (0, 1), (1, -1), (1, 0), (1, 1)]:
        tri += np.abs(z - np.roll(np.roll(z, sx, 0), sy, 1))
    tri /= 8.0

    conv = np.maximum(0.0, -plan).astype("float32")
    cell_m2 = float(dy) * float(dx.mean())
    acc = ndimage.uniform_filter(conv, 17) * 1e4 * cell_m2 + cell_m2
    twi = np.log(acc / np.maximum(np.tan(np.radians(np.maximum(slope, 0.1))),
                                  1e-3)).astype("float32")
    del conv, acc

    r = max(1, int(round(500.0 / float(dy))))
    relief = (ndimage.maximum_filter(z, 2 * r + 1) -
              ndimage.minimum_filter(z, 2 * r + 1)).astype("float32")

    return dict(elevation=z, slope_deg=slope,
                aspect_sin=np.sin(np.radians(aspect)).astype("float32"),
                aspect_cos=np.cos(np.radians(aspect)).astype("float32"),
                plan_curv=plan, prof_curv=prof, tri=tri, twi=twi,
                relief_500m=relief)


def run(points, tifs, bbox):
    out = pd.DataFrame(np.nan, index=np.arange(len(points)),
                       columns=COLS, dtype="float32")
    srcs = [rasterio.open(t) for t in tifs]
    lon0, lat0, lon1, lat1 = bbox
    nb = 0
    for by in np.arange(lat0, lat1, BLOCK):
        for bx in np.arange(lon0, lon1, BLOCK):
            m = (points.lon.values >= bx) & (points.lon.values < bx + BLOCK) & \
                (points.lat.values >= by) & (points.lat.values < by + BLOCK)
            if not m.any():
                continue
            win = (bx - BUF, by - BUF, bx + BLOCK + BUF, by + BLOCK + BUF)
            try:
                arr, tr = merge(srcs, bounds=win, nodata=-32767.0)
            except Exception as e:
                print(f"  block {bx},{by}: merge failed ({e})")
                continue
            z = arr[0].astype("float32")
            bad = (z <= -32000) | ~np.isfinite(z)
            if bad.all():
                continue
            if bad.any():
                z[bad] = 0.0                      # sea / missing -> 0 m
            f = features(z, tr, by)
            inv = ~tr
            cols, rows = [], []
            for x, y in zip(points.lon.values[m], points.lat.values[m]):
                c, r = inv * (x, y)
                cols.append(int(c)); rows.append(int(r))
            rows = np.clip(np.array(rows), 0, z.shape[0] - 1)
            cols = np.clip(np.array(cols), 0, z.shape[1] - 1)
            idx = np.where(m)[0]
            for k in COLS:
                out.loc[idx, k] = f[k][rows, cols]
            # points that landed on sea/missing DEM stay NaN
            out.loc[idx[bad[rows, cols]], COLS] = np.nan
            nb += 1
            print(f"  block lon{bx:.0f} lat{by:.0f}: {z.shape} "
                  f"-> {int(m.sum())} pts", flush=True)
            del arr, z, f
    for s in srcs:
        s.close()
    print(f"{nb} blocks processed")
    return out


if __name__ == "__main__":
    tifs = collect_tifs()
    pts = pd.read_csv(sys.argv[1] if len(sys.argv) > 1 else "data/interim/query_points.csv")
    feats = run(pts, tifs, (87.0, 21.0, 98.0, 30.0))
    res = pd.concat([pts.reset_index(drop=True), feats], axis=1)
    dst = sys.argv[2] if len(sys.argv) > 2 else "data/interim/query_terrain.csv"
    res.to_csv(dst, index=False)
    print(f"wrote {dst}: {res.shape}; off-DEM: {res.slope_deg.isna().sum()}")
