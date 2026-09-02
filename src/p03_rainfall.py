"""
Step 3 — antecedent rainfall from the IMD 0.25 degree daily gridded product.

Loads the yearly NetCDF files, clips to the NER window (small enough that the
whole 12-year daily cube fits in memory), then for every (lon, lat, date)
sample computes the rainfall accumulations that actually trigger slope failure:
same-day, 3, 7, 15 and 30 day antecedent totals, plus an antecedent
precipitation index with a 0.9 daily decay.

Windows are INCLUSIVE of the event day and look strictly backwards. Getting
this wrong -- centring the window on the event, or including the day after --
leaks the future into the features and is the single easiest way to build a
model that cannot work in production.
"""
import glob
import os
import numpy as np
import pandas as pd
import xarray as xr

BBOX = dict(min_lon=87.0, min_lat=21.0, max_lon=98.0, max_lat=30.0)
WINDOWS = [1, 3, 7, 15, 30]
API_DECAY = 0.9
API_LEN = 30


def _pick(ds):
    """IMD files vary in variable/dim naming between years."""
    var = next(v for v in ds.data_vars
               if ds[v].ndim == 3 or "rain" in v.lower() or "rf" in v.lower())
    dims = {d.lower(): d for d in ds[var].dims}
    tim = next(dims[k] for k in dims if k.startswith("t"))
    lat = next(dims[k] for k in dims if k.startswith("lat"))
    lon = next(dims[k] for k in dims if k.startswith("lon"))
    return var, tim, lat, lon


def load_cube(raw_dir="data/raw/rainfall"):
    files = sorted(glob.glob(os.path.join(raw_dir, "ind*_rfp25.nc")))
    if not files:
        raise SystemExit("no IMD NetCDF files in " + raw_dir)
    parts = []
    for f in files:
        ds = xr.open_dataset(f, decode_times=True)
        var, tim, lat, lon = _pick(ds)
        da = ds[var]
        da = da.sortby(lat).sortby(lon)
        da = da.sel({lat: slice(BBOX["min_lat"], BBOX["max_lat"]),
                     lon: slice(BBOX["min_lon"], BBOX["max_lon"])})
        da = da.rename({tim: "time", lat: "lat", lon: "lon"})
        parts.append(da.load())
        ds.close()
        print(f"  {os.path.basename(f)}: {da.sizes}", flush=True)
    cube = xr.concat(parts, dim="time").sortby("time")
    cube = cube.where(cube >= 0)          # IMD uses negatives as fill
    print(f"cube {dict(cube.sizes)}  {str(cube.time.values[0])[:10]}"
          f" .. {str(cube.time.values[-1])[:10]}")
    return cube


def antecedent(cube, points):
    """points: DataFrame with lon, lat, event_date (YYYY-MM-DD)."""
    lats = cube.lat.values
    lons = cube.lon.values
    times = pd.to_datetime(cube.time.values).normalize()
    tindex = pd.Series(np.arange(len(times)), index=times)
    arr = cube.values.astype("float32")          # (t, lat, lon)
    arr = np.nan_to_num(arr, nan=0.0)

    iy = np.abs(lats[None, :] - points.lat.values[:, None]).argmin(1)
    ix = np.abs(lons[None, :] - points.lon.values[:, None]).argmin(1)
    dates = pd.to_datetime(points.event_date).dt.normalize()
    it = dates.map(tindex).values                # NaN where the date is outside

    weights = API_DECAY ** np.arange(API_LEN)    # today = 1.0, then decaying
    out = {f"rain_{w}d": np.full(len(points), np.nan, "float32") for w in WINDOWS}
    out["api"] = np.full(len(points), np.nan, "float32")

    for i in range(len(points)):
        t = it[i]
        if not np.isfinite(t):
            continue
        t = int(t)
        series = arr[max(0, t - API_LEN + 1): t + 1, iy[i], ix[i]]
        if series.size == 0:
            continue
        rev = series[::-1]                        # rev[0] = event day
        for w in WINDOWS:
            out[f"rain_{w}d"][i] = rev[:w].sum()
        out["api"][i] = float((rev * weights[:rev.size]).sum())

    # static climatology at the same cell: mean annual total
    yearly = cube.groupby("time.year").sum("time").mean("year").values
    out["rain_annual_mean"] = yearly[iy, ix].astype("float32")
    return pd.DataFrame(out, index=points.index)


if __name__ == "__main__":
    import sys
    cube = load_cube()
    pts = pd.read_csv(sys.argv[1] if len(sys.argv) > 1 else "data/interim/samples.csv")
    r = antecedent(cube, pts)
    res = pd.concat([pts.reset_index(drop=True), r.reset_index(drop=True)], axis=1)
    dst = sys.argv[2] if len(sys.argv) > 2 else "data/interim/samples_rain.csv"
    res.to_csv(dst, index=False)
    print(f"wrote {dst}: {res.shape}; missing rain_3d: {res.rain_3d.isna().sum()}")
    print(res[[f"rain_{w}d" for w in WINDOWS] + ["api", "rain_annual_mean"]]
          .describe().round(1).to_string())
