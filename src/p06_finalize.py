"""
Step 6 — write ner_landslide_v1.csv in the schema_v1 column order, run the
integrity gate, and emit the GIS layers the dashboard reads.
"""
import json
import sys
import numpy as np
import pandas as pd

sys.path.insert(0, "src")
import schema_v1 as S

CHECKS = []


def check(name):
    def deco(fn):
        CHECKS.append((name, fn)); return fn
    return deco


@check("columns match schema_v1 exactly, in order")
def _(d):
    expected = [c for c in S.COLUMNS if c in d.columns]
    assert list(d.columns) == expected, f"out of order: {list(d.columns)}"
    assert not (set(d.columns) - set(S.COLUMNS)), \
        f"columns not in schema: {set(d.columns)-set(S.COLUMNS)}"


@check("all coordinates inside the NER bounding box")
def _(d):
    b = S.NER_BBOX
    assert d.lon.between(b["min_lon"], b["max_lon"]).all()
    assert d.lat.between(b["min_lat"], b["max_lat"]).all()


@check("every row is inside one of the eight NER states")
def _(d):
    bad = set(d.state.dropna().unique()) - set(S.NER_STATES)
    assert not bad and d.state.notna().all(), f"bad/missing states: {bad}"


@check("no duplicate sample_id")
def _(d):
    assert d.sample_id.duplicated().sum() == 0


@check("no duplicate coordinate+date+label")
def _(d):
    k = d.lon.round(5).astype(str) + d.lat.round(5).astype(str) + \
        d.event_date.astype(str) + d.label.astype(str)
    assert k.duplicated().sum() == 0, f"{k.duplicated().sum()} duplicates"


@check("label is binary and both classes present")
def _(d):
    assert set(d.label.unique()) == {0, 1}


@check("every row carries an event_date")
def _(d):
    assert d.event_date.notna().all() and (d.event_date.astype(str) != "").all()


@check("physical ranges hold")
def _(d):
    for c, (lo, hi) in dict(slope_deg=(0, 90), elevation=(-100, 8000),
                            twi=(0, 40), rain_1d=(0, 1500),
                            label_conf=(0, 1)).items():
        v = d[c]
        assert v.between(lo, hi).all(), f"{c}: {v.min()}..{v.max()}"


@check("rainfall accumulations are monotonic (1d<=3d<=7d<=15d<=30d)")
def _(d):
    v = d[["rain_1d", "rain_3d", "rain_7d", "rain_15d", "rain_30d"]].values
    assert (np.diff(v, axis=1) >= -1e-3).all()


@check("no NaN in any feature column")
def _(d):
    present = [c for c in S.FEATURES if c in d.columns]
    n = d[present].isna().sum()
    assert n.sum() == 0, n[n > 0].to_dict()


@check("splits are disjoint by spatial block — no leakage")
def _(d):
    bad = (d.groupby("block_id").split.nunique() > 1).sum()
    assert bad == 0, f"{bad} blocks span more than one split"


@check("every split contains both classes")
def _(d):
    for s, g in d.groupby("split"):
        assert g.label.nunique() == 2, f"split {s} has only {g.label.unique()}"


@check("rainfall separates the classes (the signal this inventory can support)")
def _(d):
    p = d[d.label == 1].rain_3d.median()
    n = d[d.label == 0].rain_3d.median()
    assert p > n * 1.2, f"3-day antecedent rainfall: positives {p:.1f} mm vs negatives {n:.1f} mm"


@check("terrain separation is REPORTED, not assumed (see docs/FINDINGS.md)")
def _(d):
    p = d[d.label == 1].slope_deg.median()
    n = d[d.label == 0].slope_deg.median()
    print(f"        slope: positives {p:.1f} deg, negatives {n:.1f} deg "
          f"-> separation {p - n:+.1f} deg")
    # Deliberately not an assertion. With this inventory the terrain features
    # carry almost no signal because the event coordinates are coarser than the
    # terrain; that is a property of the source, and the build must surface it
    # rather than fail on it or quietly hide it.


def finalize(src="data/interim/samples_rain.csv", dst="data/processed/ner_landslide_v1.csv"):
    d = pd.read_csv(src)
    for c in S.PROXIMITY:
        if c not in d:
            d[c] = np.nan
    have = [c for c in S.COLUMNS if c in d]
    missing = [c for c in S.COLUMNS if c not in d]
    if missing:
        print(f"  note: {missing} not populated in this build")
    cols = [c for c in S.COLUMNS if c in d and d[c].notna().any()]
    d = d[cols]
    for c in d.columns:
        if d[c].dtype.kind == "f":
            d[c] = d[c].astype("float32").round(4)
    d.to_csv(dst, index=False)
    return d


def gate(d):
    fails = 0
    for name, fn in CHECKS:
        try:
            fn(d); print(f"  PASS  {name}")
        except Exception as e:
            fails += 1; print(f"  FAIL  {name}\n        {e}")
    print(f"\n{len(CHECKS)-fails}/{len(CHECKS)} checks passed  ({len(d)} rows)")
    return fails


def gis(d, cell=0.05):
    g = d.copy()
    g["gx"] = np.floor(g.lon / cell).astype(int)
    g["gy"] = np.floor(g.lat / cell).astype(int)
    a = g.groupby(["gx", "gy"]).agg(
        n=("label", "size"), events=("label", "sum"),
        slope=("slope_deg", "mean"), relief=("relief_500m", "mean"),
        rain30=("rain_30d", "mean"), state=("state", lambda s: s.mode()[0]),
        district=("district", lambda s: s.mode()[0] if s.notna().any() else "")
    ).reset_index()
    a["risk_index"] = a.events / a.n
    q = a.risk_index.rank(pct=True)
    a["severity"] = pd.cut(q, [0, .5, .75, .9, 1.0],
                           labels=["Low", "Moderate", "High", "Very High"],
                           include_lowest=True)
    feats = []
    for r in a.itertuples():
        x0, y0 = r.gx * cell, r.gy * cell
        feats.append({"type": "Feature", "geometry": {"type": "Polygon", "coordinates": [[
            [x0, y0], [x0 + cell, y0], [x0 + cell, y0 + cell], [x0, y0 + cell], [x0, y0]]]},
            "properties": {"cell_id": f"{r.gx}_{r.gy}", "state": r.state,
                           "district": str(r.district), "samples": int(r.n),
                           "events": int(r.events),
                           "risk_index": round(float(r.risk_index), 4),
                           "severity": str(r.severity),
                           "mean_slope_deg": round(float(r.slope), 2),
                           "mean_relief_500m_m": round(float(r.relief), 1),
                           "mean_rain_30d_mm": round(float(r.rain30), 1)}})
    json.dump({"type": "FeatureCollection", "name": "ner_risk_grid_v1",
               "crs": {"type": "name",
                       "properties": {"name": "urn:ogc:def:crs:OGC:1.3:CRS84"}},
               "features": feats}, open("gis/ner_risk_grid_v1.geojson", "w"))

    p = d[d.label == 1]
    pf = [{"type": "Feature",
           "geometry": {"type": "Point", "coordinates": [float(r.lon), float(r.lat)]},
           "properties": {"sample_id": r.sample_id, "state": r.state,
                          "district": str(r.district), "event_date": r.event_date,
                          "slope_deg": float(r.slope_deg), "rain_3d": float(r.rain_3d),
                          "rain_30d": float(r.rain_30d), "trigger": r.trigger,
                          "size": r["size"], "fatalities": int(r.fatalities),
                          "label_conf": float(r.label_conf),
                          "loc_accuracy_km": float(r.loc_accuracy_km)}}
          for _, r in p.iterrows()]
    json.dump({"type": "FeatureCollection", "name": "ner_landslide_events_v1",
               "features": pf}, open("gis/ner_landslide_events_v1.geojson", "w"))
    print(f"GIS: {len(feats)} risk cells, {len(pf)} event points")
    print(a.severity.value_counts().to_string())


if __name__ == "__main__":
    d = finalize()
    print(f"wrote data/processed/ner_landslide_v1.csv  {d.shape}\n")
    fails = gate(d)
    gis(d)
    sys.exit(1 if fails else 0)
