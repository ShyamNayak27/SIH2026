"""
Data-integrity gate. This is the security/QA check for Milestone 1 -- CI should
run it on every push and fail the build, not print warnings nobody reads.

Run: python src/validate.py data/processed/MOCK_ner_landslide_v0.csv
"""
import sys
import pandas as pd
import numpy as np

from schema import COLUMN_NAMES, NER_BBOX, NER_STATES, FEATURE_NAMES

CHECKS = []


def check(name):
    def deco(fn):
        CHECKS.append((name, fn))
        return fn
    return deco


@check("columns match schema exactly, in order")
def _(df):
    assert list(df.columns) == COLUMN_NAMES, \
        f"missing={set(COLUMN_NAMES) - set(df.columns)} extra={set(df.columns) - set(COLUMN_NAMES)}"


@check("all coordinates inside NER bounding box")
def _(df):
    b = NER_BBOX
    bad = ~(df.lon.between(b["min_lon"], b["max_lon"]) &
            df.lat.between(b["min_lat"], b["max_lat"]))
    assert not bad.any(), f"{bad.sum()} points outside NER"


@check("states are from the canonical list")
def _(df):
    bad = set(df.state.unique()) - set(NER_STATES)
    assert not bad, f"unknown states: {bad}"


@check("no duplicate sample_id")
def _(df):
    d = df.sample_id.duplicated().sum()
    assert d == 0, f"{d} duplicate ids"


@check("no near-duplicate coordinates (<30 m apart, same label)")
def _(df):
    key = df.lon.round(4).astype(str) + "_" + df.lat.round(4).astype(str) + "_" + df.label.astype(str)
    d = key.duplicated().sum()
    assert d < 0.01 * len(df), f"{d} near-duplicate samples -- inventory de-dup failed"


@check("label is strictly binary")
def _(df):
    assert set(df.label.unique()) <= {0, 1}, df.label.unique()


@check("positives carry an event_date, negatives do not")
def _(df):
    pos_missing = ((df.label == 1) & (df.event_date.isna() | (df.event_date == ""))).sum()
    assert pos_missing == 0, f"{pos_missing} positives without a date"


@check("physical ranges hold")
def _(df):
    r = dict(slope_deg=(0, 90), ndvi=(-1, 1), soil_moist=(0, 0.7),
             elevation=(-50, 8000), rain_1d=(0, 1200), twi=(0, 30))
    for c, (lo, hi) in r.items():
        v = df[c]
        assert v.between(lo, hi).all(), f"{c} out of [{lo},{hi}]: {v.min()}..{v.max()}"


@check("rainfall accumulations are monotonic (1d <= 3d <= 7d <= 15d <= 30d)")
def _(df):
    cols = ["rain_1d", "rain_3d", "rain_7d", "rain_15d", "rain_30d"]
    v = df[cols].values
    assert (np.diff(v, axis=1) >= -1e-3).all(), "non-monotonic antecedent rainfall"


@check("no NaN in any feature column")
def _(df):
    n = df[FEATURE_NAMES].isna().sum()
    assert n.sum() == 0, f"NaNs: {n[n > 0].to_dict()}"


@check("splits are disjoint by spatial block (no leakage)")
def _(df):
    overlap = df.groupby("block_id").split.nunique()
    bad = (overlap > 1).sum()
    assert bad == 0, f"{bad} blocks appear in more than one split -- SPATIAL LEAKAGE"


@check("every split has both classes")
def _(df):
    for s, g in df.groupby("split"):
        assert g.label.nunique() == 2, f"split '{s}' has only class {g.label.unique()}"


def main(path):
    df = pd.read_csv(path)
    fails = 0
    for name, fn in CHECKS:
        try:
            fn(df)
            print(f"  PASS  {name}")
        except AssertionError as e:
            fails += 1
            print(f"  FAIL  {name}\n        {e}")
    print(f"\n{len(CHECKS) - fails}/{len(CHECKS)} checks passed on {path} ({len(df)} rows)")
    sys.exit(1 if fails else 0)


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "data/processed/MOCK_ner_landslide_v0.csv")
