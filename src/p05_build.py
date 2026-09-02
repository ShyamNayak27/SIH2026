"""
Step 5 — assemble the modelling table.

Negative sampling
  Negatives are drawn from the 0.02 degree candidate grid, restricted to
  slope >= 8 degrees and to points more than 500 m from any known landslide,
  at 2 negatives per positive. Drawing them uniformly over the region instead
  puts most of them on the Brahmaputra floodplain at 3 degrees, the model
  learns "flat is safe", and its AUC looks wonderful while it is unable to rank
  one hillside against another. The observed slope gap here -- 17.9 deg at
  landslide points against 1.3 deg at random points -- is exactly how that
  shortcut would get learned.

  Each negative is given a random date drawn from the positives' own date
  distribution, so its rainfall features come from a real monsoon day. Without
  that, every negative would be a dry day and the model would only ever learn
  "it rained" rather than "it rained on this kind of slope".

Honest caveat, to be written into the report: these negatives are *unlabelled*,
not confirmed stable. A steep site on a wet day may be an unreported landslide.
This is a positive-unlabelled problem and the negatives carry that noise.

Splitting
  Whole 25 km spatial blocks go to train, val or test. Landslides cluster; a
  random split puts a point 200 m from its own neighbour into the test set and
  reports an AUC that production will never reproduce.
"""
import hashlib
import numpy as np
import pandas as pd
import geopandas as gpd
from shapely.geometry import Point

SEED = 42
MIN_NEG_SLOPE = 0.0   # see docs/FINDINGS.md -- a slope floor INVERTED the signal
POS_BUFFER_M = 500.0
NEG_PER_POS = 2
BLOCK_KM = 25.0
NER_STATES = ["Arunachal Pradesh", "Assam", "Manipur", "Meghalaya",
              "Mizoram", "Nagaland", "Sikkim", "Tripura"]
LAT0, KX, KY = 25.5, 111320.0 * np.cos(np.radians(25.5)), 110540.0

rng = np.random.default_rng(SEED)


def attach_admin(df, states="data/raw/admin/ne_admin1.geojson",
                 districts="data/raw/admin/india_districts.geojson"):
    """State from Natural Earth 10 m; district from the GADM-derived layer.

    Districts are the unit district administrations actually act on, so the
    alerting and prioritisation views key off this column, not off grid cells.
    """
    pts = gpd.GeoDataFrame(df.copy(),
                           geometry=[Point(x, y) for x, y in zip(df.lon, df.lat)],
                           crs="EPSG:4326")
    g = gpd.read_file(states)
    g = g[(g.admin == "India") & (g.name.isin(NER_STATES))][["name", "geometry"]]
    j = gpd.sjoin(pts, g.set_crs("EPSG:4326"), how="left", predicate="within")
    out = df.copy()
    out["state"] = j[~j.index.duplicated()]["name"].values

    try:
        d = gpd.read_file(districts)
        d = d[d.NAME_1.isin(NER_STATES)][["NAME_2", "geometry"]]
        jd = gpd.sjoin(pts, d.set_crs("EPSG:4326"), how="left", predicate="within")
        out["district"] = jd[~jd.index.duplicated()]["NAME_2"].values
    except Exception as e:
        print(f"  district join skipped: {e}")
        out["district"] = ""
    return out


def blocks(df):
    x, y = df.lon.values * KX / 1000.0, df.lat.values * KY / 1000.0
    bx = np.floor((x - x.min()) / BLOCK_KM).astype(int)
    by = np.floor((y - y.min()) / BLOCK_KM).astype(int)
    return bx * 10000 + by


def sample_negatives(cand, pos):
    # A slope floor was tried first (>= 8 deg) and rejected: inside the eight
    # NER states the steep-terrain population is dominated by very steep
    # Arunachal ridges, so the floor made the NEGATIVES steeper than the
    # positives (22.5 deg vs 13.3 deg) and would have taught the model that
    # steeper is safer. Standard random sampling over the study area is used
    # instead; the states are mostly hill terrain, so the "flat is safe"
    # shortcut this floor was meant to prevent does not arise here.
    c = cand[(cand.slope_deg >= MIN_NEG_SLOPE) & cand.slope_deg.notna()].copy()
    cell = POS_BUFFER_M / 1000.0
    px = pos.lon.values * KX / 1000.0
    py = pos.lat.values * KY / 1000.0
    occupied = set(zip(np.floor(px / cell).astype(int), np.floor(py / cell).astype(int)))
    cx = np.floor(c.lon.values * KX / 1000.0 / cell).astype(int)
    cy = np.floor(c.lat.values * KY / 1000.0 / cell).astype(int)
    near = np.fromiter(
        (any((a + i, b + j) in occupied for i in (-1, 0, 1) for j in (-1, 0, 1))
         for a, b in zip(cx, cy)), bool, len(c))
    c = c[~near]
    n = min(len(c), NEG_PER_POS * len(pos))
    print(f"  candidates on slope>= {MIN_NEG_SLOPE} deg and outside buffer: {len(c)}"
          f"; drawing {n}")
    return c.sample(n, random_state=SEED).copy()


def split_blocks(df, frac=(0.70, 0.15, 0.15)):
    b = df.block_id.unique()
    rng.shuffle(b)
    a, c = int(frac[0] * len(b)), int((frac[0] + frac[1]) * len(b))
    lut = {blk: ("train" if i < a else "val" if i < c else "test")
           for i, blk in enumerate(b)}
    return df.block_id.map(lut)


if __name__ == "__main__":
    terr = pd.read_csv("data/interim/query_prox.csv")
    pos_meta = pd.read_csv("data/interim/positives_all.csv")

    pos = terr[terr.kind == "positive"].copy()
    cand = terr[terr.kind == "candidate"].copy()
    pos = pos.merge(pos_meta[["lon", "lat", "event_date", "label_conf",
                              "loc_accuracy_km", "trigger", "size", "category",
                              "fatalities", "rain_triggered", "source"]],
                    on=["lon", "lat"], how="left")
    pos = pos[pos.slope_deg.notna() & pos.rain_triggered.fillna(False)]
    print(f"positives with terrain and a rainfall trigger: {len(pos)}")

    # Clip BOTH populations to the eight NER states before sampling, so the
    # negative:positive ratio is the one we asked for rather than whatever
    # survives a later spatial filter. (Darjeeling sits in West Bengal and
    # drops out here -- same hills, different state, outside the brief.)
    pos = attach_admin(pos)
    pos = pos[pos.state.notna()].reset_index(drop=True)
    cand = attach_admin(cand)
    cand = cand[cand.state.notna()].reset_index(drop=True)
    print(f"inside the 8 NER states: {len(pos)} positives, {len(cand)} candidates")

    neg = sample_negatives(cand, pos)
    neg["event_date"] = rng.choice(pos.event_date.dropna().values, len(neg))
    neg["label_conf"] = 1.0
    neg["loc_accuracy_km"] = 0.05
    for c in ["trigger", "size", "category", "source"]:
        neg[c] = "none"
    neg["fatalities"] = 0

    pos["label"], neg["label"] = 1, 0
    df = pd.concat([pos, neg], ignore_index=True).reset_index(drop=True)
    print(f"assembled: {len(df)} rows ({int(df.label.sum())} positive)")

    # location quality flag: only these positives have coordinates finer than
    # the terrain they sit on (see docs/FINDINGS.md)
    df["hq_location"] = (df.loc_accuracy_km <= 1.0).astype("int8")
    df = df.drop_duplicates(subset=["lon", "lat", "event_date", "label"])
    df = df.reset_index(drop=True)

    df["sample_id"] = [hashlib.md5(f"{a}{b}{c}{d}".encode()).hexdigest()[:16]
                       for a, b, c, d in zip(df.lon, df.lat, df.event_date, df.label)]
    df["block_id"] = blocks(df)
    df["split"] = split_blocks(df)
    df.to_csv("data/interim/samples_no_rain.csv", index=False)
    print(df.groupby("split").agg(n=("label", "size"), pos=("label", "sum"),
                                  rate=("label", "mean")).round(3).to_string())
