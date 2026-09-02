"""
Generates a SCHEMA-CORRECT SYNTHETIC dataset so that the modelling, dashboard
and explainability tracks can start on day 1 instead of waiting for downloads.

!!! THE NUMBERS ARE SIMULATED. NOTHING HERE IS A REAL LANDSLIDE. !!!
Its only contract is the column names, dtypes, ranges and split structure --
which are identical to the real build. When data/processed/ner_landslide_v1.csv
lands, every downstream script should run against it unchanged.

Run: python src/make_mock.py
"""
import hashlib
import numpy as np
import pandas as pd

from schema import (NER_BBOX, COLUMN_NAMES, NODATA)
from sampling import assign_blocks, split_by_block, SEED

rng = np.random.default_rng(SEED)
N = 12000

STATES = {  # rough centroid + spread, hill states weighted higher
    "Sikkim":            (88.5, 27.5, 0.35, 0.10),
    "Arunachal Pradesh": (94.0, 28.0, 2.20, 0.55, ),
    "Assam":             (92.8, 26.3, 1.80, 0.10),
    "Meghalaya":         (91.3, 25.5, 0.90, 0.30),
    "Nagaland":          (94.4, 26.0, 0.55, 0.30),
    "Manipur":           (93.9, 24.7, 0.55, 0.25),
    "Mizoram":           (92.8, 23.4, 0.55, 0.30),
    "Tripura":           (91.7, 23.7, 0.45, 0.08),
}
LITHO = ["Sandstone-shale", "Phyllite-schist", "Gneiss", "Granite",
         "Alluvium", "Limestone", "Quartzite"]
GEOMORPH = ["Structural hill", "Denudational hill", "Valley fill",
            "Piedmont slope", "Flood plain"]
LULC = ["Forest", "Shrubland", "Cropland", "Built-up", "Bare", "Grassland"]
SOIL = ["Sandy loam", "Clay loam", "Silty clay", "Loam", "Sandy clay loam"]


def build():
    w = np.array([s[3] for s in STATES.values()], float)
    w /= w.sum()
    picks = rng.choice(list(STATES), N, p=w)
    lon = np.array([STATES[s][0] for s in picks]) + rng.normal(0, 0.35, N) * \
          np.array([STATES[s][2] for s in picks])
    lat = np.array([STATES[s][1] for s in picks]) + rng.normal(0, 0.30, N)
    lon = np.clip(lon, NER_BBOX["min_lon"], NER_BBOX["max_lon"])
    lat = np.clip(lat, NER_BBOX["min_lat"], NER_BBOX["max_lat"])

    hilly = np.array([STATES[s][3] > 0.15 for s in picks])
    elevation = np.where(hilly, rng.gamma(4, 260, N), rng.gamma(2, 60, N))
    slope = np.clip(np.where(hilly, rng.gamma(4.5, 5.5, N), rng.gamma(2, 2.2, N)), 0, 78)
    aspect = rng.uniform(0, 360, N)
    tri = slope * rng.uniform(0.25, 0.6, N)
    twi = np.clip(12 - 0.12 * slope + rng.normal(0, 1.1, N), 1, 20)
    relief = np.clip(slope * rng.uniform(6, 12, N), 5, 1400)
    dist_road = np.clip(rng.gamma(1.6, 800, N), 5, 25000)
    dist_stream = np.clip(rng.gamma(2.0, 260, N), 2, 6000)
    dist_fault = np.clip(rng.gamma(2.2, 4200, N), 30, 60000)
    ndvi = np.clip(0.72 - 0.004 * slope + rng.normal(0, 0.13, N), -0.1, 0.95)

    rain_annual = np.clip(rng.normal(2600, 900, N) +
                          400 * (picks == "Meghalaya"), 900, 11000)
    rain_1d = np.clip(rng.gamma(1.1, 22, N) * (rain_annual / 2600), 0, 520)
    rain_3d = rain_1d + np.clip(rng.gamma(1.5, 26, N), 0, 400)
    rain_7d = rain_3d + np.clip(rng.gamma(2.0, 30, N), 0, 600)
    rain_15d = rain_7d + np.clip(rng.gamma(2.5, 34, N), 0, 800)
    rain_30d = rain_15d + np.clip(rng.gamma(3.0, 38, N), 0, 1100)
    api = 0.55 * rain_3d + 0.3 * rain_7d + 0.15 * rain_15d
    soil_moist = np.clip(0.16 + 0.00035 * rain_15d + rng.normal(0, 0.03, N), 0.03, 0.55)

    # latent hazard -> label. The coefficients encode the physical story we
    # expect the model to rediscover; do NOT tune a model to beat this number.
    # intercept is tuned so the mock lands near a 1:2 positive:negative ratio,
    # which is the ratio the real build targets (NEG_PER_POS in sampling.py)
    z = (-4.15
         + 0.085 * slope
         + 0.0011 * relief
         + 0.0032 * rain_3d
         + 2.9 * (soil_moist - 0.2)
         - 0.00016 * dist_road
         - 0.00004 * dist_fault
         - 1.4 * ndvi
         + 0.10 * (twi - 10)
         + rng.normal(0, 0.85, N))
    p = 1 / (1 + np.exp(-z))
    label = (rng.uniform(size=N) < p).astype("int8")

    df = pd.DataFrame(dict(
        lon=lon.round(5), lat=lat.round(5), state=picks,
        district=[f"{s.split()[0][:6]}-D{rng.integers(1, 8)}" for s in picks],
        elevation=elevation, slope_deg=slope,
        aspect_sin=np.sin(np.radians(aspect)), aspect_cos=np.cos(np.radians(aspect)),
        plan_curv=rng.normal(0, 0.02, N), prof_curv=rng.normal(0, 0.02, N),
        tri=tri, twi=twi, relief_500m=relief,
        dist_stream_m=dist_stream, dist_road_m=dist_road, dist_fault_m=dist_fault,
        lithology=rng.choice(LITHO, N), geomorph=rng.choice(GEOMORPH, N),
        lulc=rng.choice(LULC, N, p=[.45, .12, .2, .07, .06, .10]),
        soil_texture=rng.choice(SOIL, N), ndvi=ndvi,
        rain_1d=rain_1d, rain_3d=rain_3d, rain_7d=rain_7d, rain_15d=rain_15d,
        rain_30d=rain_30d, rain_annual_mean=rain_annual,
        soil_moist=soil_moist, api=api,
        label=label,
        label_conf=np.where(label == 1, rng.choice([0.4, 0.7, 1.0], N), 1.0),
    ))

    # monsoon-weighted dates for positives, null for negatives
    days = rng.choice(np.arange(365), N, p=_monsoon_weights())
    dates = pd.Timestamp("2023-01-01") + pd.to_timedelta(days, "D")
    df["event_date"] = np.where(df.label == 1, dates.strftime("%Y-%m-%d"), "")

    df["sample_id"] = [hashlib.md5(f"{a}{b}{c}".encode()).hexdigest()[:16]
                       for a, b, c in zip(df.lon, df.lat, df.event_date)]
    df["block_id"] = assign_blocks(df)
    df["split"] = split_by_block(df)

    for c in df.columns:
        if df[c].dtype.kind == "f":
            df[c] = df[c].astype("float32").round(4)
    return df[COLUMN_NAMES]


def _monsoon_weights():
    d = np.arange(365)
    w = np.exp(-((d - 190) ** 2) / (2 * 42 ** 2)) + 0.05
    return w / w.sum()


if __name__ == "__main__":
    df = build()
    df.to_csv("data/processed/MOCK_ner_landslide_v0.csv", index=False)
    print(df.groupby("split").agg(n=("label", "size"), pos_rate=("label", "mean")))
    print(f"\nwrote data/processed/MOCK_ner_landslide_v0.csv  "
          f"{df.shape[0]} rows x {df.shape[1]} cols")
