"""
Negative sampling + spatial-block splitting.

Two things here decide whether the model's reported AUC means anything:

1. NEGATIVE SAMPLING. If you draw negatives uniformly over NER, most land on the
   Brahmaputra floodplain at 3 degrees slope and the model learns "flat = safe".
   You get AUC 0.98 and a system that cannot rank one hillside against another.
   We sample negatives only from terrain that is plausibly landslide-capable
   (slope >= MIN_NEG_SLOPE) and outside a buffer around every known landslide.

2. SPATIAL BLOCKS. Landslides cluster. A random train/test split puts a point
   200 m from its own neighbour in the test set -> leakage -> inflated scores.
   We split by ~25 km blocks so test blocks are geographically held out.
"""
import numpy as np
import pandas as pd

MIN_NEG_SLOPE = 8.0        # degrees
POS_BUFFER_M = 500.0       # no negative within this distance of a positive
NEG_PER_POS = 2.0
BLOCK_KM = 25.0
SEED = 42


def _to_km(lon, lat):
    """Cheap equirectangular km, fine for block assignment at NER latitudes."""
    return lon * 111.32 * np.cos(np.radians(lat)), lat * 110.57


def assign_blocks(df, block_km=BLOCK_KM):
    x, y = _to_km(df.lon.values, df.lat.values)
    bx = np.floor((x - x.min()) / block_km).astype(int)
    by = np.floor((y - y.min()) / block_km).astype(int)
    return bx * 10_000 + by


def sample_negatives(candidates, positives, n, rng=None):
    """
    candidates : DataFrame with lon, lat, slope_deg  (a dense grid over NER)
    positives  : DataFrame with lon, lat
    """
    rng = rng or np.random.default_rng(SEED)
    c = candidates[candidates.slope_deg >= MIN_NEG_SLOPE].copy()

    px, py = _to_km(positives.lon.values, positives.lat.values)
    cx, cy = _to_km(c.lon.values, c.lat.values)
    # grid-hash the positives so this stays O(n) instead of O(n*m)
    cell = POS_BUFFER_M / 1000.0
    occupied = set(zip(np.floor(px / cell).astype(int), np.floor(py / cell).astype(int)))
    keys = list(zip(np.floor(cx / cell).astype(int), np.floor(cy / cell).astype(int)))
    near = np.array([any((kx + i, ky + j) in occupied
                         for i in (-1, 0, 1) for j in (-1, 0, 1))
                     for kx, ky in keys])
    c = c[~near]
    take = min(int(n), len(c))
    return c.sample(take, random_state=SEED).assign(label=0)


def split_by_block(df, frac=(0.7, 0.15, 0.15), rng=None):
    """Assign train/val/test whole-block-wise, keeping the label ratio close."""
    rng = rng or np.random.default_rng(SEED)
    blocks = df.block_id.unique()
    rng.shuffle(blocks)
    n = len(blocks)
    a, b = int(frac[0] * n), int((frac[0] + frac[1]) * n)
    lut = {}
    for i, blk in enumerate(blocks):
        lut[blk] = "train" if i < a else ("val" if i < b else "test")
    return df.block_id.map(lut)


def report(df):
    print(f"rows={len(df)}  positives={int(df.label.sum())}  "
          f"blocks={df.block_id.nunique()}")
    print(df.groupby("split").agg(n=("label", "size"), pos_rate=("label", "mean")))
