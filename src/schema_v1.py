"""
Dataset contract for ner_landslide_v1 — the REAL build.

This supersedes the placeholder schema shipped with the mock. Columns that the
mock guessed at but which no open source could actually supply in the build
window have been REMOVED rather than filled with plausible numbers. They are
listed in PENDING so the modelling code can be written against a stable set and
extended later without a rename.

Everything here is derived from data that was downloaded, and every source is
named in docs/PROVENANCE.md with its retrieval date and file hash.
"""

NER_BBOX = dict(min_lon=87.5, min_lat=21.5, max_lon=97.5, max_lat=29.6)
CRS = "EPSG:4326"
NER_STATES = ["Arunachal Pradesh", "Assam", "Manipur", "Meghalaya",
              "Mizoram", "Nagaland", "Sikkim", "Tripura"]

IDENTITY = ["sample_id", "lon", "lat", "state", "district", "event_date"]

# Copernicus GLO-90, mosaicked in overlapping 3-degree blocks
TERRAIN = ["elevation", "slope_deg", "aspect_sin", "aspect_cos",
           "plan_curv", "prof_curv", "tri", "twi", "relief_500m"]

# OpenStreetMap via Overpass
PROXIMITY = ["dist_road_m", "dist_major_road_m", "dist_stream_m"]

# IMD 0.25 degree daily gridded rainfall, 2006-2017
DYNAMIC = ["rain_1d", "rain_3d", "rain_7d", "rain_15d", "rain_30d",
           "api", "rain_annual_mean"]

TARGETS = ["label", "label_conf"]
META = ["loc_accuracy_km", "trigger", "size", "category", "fatalities", "source",
        "hq_location"]
SPLIT = ["block_id", "split"]

FEATURES = TERRAIN + PROXIMITY + DYNAMIC
COLUMNS = IDENTITY + FEATURES + TARGETS + META + SPLIT

# Not in v1. Each needs either a login (Earthdata, Copernicus, OpenTopography)
# or a portal that cannot be scripted (GSI Bhukosh toposheet export).
PENDING = {
    "lithology":    "GSI Bhukosh seamless geoscientific map — toposheet-wise export",
    "geomorph":     "GSI / Bhuvan",
    "dist_fault_m": "GSI structural layer",
    "lulc":         "ESA WorldCover 10 m — 3-degree tiles too large for this link",
    "soil_texture": "SoilGrids 250 m",
    "ndvi":         "Sentinel-2 L2A — needs a Copernicus Data Space login",
    "soil_moist":   "SMAP L4 or ERA5-Land — needs an Earthdata or CDS login",
}

if __name__ == "__main__":
    print(f"v1: {len(COLUMNS)} columns, {len(FEATURES)} features")
    print("features:", ", ".join(FEATURES))
    print("\npending (need credentials or a manual export):")
    for k, v in PENDING.items():
        print(f"  {k:<14} {v}")
