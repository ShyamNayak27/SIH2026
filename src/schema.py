"""
Canonical dataset contract for the NER landslide early-warning system.

THIS FILE IS THE INTERFACE between Milestone-1 owners.
Anyone who changes a column name changes it here first, in a PR, or the
downstream pipeline breaks silently.

Owner: Shyam (data foundation)
Consumers: Charu + Shourya (preprocessing / baseline), Rhea (temporal),
           Neil (CV + dashboard), Stuti (explanations)
"""

# ---------------------------------------------------------------- geography
# North Eastern Region: AR, AS, MN, ML, MZ, NL, TR, SK
NER_BBOX = dict(min_lon=87.5, min_lat=21.5, max_lon=97.5, max_lat=29.6)

CRS_GEOGRAPHIC = "EPSG:4326"   # canonical storage CRS for every vector/table
CRS_METRIC = "EPSG:32646"      # UTM 46N — use for ANY distance/area/slope math
GRID_CELL_M = 1000             # dashboard heatmap cell size

NER_STATES = ["Arunachal Pradesh", "Assam", "Manipur", "Meghalaya",
              "Mizoram", "Nagaland", "Sikkim", "Tripura"]

NODATA = -9999.0               # never use NaN in the shipped CSV

# ------------------------------------------------------------------ columns
# (name, dtype, unit, source, notes)
IDENTITY = [
    ("sample_id",   "str",   "-",     "generated", "stable uuid5 of (lon,lat,date)"),
    ("lon",         "f8",    "deg",   "-",         "EPSG:4326"),
    ("lat",         "f8",    "deg",   "-",         "EPSG:4326"),
    ("state",       "str",   "-",     "admin",     "one of NER_STATES"),
    ("district",    "str",   "-",     "admin",     "LGD district name"),
    ("event_date",  "date",  "-",     "inventory", "NODATA-null for negatives"),
]

# Static terrain — derived from a 30 m DEM (Copernicus GLO-30 preferred)
TERRAIN = [
    ("elevation",     "f4", "m",     "COP30",  ""),
    ("slope_deg",     "f4", "deg",   "COP30",  "Horn 3x3, computed in metric CRS"),
    ("aspect_sin",    "f4", "-",     "COP30",  "sin(aspect) — never feed raw degrees"),
    ("aspect_cos",    "f4", "-",     "COP30",  "cos(aspect)"),
    ("plan_curv",     "f4", "1/m",   "COP30",  "negative = convergent"),
    ("prof_curv",     "f4", "1/m",   "COP30",  ""),
    ("tri",           "f4", "m",     "COP30",  "terrain ruggedness index"),
    ("twi",           "f4", "-",     "COP30",  "topographic wetness index"),
    ("relief_500m",   "f4", "m",     "COP30",  "max-min elevation in 500 m radius"),
    ("dist_stream_m", "f4", "m",     "COP30",  "from D8 stream network"),
    ("dist_road_m",   "f4", "m",     "OSM",    "hill-cutting proxy — key NER feature"),
    ("dist_fault_m",  "f4", "m",     "GSI",    "seamless geoscientific map"),
]

# Categorical context — one-hot / target-encode downstream, not here
CONTEXT = [
    ("lithology",   "str", "-", "GSI Bhukosh",   "GSI lithology class"),
    ("geomorph",    "str", "-", "GSI/Bhuvan",    "geomorphological unit"),
    ("lulc",        "str", "-", "ESA WorldCover","10 m, resampled to majority @30 m"),
    ("soil_texture","str", "-", "NBSS-LUP/SoilGrids", ""),
    ("ndvi",        "f4",  "-", "Sentinel-2",    "pre-event median, -1..1"),
]

# Dynamic / weather-linked — the columns that make this a *forecast* system
DYNAMIC = [
    ("rain_1d",   "f4", "mm", "IMD 0.25deg / IMERG", "rainfall on event_date"),
    ("rain_3d",   "f4", "mm", "IMD/IMERG", "antecedent 3-day cumulative"),
    ("rain_7d",   "f4", "mm", "IMD/IMERG", ""),
    ("rain_15d",  "f4", "mm", "IMD/IMERG", ""),
    ("rain_30d",  "f4", "mm", "IMD/IMERG", "antecedent wetness proxy"),
    ("rain_annual_mean", "f4", "mm", "IMD climatology", "static climate normal"),
    ("soil_moist", "f4", "m3/m3", "SMAP L4 / ERA5-Land", "0-10 cm, event date"),
    ("api",       "f4", "mm", "derived", "antecedent precipitation index, k=0.9"),
]

TARGETS = [
    ("label",      "i1", "-", "inventory", "1 = landslide, 0 = non-landslide"),
    ("label_conf", "f4", "-", "inventory", "0.3 news-report .. 1.0 GSI mapped"),
]

SPLIT = [
    ("block_id",   "i4", "-", "generated", "spatial block for CV — DO NOT random-split"),
    ("split",      "str","-", "generated", "train | val | test"),
]

ALL_COLUMNS = IDENTITY + TERRAIN + CONTEXT + DYNAMIC + TARGETS + SPLIT
COLUMN_NAMES = [c[0] for c in ALL_COLUMNS]
FEATURE_NAMES = [c[0] for c in TERRAIN + CONTEXT + DYNAMIC]
NUMERIC_FEATURES = [c[0] for c in TERRAIN + CONTEXT + DYNAMIC if c[1].startswith("f")]
CATEGORICAL_FEATURES = [c[0] for c in CONTEXT if c[1] == "str"]

if __name__ == "__main__":
    print(f"{len(COLUMN_NAMES)} columns, {len(FEATURE_NAMES)} features")
    for n, d, u, s, note in ALL_COLUMNS:
        print(f"  {n:<18} {d:<5} {u:<8} {s:<22} {note}")
