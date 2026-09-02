"""
Step 1 — normalise the landslide inventory to the schema's positive-sample form.

Input : NASA Global Landslide Catalog export CSV
Output: interim/positives.csv  [sample_id, lon, lat, event_date, label_conf,
                                source, loc_accuracy_km, trigger, size]

label_conf is derived from the catalogue's own location_accuracy field, so the
model and the loss can down-weight a point that is only known to 25 km. That
column is the honest record of how much each positive is worth.
"""
import hashlib
import numpy as np
import pandas as pd

BBOX = dict(min_lon=87.5, min_lat=21.5, max_lon=97.5, max_lat=29.6)

# location_accuracy string -> (km, confidence)
ACC = {
    "exact": (0.1, 1.00), "1km": (1.0, 0.85), "5km": (5.0, 0.60),
    "10km": (10.0, 0.45), "25km": (25.0, 0.30), "50km": (50.0, 0.20),
    "100km": (100.0, 0.10), "250km": (250.0, 0.05), "unknown": (25.0, 0.25),
}
# Only rainfall-driven mass movements belong in a rainfall-triggered model.
DROP_TRIGGERS = {"earthquake", "volcano", "dam_embankment_collapse",
                 "construction", "mining", "freeze_thaw", "snowfall_snowmelt",
                 "no_apparent_trigger", "other", "unknown"}


def load_glc(path):
    d = pd.read_csv(path, low_memory=False)
    d = d[d.longitude.between(BBOX["min_lon"], BBOX["max_lon"]) &
          d.latitude.between(BBOX["min_lat"], BBOX["max_lat"])].copy()
    d["event_date"] = pd.to_datetime(d.event_date, errors="coerce", format="mixed")
    d = d[d.event_date.notna()]

    acc = d.location_accuracy.fillna("unknown").map(ACC)
    d["loc_accuracy_km"] = [a[0] for a in acc]
    d["label_conf"] = [a[1] for a in acc]

    d["trigger"] = d.landslide_trigger.fillna("unknown")
    d["rain_triggered"] = ~d.trigger.isin(DROP_TRIGGERS)

    out = pd.DataFrame(dict(
        lon=d.longitude.round(5), lat=d.latitude.round(5),
        event_date=d.event_date.dt.strftime("%Y-%m-%d"),
        label_conf=d.label_conf, loc_accuracy_km=d.loc_accuracy_km,
        trigger=d.trigger, size=d.landslide_size.fillna("unknown"),
        category=d.landslide_category.fillna("unknown"),
        fatalities=d.fatality_count.fillna(0).astype(int),
        country=d.country_name, admin=d.admin_division_name,
        rain_triggered=d.rain_triggered, source="NASA_GLC",
    ))
    out["sample_id"] = [hashlib.md5(f"{a}{b}{c}".encode()).hexdigest()[:16]
                        for a, b, c in zip(out.lon, out.lat, out.event_date)]
    return out.drop_duplicates("sample_id").reset_index(drop=True)


if __name__ == "__main__":
    import sys
    src = sys.argv[1] if len(sys.argv) > 1 else \
        "data/raw/inventory/Global_Landslide_Catalog_Export_rows.csv"
    p = load_glc(src)
    p.to_csv("data/interim/positives_all.csv", index=False)

    india = p[p.country == "India"]
    rain = p[p.rain_triggered]
    good = p[p.label_conf >= 0.45]
    print(f"NER bbox positives      : {len(p)}")
    print(f"  in India              : {len(india)}")
    print(f"  rainfall-triggered    : {len(rain)}")
    print(f"  location <= 10 km     : {len(good)}")
    print(f"  both (modelling set)  : {len(p[p.rain_triggered & (p.label_conf >= 0.45)])}")
    print(f"  date range            : {p.event_date.min()} .. {p.event_date.max()}")
    print("\nby state (India only):")
    print(india.admin.value_counts().head(10).to_string())
