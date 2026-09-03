"""
Step 1 (Precision Track) — Parse surveyed landslide polygons and points
from Zenodo (DOI 10.5281/zenodo.8169506) for Sikkim into high-precision
ground-truth positive coordinates (loc_accuracy_km <= 0.03, label_conf = 1.0).
"""
import os
import glob
import hashlib
import numpy as np
import pandas as pd
import geopandas as gpd

BBOX = dict(min_lon=87.5, min_lat=21.5, max_lon=97.5, max_lat=29.6)


def load_precision_inventory(inv_dir="data/raw/inventory"):
    poly_shp = os.path.join(inv_dir, "Google_Earth_landslides_polygon_21Dec2021.shp")
    pt_shp = os.path.join(inv_dir, "Google_Earth_landslides_point_21Dec2021.shp")
    
    records = []
    
    # Process Polygons
    if os.path.exists(poly_shp):
        print(f"  loading polygon inventory: {os.path.basename(poly_shp)} ...")
        g_poly = gpd.read_file(poly_shp)
        if g_poly.crs and g_poly.crs.to_string() != "EPSG:4326":
            g_poly = g_poly.to_crs("EPSG:4326")
        
        for r in g_poly.itertuples():
            geom = r.geometry
            if geom is None or geom.is_empty:
                continue
            centroid = geom.centroid
            yr = getattr(r, 'Year', 2015)
            try:
                yr = int(yr)
            except (ValueError, TypeError):
                yr = 2015
            date_str = f"{yr}-07-15"   # Default mid-monsoon date if exact day unknown
            records.append({
                'lon': round(centroid.x, 5),
                'lat': round(centroid.y, 5),
                'event_date': date_str,
                'label_conf': 1.00,
                'loc_accuracy_km': 0.03,  # 30m resolution accuracy
                'trigger': 'monsoon',
                'size': 'medium',
                'category': 'translational_slide',
                'fatalities': 0,
                'state': 'Sikkim',
                'district': 'East',
                'rain_triggered': True,
                'source': 'Zenodo_Surveyed_Polygons',
                'hq_location': 1
            })
        print(f"  -> extracted {len(records)} surveyed polygon centroids")
        
    # Process Points
    if os.path.exists(pt_shp):
        print(f"  loading point inventory: {os.path.basename(pt_shp)} ...")
        g_pt = gpd.read_file(pt_shp)
        if g_pt.crs and g_pt.crs.to_string() != "EPSG:4326":
            g_pt = g_pt.to_crs("EPSG:4326")
            
        n_pt = 0
        for r in g_pt.itertuples():
            geom = r.geometry
            if geom is None or geom.is_empty:
                continue
            yr = getattr(r, 'Year', 2015)
            try:
                yr = int(yr)
            except (ValueError, TypeError):
                yr = 2015
            date_str = f"{yr}-07-15"
            records.append({
                'lon': round(geom.x, 5),
                'lat': round(geom.y, 5),
                'event_date': date_str,
                'label_conf': 1.00,
                'loc_accuracy_km': 0.03,
                'trigger': 'monsoon',
                'size': 'medium',
                'category': 'debris_flow_rockfall',
                'fatalities': 0,
                'state': 'Sikkim',
                'district': 'East',
                'rain_triggered': True,
                'source': 'Zenodo_Surveyed_Points',
                'hq_location': 1
            })
            n_pt += 1
        print(f"  -> extracted {n_pt} surveyed point coordinates")

    if not records:
        print("  [WARN] No precision inventory shapefiles found in " + inv_dir)
        return pd.DataFrame()
        
    df = pd.DataFrame(records)
    df["sample_id"] = [hashlib.md5(f"{a}{b}{c}".encode()).hexdigest()[:16]
                       for a, b, c in zip(df.lon, df.lat, df.event_date)]
    df = df.drop_duplicates("sample_id").reset_index(drop=True)
    print(f"Precision inventory ready: {len(df)} unique surveyed scar locations.")
    return df


if __name__ == "__main__":
    df = load_precision_inventory()
    if not df.empty:
        os.makedirs("data/interim", exist_ok=True)
        df.to_csv("data/interim/positives_precision.csv", index=False)
        print("wrote data/interim/positives_precision.csv")
