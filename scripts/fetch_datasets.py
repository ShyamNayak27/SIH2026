"""
Script to download and organize high-precision datasets for NER Landslide modelling:
1. Geofabrik full OSM extract shapefile for North-Eastern Zone (100% road/river coverage)
2. Zenodo multi-temporal surveyed landslide polygon inventory for Sikkim (DOI 10.5281/zenodo.8169506)
3. Copernicus GLO-30 30m DEM tiles via AWS S3 public COG HTTPS bucket
"""
import os
import sys
import zipfile
import urllib.request

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAW_DIR = os.path.join(ROOT, "data", "raw")

OSM_DIR = os.path.join(RAW_DIR, "osm")
INV_DIR = os.path.join(RAW_DIR, "inventory")
DEM_DIR = os.path.join(RAW_DIR, "dem_30m")

for d in (OSM_DIR, INV_DIR, DEM_DIR):
    os.makedirs(d, exist_ok=True)


def download_file(url, dest_path, desc="File"):
    if os.path.exists(dest_path) and os.path.getsize(dest_path) > 0:
        print(f"[SKIP] {desc} already exists: {os.path.basename(dest_path)}")
        return True
    print(f"[FETCH] Downloading {desc} from {url} ...")
    try:
        req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
        with urllib.request.urlopen(req) as resp, open(dest_path, 'wb') as out_file:
            chunk_size = 1024 * 1024
            while True:
                chunk = resp.read(chunk_size)
                if not chunk:
                    break
                out_file.write(chunk)
        print(f"  -> Saved {dest_path} ({os.path.getsize(dest_path) / (1024*1024):.2f} MB)")
        return True
    except Exception as e:
        print(f"  [ERROR] Failed to download {desc}: {e}")
        if os.path.exists(dest_path):
            os.remove(dest_path)
        return False


def fetch_osm():
    url = "https://download.geofabrik.de/asia/india/north-eastern-zone-latest-free.shp.zip"
    zip_path = os.path.join(OSM_DIR, "north-eastern-zone-latest-free.shp.zip")
    if download_file(url, zip_path, "Geofabrik OSM Shapefile ZIP"):
        # Unzip key layers
        print("  Extracting OSM shapefiles...")
        try:
            with zipfile.ZipFile(zip_path, 'r') as z:
                for item in z.namelist():
                    if any(item.startswith(p) for p in ["gis_osm_roads_", "gis_osm_waterways_"]):
                        z.extract(item, OSM_DIR)
            print("  -> OSM road and waterway shapefiles extracted successfully.")
        except Exception as e:
            print(f"  [ERROR] Unzipping OSM extract failed: {e}")


def fetch_zenodo_sikkim():
    base_api = "https://zenodo.org/api/records/8169506/files/"
    files = [
        "Google_Earth_landslides_polygon_21Dec2021.shp",
        "Google_Earth_landslides_polygon_21Dec2021.shx",
        "Google_Earth_landslides_polygon_21Dec2021.dbf",
        "Google_Earth_landslides_polygon_21Dec2021.prj",
        "Google_Earth_landslides_polygon_21Dec2021.csv",
        "Google_Earth_landslides_point_21Dec2021.shp",
        "Google_Earth_landslides_point_21Dec2021.shx",
        "Google_Earth_landslides_point_21Dec2021.dbf",
        "Google_Earth_landslides_point_21Dec2021.prj",
        "Google_Earth_landslides_point_21Dec2021.csv",
    ]
    for fn in files:
        url = f"{base_api}{fn}/content"
        dest = os.path.join(INV_DIR, fn)
        download_file(url, dest, f"Zenodo Sikkim Inventory ({fn})")


def fetch_copernicus_30m():
    # Primary bounding tiles covering main NER landslide hot-spots (lat 23-28, lon 88-96)
    tiles = [
        (27, 88), (27, 89), (27, 90), (27, 91), (27, 92), (27, 93), (27, 94), (27, 95),
        (26, 88), (26, 91), (26, 92), (26, 93), (26, 94), (26, 95),
        (25, 88), (25, 89), (25, 90), (25, 91), (25, 92), (25, 93), (25, 94),
        (24, 88), (24, 91), (24, 92), (24, 93), (24, 94),
        (23, 88), (23, 91), (23, 92), (23, 93), (23, 94)
    ]
    base_s3 = "https://copernicus-dem-30m.s3.eu-central-1.amazonaws.com/"
    success = 0
    for lat, lon in tiles:
        tile_name = f"Copernicus_DSM_COG_10_N{lat:02d}_00_E{lon:03d}_00_DEM"
        url = f"{base_s3}{tile_name}/{tile_name}.tif"
        dest = os.path.join(DEM_DIR, f"{tile_name}.tif")
        if download_file(url, dest, f"Copernicus 30m DEM Tile (N{lat}E{lon})"):
            success += 1
    print(f"Copernicus 30m DEM: {success}/{len(tiles)} tiles fetched.")


if __name__ == "__main__":
    print("==================================================")
    print("  NER Landslide Platform: Data Acquisition Script ")
    print("==================================================")
    print("1. Fetching Zenodo Precision Landslide Inventory...")
    fetch_zenodo_sikkim()
    print("\n2. Fetching Geofabrik Complete OSM Vector Shapefiles...")
    fetch_osm()
    print("\n3. Fetching Copernicus GLO-30 (30m DEM) Rasters...")
    fetch_copernicus_30m()
    print("\nData acquisition stage complete.")
