#!/usr/bin/env bash
# setup_my_parts.sh
# Scaffolds ONLY your two workstreams: #4 Remote Sensing/Vision and the
# dashboard/map slice of #6 Product/Integration. Doesn't touch anything
# owned by other team members — merge with the full repo once the team
# agrees on the overall layout.
#
#   chmod +x setup_my_parts.sh
#   ./setup_my_parts.sh

set -e

echo "Scaffolding your parts (#4 Vision + #6 Dashboard)..."

# --- #4 Remote Sensing / Vision ------------------------------------------
mkdir -p src/models/vision
mkdir -p data/external

touch src/models/vision/__init__.py

if [ ! -f src/models/vision/feasibility_check.py ]; then
  echo "  (move your feasibility_check.py here: mv feasibility_check.py src/models/vision/)"
fi

cat > src/models/vision/extract_cv_features.py << 'EOF'
# Owner: #4 Remote Sensing / Vision
# Once feasibility_check.py confirms the pull works, this turns it into a
# reusable function: given a list of (lat, lon) points, return a DataFrame
# with ndvi, elevation, slope, landcover_class per point — ready to merge
# into the shared feature table for the #2 XGBoost model.
#
# Output should land in: data/external/satellite_features.csv

def extract_features(locations: list[dict]) -> "pandas.DataFrame":
    """
    locations: [{"name": ..., "lat": ..., "lon": ...}, ...]
    returns: DataFrame with one row per location, columns =
             [name, lat, lon, ndvi, elevation_m, slope_deg, landcover_class]
    """
    raise NotImplementedError("fill in using the working parts of feasibility_check.py")
EOF

if [ ! -f src/models/vision/locations.csv ]; then
  echo "name,lat,lon" > src/models/vision/locations.csv
  echo "  created: src/models/vision/locations.csv (replace placeholder rows with real GSI/COOLR points)"
fi

# --- #6 Product/Integration (dashboard + map slice) ------------------------
mkdir -p app/components
mkdir -p app/assets

if [ ! -f app/dashboard.py ]; then
cat > app/dashboard.py << 'EOF'
# Owner: #6 Product/Integration (your slice)
# Streamlit entry point. Wires together:
#   - a location picker (map click or dropdown)
#   - the saved risk model (artifacts/models/ — owned by #2, read-only for you)
#   - components/map_view.py, risk_card.py, forecast_panel.py
#
# Run with: streamlit run app/dashboard.py

import streamlit as st

st.set_page_config(page_title="Landslide Risk", layout="wide")
st.title("Landslide Risk Prediction & Early Warning")

st.write("Scaffold only — wire up map_view, risk_card, forecast_panel here.")
EOF
fi

for f in map_view risk_card forecast_panel; do
  if [ ! -f "app/components/${f}.py" ]; then
    echo "# Owner: #6 — ${f} component, imported by app/dashboard.py" > "app/components/${f}.py"
  fi
done

touch app/components/__init__.py

echo ""
echo "Done. Your working tree:"
echo ""
echo "  src/models/vision/"
echo "    __init__.py"
echo "    feasibility_check.py       <- move your existing script here"
echo "    extract_cv_features.py     <- fill in once feasibility check passes"
echo "    locations.csv              <- your real coordinate list goes here"
echo ""
echo "  data/external/"
echo "    satellite_features.csv     <- output of extract_cv_features.py (hand this to #2)"
echo ""
echo "  app/"
echo "    dashboard.py                <- streamlit entry point"
echo "    components/"
echo "      map_view.py"
echo "      risk_card.py"
echo "      forecast_panel.py"
