#!/usr/bin/env python3
"""
Convert Wildfire GeoJSON to CSV - California Only
"""

import os
import sys
import geopandas as gpd
import pandas as pd
import warnings

# Suppress warnings
warnings.filterwarnings('ignore')

# Configuration
DATA_DIR = "data"
INPUT_FILE = os.path.join(DATA_DIR, "wildfire_data.geojson")
STATES_FILE = os.path.join(DATA_DIR, "us_states.json")
OUTPUT_FILE = os.path.join(DATA_DIR, "california_wildfire_data.csv")
BUFFER_DEG = 0.74  # Buffer to match analysis (approx 50 miles)

def main():
    print("=" * 60)
    print("WILDFIRE DATA CONVERSION: GEOJSON -> CSV")
    print("=" * 60)

    # 1. Load Data
    if not os.path.exists(INPUT_FILE):
        print(f"❌ Error: Input file not found: {INPUT_FILE}")
        return

    print(f"Loading data from {INPUT_FILE}...")
    try:
        gdf = gpd.read_file(INPUT_FILE)
        print(f"✓ Loaded {len(gdf)} records")
    except Exception as e:
        print(f"❌ Error loading GeoJSON: {e}")
        return

    # 1.5 Convert to Centroids (Match analysis logic)
    # The analysis script converts to centroids BEFORE filtering
    if not gdf.empty and gdf.geometry.iloc[0].geom_type in ['Polygon', 'MultiPolygon']:
        print("Converting polygons to centroids (for accurate spatial filtering)...")
        gdf['geometry'] = gdf.geometry.centroid

    # 2. Filter for California with Buffer
    if os.path.exists(STATES_FILE):
        print(f"Loading states from {STATES_FILE}...")
        try:
            states = gpd.read_file(STATES_FILE)
            california = states[states['name'] == 'California']
            
            if not california.empty:
                # Ensure CRS matches
                if gdf.crs != california.crs:
                    california = california.to_crs(gdf.crs)
                
                # Apply buffer
                print(f"Applying {BUFFER_DEG} degree buffer to California state boundary...")
                filtering_geom = california.copy()
                filtering_geom['geometry'] = california.geometry.buffer(BUFFER_DEG)
                
                # Spatial join
                original_count = len(gdf)
                gdf = gpd.sjoin(gdf, filtering_geom, how='inner', predicate='within')
                
                # Cleanup join columns
                cols_to_drop = ['index_right', 'name', 'density', 'id']
                existing_cols_to_drop = [c for c in cols_to_drop if c in gdf.columns]
                if existing_cols_to_drop:
                    gdf = gdf.drop(columns=existing_cols_to_drop)
                    
                print(f"✓ Filtered to California (+ buffer): {original_count} -> {len(gdf)} records")
            else:
                print("⚠️ Warning: 'California' not found in states file. Skipping filtering.")
        except Exception as e:
            print(f"⚠️ Warning: Error during state filtering: {e}")
    else:
        print(f"⚠️ Warning: States file not found at {STATES_FILE}. Skipping filtering.")

    # 3. Extract Coordinates
    print("Extracting coordinates...")
    gdf['longitude'] = gdf.geometry.x
    gdf['latitude'] = gdf.geometry.y
    
    # 4. Export to CSV
    print(f"Exporting to {OUTPUT_FILE}...")
    
    # Convert to regular DataFrame and drop geometry
    df = pd.DataFrame(gdf.drop(columns='geometry'))
    
    # Reorder columns to put coords first
    cols = ['longitude', 'latitude'] + [c for c in df.columns if c not in ['longitude', 'latitude']]
    df = df[cols]
    
    try:
        df.to_csv(OUTPUT_FILE, index=False)
        print(f"✓ Successfully saved {len(df)} records to CSV")
        print(f"  Path: {os.path.abspath(OUTPUT_FILE)}")
    except Exception as e:
        print(f"❌ Error saving CSV: {e}")

if __name__ == "__main__":
    main()
