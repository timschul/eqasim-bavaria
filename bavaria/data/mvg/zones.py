import os, json, re
import geopandas as gpd
import pandas as pd
import shapely.geometry as sgeo

"""
Creates 2 zones for Hof:
- Landkreis
- Stadt

This replaces the MVG zones for Munich and creates transit zones
"""

def configure(context):
    context.config("data_path")
    context.config("hof_landkreis_path", 
        os.path.join(context.config("data_path"), "bavaria/zones/Landkreis_Stadt_Hof_difference.json"))
    context.config("hof_stadt_path", 
        os.path.join(context.config("data_path"), "bavaria/zones/Stadt_Hof.json"))

def execute(context):
    # Load GeoJSON files directly with geopandas
    landkreis_gdf = gpd.read_file(context.config("hof_landkreis_path"))
    stadt_gdf = gpd.read_file(context.config("hof_stadt_path"))
    
    # Create DataFrame for both zones
    df_zones = []
    
    # Add Stadt (inner city)
    df_zones.append({
        'name': 'stadt',
        'id': 'stadt',
        'zone': '1',  # Changed to match Java expectation
        'geometry': stadt_gdf.geometry.unary_union.buffer(0)  # Fix potential geometry issues
    })
    
    # Add Landkreis (county)
    df_zones.append({
        'name': 'landkreis',
        'id': 'landkreis',
        'zone': '2',  # Changed to match Java expectation
        'geometry': landkreis_gdf.geometry.unary_union.buffer(0)  # Fix potential geometry issues
    })
    
    # Convert to GeoDataFrame
    df_zones = gpd.GeoDataFrame(pd.DataFrame.from_records(df_zones), crs="EPSG:4326")    
    # Convert to target CRS
    df_zones = df_zones.to_crs("EPSG:25832")

    print("\nZone Information:")
    print(df_zones[['name', 'zone']].to_string())
    
    # Return original zones for population synthesis
    return df_zones[["name","zone", "geometry"]]  # Return with 'name' and 'zone' columns

def validate(context):
    landkreis_path = context.config("hof_landkreis_path")
    stadt_path = context.config("hof_stadt_path")
    
    files_exist = (
        os.path.exists(landkreis_path) and 
        os.path.exists(stadt_path)
    )
    if not files_exist:
        print(f"Missing files:")
        if not os.path.exists(landkreis_path):
            print(f"- {landkreis_path}")
        if not os.path.exists(stadt_path):
            print(f"- {stadt_path}")
        raise RuntimeError("Hof zone data files are not available")
    return True