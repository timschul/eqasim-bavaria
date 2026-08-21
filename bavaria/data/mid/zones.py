import geopandas as gpd
import os

"""
This stage provides a zoning system for Hof based on the created gpkg file. The 2 zones are:
- Landkreis
- Stadt

"""

def configure(context):
    pass

def execute(context):
    self_path = os.path.dirname(os.path.abspath(__file__))
    df_zones = gpd.read_file("{}/landkreis_stadt_hof.gpkg".format(self_path))

    # Map numeric IDs to zone names
    df_zones["name"] = df_zones["id"].replace({
        0: "landkreis",
        1: "stadt"
    })

    # Print debug information
    print("\nMID Zones Information:")
    print("Columns:", df_zones.columns.tolist())
    print("CRS:", df_zones.crs)
    print("\nZone counts:")
    print(df_zones["name"].value_counts())
    
    # Verify geometries
    print("\nGeometry checks:")
    for _, zone in df_zones.iterrows():
        print(f"\nZone: {zone['name']}")
        print(f"Geometry type: {zone.geometry.geom_type}")
        print(f"Is valid: {zone.geometry.is_valid}")
        if not zone.geometry.is_valid:
            print("Fixing invalid geometry...")
            df_zones.loc[df_zones.index == _, "geometry"] = zone.geometry.buffer(0)
    
    # Ensure the CRS is correct
    if df_zones.crs is None:
        print("Warning: No CRS found, setting to EPSG:4326")
        df_zones.set_crs("EPSG:4326", inplace=True)
    
    # Convert to target CRS if needed
    df_zones = df_zones.to_crs("EPSG:25832")
    
    return df_zones[["name", "geometry"]]

def validate(context):
    self_path = os.path.dirname(os.path.abspath(__file__))
    path = "{}/landkreis_stadt_hof.gpkg".format(self_path)
    
    if not os.path.exists(path):
        raise RuntimeError(f"Zone file not found: {path}")
    
    return True

"""
    df_zones["name"] = df_zones["name"].replace({
        "mittlerer_ring": "mr",
        "mittlerer_ring_to_city_boundary": "mrs"
    })

"""   

