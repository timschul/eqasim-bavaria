import os
import geopandas as gpd
import zipfile
import numpy as np

from shapely.geometry import Point
import pandas as pd
from shapely.wkb import loads
import pyogrio

"""
This stages loads a file containing population data for Germany including the adminstrative codes.
"""

def configure(context):
    context.config("data_path")
    context.config("bavaria.population_path", "bavaria/meldedaten_altersgruppen_mid.csv")
    context.config("gemeinde_path", "datalake/grenze_gemeinde.zip")

    context.config("bavaria.political_prefix", ["091", "092", "093", "094", "095", "096", "097"])
    context.config("germany.population_path", "germany/vg250-ew_12-31.utm32s.gpkg.ebenen.zip")
    context.config("germany.population_source", "vg250-ew_12-31.utm32s.gpkg.ebenen/vg250-ew_ebenen_1231/DE_VG250.gpkg")


def parse_coordinates_to_point(coord_str):
    # Convert the string representation of a tuple to an actual tuple
    coords = eval(coord_str)  # This safely converts the string like "(4454750, 3035250)" to a tuple (4454750, 3035250)
    return Point(coords[0], coords[1])  # Create a Shapely Point from the coordinates


def execute(context):
    # First process the census raster data
    df = pd.read_csv("{}/{}".format(
        context.config("data_path"), context.config("bavaria.population_path")
    ), sep = ",", decimal = ".")

    # Convert coordinates to Point geometries
    df['sw_corner'] = df['sw_corner'].apply(parse_coordinates_to_point)

    # Convert the geometry column from WKB (hex) to shapely geometries
    df['geometry'] = df['geometry'].apply(lambda x: loads(bytes.fromhex(x)))

    # Convert the DataFrame to a GeoDataFrame
    df_census1 = gpd.GeoDataFrame(df, geometry='geometry')
    df_census1.set_crs("EPSG:3035", inplace=True)

    # Now convert the 'sw_corner' column to Point geometries and set its CRS to EPSG:3035
    gdf_sw_corner = gpd.GeoSeries(df_census1['sw_corner'], crs="EPSG:3035")
    
    # Reproject to target CRS
    df_census1['sw_corner'] = gdf_sw_corner.to_crs("EPSG:25832")
    df_census1 = df_census1.to_crs("EPSG:25832")

    # Then process the municipality shapefile
    zip_path = "{}/{}".format(
        context.config("data_path"), context.config("gemeinde_path")
    )
    
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        file_list = zip_ref.namelist()
        shp_file = next((f for f in file_list if f.endswith('.shp')), None)
        
        if not shp_file:
            raise RuntimeError("No .shp file found in the ZIP archive.")
            
        temp_dir = './temp_shp_files'
        zip_ref.extractall(temp_dir)
        shp_path = os.path.join(temp_dir, shp_file)
        df_gemeinde = pyogrio.read_dataframe(shp_path)
        df_gemeinde = df_gemeinde.to_crs("EPSG:25832")
            
            
        # Here you can do any processing that combines df_census and df_gemeinde


        # Perform spatial join between the raster cells and communes
        # Each raster cell will be assigned the information of the commune it falls within
        df_joined = gpd.sjoin(df_census1, df_gemeinde, how='left', predicate='within')

        # Now df_joined contains all the original columns from df_census 
        # plus the columns from df_gemeinde for the commune that contains each raster cell

        # To verify the results, you could:
        # 1. Count how many raster cells are in each commune
        commune_counts = df_joined.groupby('name').size()
        print("Number of raster cells per commune:")
        print(commune_counts)


        df_joined_fil = df_joined[(df_joined["rs"].str.len() == 12)]

        df_joined_fil = df_joined_fil.copy()
        df_joined_fil = df_joined_fil.reset_index(drop=True)
        df_joined_fil["commune_id"] = df_joined_fil["rs"]
        df_joined_fil["commune_id"] = df_joined_fil["commune_id"].astype("category")
        df_joined_fil["population"] = df_joined_fil[['A0', 'A1','A2', 'A3','A4','A5']].sum(axis=1)
        df_joined_fil = df_joined_fil.rename(columns = { 
            "A0": "age_0",
            "A1": "age_18",
            "A2": "age_30",
            "A3": "age_50",
            "A4": "age_65",
            "A5": "age_75"
        })

        # Create an explicit copy first
        df_joined_fil = df_joined_fil.copy()

        # Reset index
        df_joined_fil = df_joined_fil.reset_index(drop=True)

        # Create raster_id by combining commune_id with zero-padded index
        df_joined_fil['raster_id'] = df_joined_fil['commune_id'].astype(str) + df_joined_fil.index.map(lambda x: f'{x:04d}')

        # Remove rows where departement_id is 09477 (Landkreis Kulmbach)
        df_joined_fil = df_joined_fil[df_joined_fil["commune_id"].str[:5] != "09477"]

        # Reset index after filtering
        df_joined_fil = df_joined_fil.reset_index(drop=True)

        df_joined_fil = df_joined_fil.copy()
        df_population = df_joined_fil[["commune_id", "raster_id", "population"]].rename(
            columns={"commune_id": "municipality_code"}
        )
    return df_population

"""
def execute(context):
    # Load IRIS registry
    with zipfile.ZipFile(
        "{}/{}".format(context.config("data_path"), context.config("germany.population_path"))) as archive:
        with archive.open(context.config("germany.population_source")) as f:
            df_population = gpd.read_file(f, layer = "vg250_gem")[[
                "ARS", "EWZ", "geometry"
            ]]

    # Filter for prefix
    prefix = context.config("bavaria.political_prefix")

    if type(prefix) == str:
        df_population = df_population[df_population["ARS"].str.startswith(prefix)].copy()
    else:
        f = np.zeros((len(df_population,)), dtype = bool)

        for item in prefix:
            f |= df_population["ARS"].str.startswith(item)
        
        df_population = df_population[f].copy()

    # Rename
    df_population = df_population.rename(columns = { 
        "ARS": "municipality_code",
        "EWZ": "population"
    })
    
    return df_population
"""

def validate(context):
    if not os.path.exists("%s/%s" % (context.config("data_path"), context.config("germany.population_path"))):
        raise RuntimeError("German population data is not available")

    return os.path.getsize("%s/%s" % (context.config("data_path"), context.config("germany.population_path")))
