import pandas as pd
import numpy as np
import os

import geopandas as gpd
import zipfile
from shapely.geometry import Point
from shapely.wkb import loads
import pyogrio

"""
This stage loads the raw census data for Bavaria.

TODO: This could be replaced with a Germany-wide extract from GENESIS
"""

def configure(context):
    context.stage("bavaria.data.spatial.codes")

    context.config("data_path")
    context.config("bavaria.population_path", "bavaria/meldedaten_altersgruppen_mid.csv")
    context.config("gemeinde_path", "datalake/grenze_gemeinde.zip")
    #context.config("bavaria.population_path", "bavaria/a1310c_202200.xla")

def construct_municipality_id(municipality_code, association_code):
    if len(municipality_code) == 3:
        # a city without a Kreis, pad with zeros
        return "09" + municipality_code + "0000000" 
    elif len(municipality_code) == 6:
        # a regular Gemeinde with a Kreis

        if association_code == "-":
            # the Gemeinde is not in an association (Verbund)
            return "".join([
                "09", # Bavaria
                municipality_code[0:3], # First digit (Bezirk) + two digits (Kreis)
                "0", # indicating that it is not in a Verbund
                municipality_code[3:], # Repeat last three digits (Gemeinde)
                municipality_code[3:], # Repeat last three digits (Gemeinde)
            ])
        
        else:
            # the Gemeinde is in an association (Verbund)
            return "".join([
                "09", # Bavaria
                municipality_code[0:3], # First digit (Bezirk) + two digits (Kreis)
                "5", # indicating that it is in a Verbund
                str(association_code), # the association code
                municipality_code[3:], # Repeat last three digits (Gemeinde)
            ])

    raise RuntimeError("Invalid format")


# Convert coordinates to Point geometries
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
        
        # Create initial melted dataframe
        df_census = pd.melt(
            df_joined_fil, 
            id_vars=["commune_id", "raster_id"], 
            value_vars=['age_0', 'age_18', 'age_30', 'age_50', 'age_65', 'age_75'],
            var_name='age_class', 
            value_name='total_population'
        )

        # Clean up age_class
        df_census['age_class'] = df_census['age_class'].str.replace('age_', '').astype(int)

        # Split population evenly between male and female
        df_census['female_population'] = df_census['total_population'] / 2
        df_census['male_population'] = df_census['total_population'] / 2

        # Create separate male and female dataframes
        df_male = df_census[["commune_id", "raster_id", "age_class", "male_population"]].rename(columns={
            "male_population": "weight"
        })
        df_male["sex"] = "male"

        df_female = df_census[["commune_id", "raster_id", "age_class", "female_population"]].rename(columns={
            "female_population": "weight"
        })
        df_female["sex"] = "female"

        # Combine male and female data
        df_census = pd.concat([df_male, df_female])
        df_census["sex"] = df_census["sex"].astype("category")


        return df_joined_fil, df_census1, df_gemeinde, df_census
    

"""
def execute(context):
    df_census = pd.read_excel("{}/{}".format(
        context.config("data_path"), context.config("bavaria.population_path")
    ), sheet_name = "Gemeinden", skiprows = 5, names = [
        "municipality_code", "association_code", "name", "sex", "total", 
        "age_0", "age_3", "age_6","age_10", "age_15", "age_18", "age_20", "age_25", "age_30", "age_40", "age_50", "age_65", "age_75", 
        "municipality_code_copy", "association_code_copy"
    ])

    # Only keep rows where we have a value
    df_census = df_census[~df_census["total"].isna()].copy()
    
    # Padding of identifiers, only one following line
    df_census["municipality_code"] = df_census["municipality_code"].ffill(limit = 1)
    df_census["association_code"] = df_census["association_code"].ffill(limit = 1)
    
    # Only keep rows where we have 6 digits (Bezirk + Kreis + Gemeinde) or 3 digits (city without Kreis)
    df_census = df_census[
        (df_census["municipality_code"].str.len() == 6) | 
        (df_census["municipality_code"].str.len() == 3)
    ].copy()

    # Now reconstruct the municipality code (ARS, the first column gives the AGS!)
    # All municipalities that are without a Kreis get a 0 suffix
    df_census["commune_id"] = [
        construct_municipality_id(*codes) for codes in zip(
            df_census["municipality_code"], df_census["association_code"]
        ) 
    ]

    df_census["commune_id"] = df_census["commune_id"].astype("category")

    # Clean up age structure
    df_census = pd.melt(df_census, ["commune_id", "sex"], [
        "age_0", "age_3", "age_6","age_10", "age_15", "age_18", "age_20", "age_25", "age_30", "age_40", "age_50", "age_65", "age_75"
    ], var_name = "age_class", value_name = "population")

    df_census["age_class"] = df_census["age_class"].str.replace("age_", "").astype(int)

    # Clean counts
    df_census["population"] = df_census["population"].replace({ "-": 0 }).astype(int)

    # Cleanup gender
    df_census["sex"] = df_census["sex"].replace({
        "  insgesamt": "total", "  weiblich": "female"
    })

    df_census = pd.merge(
        df_census[df_census["sex"] == "total"].rename(columns = { "population": "total_population" }).drop(columns = ["sex"]),
        df_census[df_census["sex"] == "female"].rename(columns = { "population": "female_population" }).drop(columns = ["sex"]),
        on = ["commune_id", "age_class"]
    )
    
    df_census["male_population"] = df_census["total_population"] - df_census["female_population"]

    df_male = df_census[["commune_id", "age_class", "male_population"]].rename(columns = {
        "male_population": "weight"
    })
    df_male["sex"] = "male"

    df_female = df_census[["commune_id", "age_class", "female_population"]].rename(columns = {
        "female_population": "weight"
    })
    df_female["sex"] = "female"

    df_census = pd.concat([df_male, df_female])
    df_census["sex"] = df_census["sex"].astype("category")

    # Filter for requested codes
    df_codes = context.stage("bavaria.data.spatial.codes")
    df_census = df_census[df_census["commune_id"].isin(df_codes["commune_id"])]

    return df_census[["commune_id", "sex", "age_class", "weight"]]

"""    

def validate(context):
    if not os.path.exists("{}/{}".format(context.config("data_path"), context.config("bavaria.population_path"))):
        raise RuntimeError("Bavarian census data is not available")

    return os.path.getsize("{}/{}".format(context.config("data_path"), context.config("bavaria.population_path")))

