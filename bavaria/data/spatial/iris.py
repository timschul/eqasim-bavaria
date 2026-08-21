"""
Generates the IRIS zoning system that is not used in Germany. Instead, we create one
fake IRIS for each municipality in Germany. See the `codes` stage for more information.
"""

def configure(context):
    context.stage("bavaria.data.census.population")

def execute(context):
    # Load shapes
    df_joined_fil, _, _, _ = context.stage("bavaria.data.census.population")
    df = df_joined_fil[["commune_id", "raster_id", "geometry"]].rename(
        columns={"commune_id": "municipality_code"}
    )

    #df = context.stage("bavaria.data.population.raw")[["municipality_code", "geometry"]]

    # Clean up identifiers
    df["commune_id"] = df["municipality_code"].astype("category")
    df["raster_id"] = df["raster_id"].astype("category")

    # Fake IRIS
    df["iris_id"] = df["commune_id"].astype(str) + "0000"
    df["iris_id"] = df["iris_id"].astype("category")

    # Departement identifiers
    df["departement_id"] = df["commune_id"].str[:5]

    # Region dummu
    df["region_id"] = 1
    df["region_id"] = df["region_id"].astype("category")

    return df[["iris_id", "commune_id", "raster_id", "departement_id",  "geometry"]]
