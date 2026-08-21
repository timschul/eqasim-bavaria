import pandas as pd
import numpy as np
import itertools

"""
This stage merge prepared datasets of employees from Kreis level 
with inhabitants from Gemeinde level using Iterative Proportional Fitting
"""

def configure(context):
    context.stage("bavaria.ipf.prepare")
    context.config("bavaria.minimum_age.employment", 16)
 
def execute(context):
    df_population, df_employment, df_licenses_country, df_licenses_kreis = context.stage("bavaria.ipf.prepare")

    # Construct a combined age class
    population_age_classes = np.sort(df_population["age_class"].unique())
    population_age_upper = list(population_age_classes[1:]) + [9999]

    employment_age_classes = np.sort(df_employment["age_class"].unique())
    employment_age_upper = list(employment_age_classes[1:]) + [9999]

    minimum_employment_age = context.config("bavaria.minimum_age.employment")

    license_age_classes = np.sort(df_licenses_country["age_class"].unique())
    license_age_upper = list(license_age_classes[1:]) + [9999]
    
    combined_age_classes = np.array(np.sort(list(
        set(population_age_classes) | 
        set(employment_age_classes) |
        set(license_age_classes) | 
        set([minimum_employment_age]))))
    
    population_age_mapping = {}
    employment_age_mapping = {}
    license_age_mapping = {}

    for age_class in combined_age_classes:
        population_age_mapping[age_class] = population_age_classes[np.count_nonzero(population_age_upper <= age_class)]
        employment_age_mapping[age_class] = employment_age_classes[np.count_nonzero(employment_age_upper <= age_class)]
        license_age_mapping[age_class] = license_age_classes[np.count_nonzero(license_age_upper <= age_class)]

    # Construct other unique values
    unique_sexes = np.sort(list(set(df_population["sex"]) | set(df_employment["sex"])))
    unique_employed = [True, False]
    unique_communes = np.sort(df_population["commune_index"].unique())
    unique_rasters = np.sort(df_population["raster_index"].unique())
    unique_departements = np.sort(df_employment["departement_index"].unique())
    unique_license = [True, False]

    # Initialize the seed with all combinations of values
    index = pd.MultiIndex.from_product([
        unique_rasters, unique_sexes, combined_age_classes, unique_employed, unique_license
    ], names = ["raster_index", "sex", "combined_age_class", "employed", "license"])


    df_model = pd.DataFrame(index = index).reset_index()
    df_model["weight"] = 1.0

    # Provide a prior based on the size of the age classes
    combined_age_classes_sizes = {
        lower: upper - lower for
        lower, upper in zip(combined_age_classes[:-1], combined_age_classes[1:])
    }
    combined_age_classes_sizes[combined_age_classes[-1]] = 1.0
    df_model["weight"] *= df_model["combined_age_class"].apply(lambda c: combined_age_classes_sizes[c])

    # Attach departement indices
    df_spatial = df_population[["commune_index", "departement_index", "raster_index"]].drop_duplicates()
    # Attach commune and department indices using raster_index as the reference
    df_model["commune_index"] = df_model["raster_index"].replace(dict(zip(
        df_spatial["raster_index"], df_spatial["commune_index"]
    )))

    df_model["departement_index"] = df_model["raster_index"].replace(dict(zip(
        df_spatial["raster_index"], df_spatial["departement_index"]
    )))

    # Attach individual age classes
    df_model["age_class_population"] = df_model["combined_age_class"].replace(population_age_mapping)
    df_model["age_class_employment"] = df_model["combined_age_class"].replace(employment_age_mapping)
    df_model["age_class_license"] = df_model["combined_age_class"].replace(license_age_mapping)

    # Initialize weighting selectors and targets
    selectors = []
    targets = []
    
    # Population constraints
    combinations = list(itertools.product(unique_rasters, unique_sexes, population_age_classes))
    chunk_size = 1000  # Process combinations in chunks to reduce memory usage
    
    for chunk_start in context.progress(range(0, len(combinations), chunk_size), label = "Generating population constraints"):
        chunk_end = min(chunk_start + chunk_size, len(combinations))
        chunk_combinations = combinations[chunk_start:chunk_end]
        
        for combination in chunk_combinations:
            raster_idx, sex_val, age_class = combination
            
            # Get reference weight
            f_reference = (df_population["raster_index"] == raster_idx) & \
                        (df_population["sex"] == sex_val) & \
                        (df_population["age_class"] == age_class)
            target_weight = df_population.loc[f_reference, "weight"].sum()
            
            # Create model filter more efficiently
            f_model = np.zeros(len(df_model), dtype=bool)
            mask = (df_model["raster_index"] == raster_idx) & \
                  (df_model["sex"] == sex_val) & \
                  (df_model["age_class_population"] == age_class)
            f_model[mask] = True
            
            selectors.append(np.where(f_model)[0])
            targets.append(target_weight)
            
            # Clear memory
            del f_model
            del mask

    # Employment constraints   
    combinations = list(itertools.product(unique_departements, unique_sexes, employment_age_classes))
    for combination in context.progress(combinations, total = len(combinations), label = "Generating employment constraints"):
        departement_idx, sex_val, age_class = combination
        
        # Get reference weight
        f_reference = (df_employment["departement_index"] == departement_idx) & \
                     (df_employment["sex"] == sex_val) & \
                     (df_employment["age_class"] == age_class)
        target_weight = df_employment.loc[f_reference, "weight"].sum()
        
        # Create model filter
        f_model = np.zeros(len(df_model), dtype=bool)
        mask = (df_model["departement_index"] == departement_idx) & \
               (df_model["sex"] == sex_val) & \
               (df_model["age_class_employment"] == age_class) & \
               df_model["employed"]
        f_model[mask] = True
        
        selectors.append(np.where(f_model)[0])
        targets.append(target_weight)
        
        del f_model
        del mask

    # License country constraints
    combinations = list(itertools.product(unique_sexes, license_age_classes))
    for combination in context.progress(combinations, total = len(combinations), label = "Generating license constraints"):
        sex_val, age_class = combination
        
        # Get reference weight
        f_reference = (df_licenses_country["sex"] == sex_val) & \
                     (df_licenses_country["age_class"] == age_class)
        target_weight = df_licenses_country.loc[f_reference, "weight"].sum()
        
        # Create model filter
        f_model = np.zeros(len(df_model), dtype=bool)
        mask = (df_model["sex"] == sex_val) & \
               (df_model["age_class_license"] == age_class) & \
               df_model["license"]
        f_model[mask] = True
        
        selectors.append(np.where(f_model)[0])
        targets.append(target_weight)
        
        del f_model
        del mask

    # License Kreis constraints
    for departement_index in context.progress(unique_departements, total = len(unique_departements), label = "Generating license constraints per Kreis"):
        # Get reference weight
        f_reference = df_licenses_kreis["departement_index"] == departement_index
        target_weight = df_licenses_kreis.loc[f_reference, "weight"].sum()
        
        # Create model filter
        f_model = np.zeros(len(df_model), dtype=bool)
        mask = (df_model["departement_index"] == departement_index) & \
               df_model["license"]
        f_model[mask] = True
        
        selectors.append(np.where(f_model)[0])
        targets.append(target_weight)
        
        del f_model
        del mask

    # Transform to index-based - no longer needed since we're already using indices
    selectors = [s for s in selectors]
    
    
    # Perform IPF
    iteration = 0
    converged = False
    weights = df_model["weight"].values

    while iteration < 1000:
        iteration_factors = []
    
        for f, target_weight in zip(selectors, targets):
            current_weight = np.sum(weights[f])
    
            if current_weight > 0:
                update_factor = target_weight / current_weight
                weights[f] *= update_factor
                iteration_factors.append(update_factor)

        print(
            "Iteration:", iteration,
            "factors:", len(iteration_factors),
            "mean:", np.mean(iteration_factors),
            "min:", np.min(iteration_factors),
            "max:", np.max(iteration_factors))
        
        if np.max(iteration_factors) - 1 < 1e-2:
            if np.min(iteration_factors) > 1 - 1e-2:
                converged = True
                break
    
        iteration += 1

    df_model["weight"] = weights

    assert converged

    # Reestablish sex categories
    df_model["sex"] = df_model["sex"].replace({ 1: "male", 2: "female" }).astype("category")

    # Add identifiers
    df_model = pd.merge(df_model, df_population[["raster_index", "raster_id"]].drop_duplicates(), on = "raster_index", how = "left")
    assert np.count_nonzero(df_model["raster_id"].isna()) == 0
    
    df_model = pd.merge(df_model, df_population[["commune_index", "commune_id"]].drop_duplicates(), on = "commune_index", how = "left")
    assert np.count_nonzero(df_model["commune_id"].isna()) == 0

    df_model = pd.merge(df_model, df_population[["departement_index", "departement_id"]].drop_duplicates(), on = "departement_index", how = "left")
    assert np.count_nonzero(df_model["departement_id"].isna()) == 0

    df_model = df_model.rename(columns = { "combined_age_class": "age_class" })
    return df_model[[ "raster_id", "commune_id", "departement_id", "sex", "age_class", "employed", "license", "weight"]]
