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
    unique_departements = np.sort(df_employment["departement_index"].unique())
    unique_license = [True, False]

    # Initialize the seed with all combinations of values
    index = pd.MultiIndex.from_product([
        unique_communes, unique_sexes, combined_age_classes, unique_employed, unique_license
    ], names = ["commune_index", "sex", "combined_age_class", "employed", "license"])

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
    df_spatial = df_population[["commune_index", "departement_index"]].drop_duplicates()
    df_model["departement_index"] = df_model["commune_index"].replace(dict(zip(
        df_spatial["commune_index"], df_spatial["departement_index"]
    )))

    # Attach individual age classes
    df_model["age_class_population"] = df_model["combined_age_class"].replace(population_age_mapping)
    df_model["age_class_employment"] = df_model["combined_age_class"].replace(employment_age_mapping)
    df_model["age_class_license"] = df_model["combined_age_class"].replace(license_age_mapping)

    # --- Fix (D_matsim-magdeburg Ae-112, timschul/eqasim-bavaria fork): the
    # employment and license constraints below only ever selected the *True*
    # side (df_model["employed"] / df_model["license"]) and left the *False*
    # side unconstrained. Wherever a license/employment age class straddles a
    # population age class boundary (e.g. license class 18-20 crossing the
    # population's 18/21 split), the IPF had no anchor stopping it from
    # shifting people across the population age boundary to satisfy the
    # True-side target alone - observed for marego/Sachsen-Anhalt as a
    # doubling of the 18-19 age year (11.300 vs. a flat ~5.730/year census)
    # and a matching hole at age 20 (1.020). Binding the False side too
    # (population total for that bucket minus the True-side target) removes
    # that degree of freedom without touching the True-side targets or the
    # raking loop itself.
    # A population age class is in general coarser than (and not aligned
    # with) the employment/license age classes - that is exactly why
    # combined_age_classes exists. Naively assigning a whole population
    # bracket to whichever side of an employment/license boundary its lower
    # bound falls on would just move the same "which side of the boundary
    # gets the weight" problem one level down. Instead, each population
    # row's weight is split across the combined_age_class sub-brackets it
    # spans, proportional to sub-bracket width - the same uniform-density-
    # within-bracket assumption already used above for the raking seed
    # prior ("Provide a prior based on the size of the age classes"). Only
    # then is it aggregated into employment/license buckets, via the same
    # *_age_mapping dictionaries df_model itself uses.
    population_age_upper_by_class = dict(zip(population_age_classes, population_age_upper))
    combined_by_population_age = {
        population_age: [
            c for c in combined_age_classes
            if population_age <= c < population_age_upper_by_class[population_age]
        ]
        for population_age in population_age_classes
    }

    df_population_expanded = []
    for population_age, sub_classes in combined_by_population_age.items():
        total_width = sum(combined_age_classes_sizes[c] for c in sub_classes)
        subset = df_population[df_population["age_class"] == population_age][["commune_index", "sex", "weight"]]

        for c in sub_classes:
            share = combined_age_classes_sizes[c] / total_width
            expanded = subset.copy()
            expanded["combined_age_class"] = c
            expanded["weight"] = expanded["weight"] * share
            df_population_expanded.append(expanded)

    df_population_expanded = pd.concat(df_population_expanded, ignore_index = True)
    df_population_expanded["departement_index"] = df_population_expanded["commune_index"].replace(dict(zip(
        df_spatial["commune_index"], df_spatial["departement_index"]
    )))
    df_population_expanded["age_class_employment"] = df_population_expanded["combined_age_class"].replace(
        employment_age_mapping)
    df_population_expanded["age_class_license"] = df_population_expanded["combined_age_class"].replace(
        license_age_mapping)

    population_totals_employment = df_population_expanded.groupby(
        ["departement_index", "sex", "age_class_employment"])["weight"].sum()
    population_totals_license_country = df_population_expanded.groupby(
        ["sex", "age_class_license"])["weight"].sum()
    population_totals_departement = df_population_expanded.groupby("departement_index")["weight"].sum()

    def not_side_target(total_lookup, key, true_side_target):
        """Population total for `key` minus the True-side target, i.e. the
        target for the False side of the same binary split. Negative results
        (True-side target exceeds the population total for that bucket, a
        sign of inconsistent input data rather than a modelling choice) are
        reported and clipped to zero rather than silently accepted."""
        total = total_lookup.get(key, 0.0)
        target = total - true_side_target
        if target < 0:
            print("WARNING (Ae-112 fix): False-side target below zero for", key,
                  "- total population", total, "vs. True-side target", true_side_target,
                  "- clipped to 0. Likely inconsistent input data across sources.")
            target = 0.0
        return target

    # Initialize weighting selectors and targets
    selectors = []
    targets = []
    
    # Population constraints
    combinations = list(itertools.product(unique_communes, unique_sexes, population_age_classes))
    for combination in context.progress(combinations, total = len(combinations), label = "Generating population constraints"):    
        f_reference = df_population["commune_index"] == combination[0]
        f_reference &= df_population["sex"] == combination[1]
        f_reference &= df_population["age_class"] == combination[2] 
    
        f_model = df_model["commune_index"] == combination[0]
        f_model &= df_model["sex"] == combination[1]
        f_model &= df_model["age_class_population"] == combination[2]
        selectors.append(f_model)
    
        target_weight = df_population.loc[f_reference, "weight"].sum()
        targets.append(target_weight)

    # Employment constraints   
    combinations = list(itertools.product(unique_departements, unique_sexes, employment_age_classes))
    for combination in context.progress(combinations, total = len(combinations), label = "Generating employment constraints"):
        f_reference = df_employment["departement_index"] == combination[0]
        f_reference &= df_employment["sex"] == combination[1]
        f_reference &= df_employment["age_class"] == combination[2] 
    
        f_model = df_model["departement_index"] == combination[0]
        f_model &= df_model["sex"] == combination[1]
        f_model &= df_model["age_class_employment"] == combination[2]
        f_model &= df_model["employed"] # Only select employed!
        selectors.append(f_model)

        target_weight = df_employment.loc[f_reference, "weight"].sum()
        targets.append(target_weight)

        # Ae-112 fix: bind the not-employed side of the same bucket too, see
        # comment above `not_side_target`.
        f_model_not_employed = df_model["departement_index"] == combination[0]
        f_model_not_employed &= df_model["sex"] == combination[1]
        f_model_not_employed &= df_model["age_class_employment"] == combination[2]
        f_model_not_employed &= ~df_model["employed"]
        selectors.append(f_model_not_employed)
        targets.append(not_side_target(population_totals_employment, combination, target_weight))

    # Minimum employment age
    f_model = df_model["combined_age_class"] < minimum_employment_age
    f_model &= df_model["employed"]
    selectors.append(f_model)
    targets.append(0.0)

    # License country constraints
    combinations = list(itertools.product(unique_sexes, license_age_classes))
    for combination in context.progress(combinations, total = len(combinations), label = "Generating license constraints"):
        f_reference = df_licenses_country["sex"] == combination[0]
        f_reference &= df_licenses_country["age_class"] == combination[1] 
    
        f_model = df_model["sex"] == combination[0]
        f_model &= df_model["age_class_license"] == combination[1]
        f_model &= df_model["license"] # Only select license owners!
        selectors.append(f_model)

        target_weight = df_licenses_country.loc[f_reference, "weight"].sum()
        targets.append(target_weight)

        # Ae-112 fix: bind the no-license side of the same bucket too.
        f_model_no_license = df_model["sex"] == combination[0]
        f_model_no_license &= df_model["age_class_license"] == combination[1]
        f_model_no_license &= ~df_model["license"]
        selectors.append(f_model_no_license)
        targets.append(not_side_target(population_totals_license_country, combination, target_weight))

    # License Kreis constraints
    for departement_index in context.progress(unique_departements, total = len(unique_departements), label = "Generating license constraints per Kreis"):
        f_reference = df_licenses_kreis["departement_index"] == departement_index
    
        f_model = df_model["departement_index"] == departement_index
        f_model &= df_model["license"] # Only select license owners!
        selectors.append(f_model)

        target_weight = df_licenses_kreis.loc[f_reference, "weight"].sum()
        targets.append(target_weight)

        # Ae-112 fix: bind the no-license side of the same Kreis too.
        f_model_no_license = df_model["departement_index"] == departement_index
        f_model_no_license &= ~df_model["license"]
        selectors.append(f_model_no_license)
        targets.append(not_side_target(population_totals_departement, departement_index, target_weight))

    # Transform to index-based
    selectors = [np.nonzero(s.values) for s in selectors]
    
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
    df_model = pd.merge(df_model, df_population[["commune_index", "commune_id"]].drop_duplicates(), on = "commune_index", how = "left")
    assert np.count_nonzero(df_model["commune_id"].isna()) == 0

    df_model = pd.merge(df_model, df_population[["departement_index", "departement_id"]].drop_duplicates(), on = "departement_index", how = "left")
    assert np.count_nonzero(df_model["departement_id"].isna()) == 0

    df_model = df_model.rename(columns = { "combined_age_class": "age_class" })
    return df_model[["commune_id", "departement_id", "sex", "age_class", "employed", "license", "weight"]]
