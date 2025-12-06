"""
Advanced preprocessing of genus_data.csv for HybProp modeling.

- Loads genus_data.csv and replaces "." with NaN.
- Incorporates phylogenetic information from family_data.csv, family.tre, and genus.tre
- Uses family-level means and phylogenetic relationships to inform imputation
- Casts all relevant columns to numeric where possible.
- Drops rows with missing HybProp (target).
- Imputes numeric features with IterativeImputer (ExtraTreesRegressor) with phylo features.
- Imputes Family and Order with most-frequent strategy.
- Writes genus_processed_advanced.csv with no missing values.
"""

import numpy as np
import pandas as pd
from pathlib import Path
from Bio import Phylo
from io import StringIO

from sklearn.experimental import enable_iterative_imputer 
from sklearn.impute import IterativeImputer, SimpleImputer
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.preprocessing import StandardScaler

IN_PATH = Path("genus_data.csv")
FAMILY_PATH = Path("family_data.csv")
FAMILY_TREE_PATH = Path("family.tre")
GENUS_TREE_PATH = Path("genus.tre")
OUT_PATH = Path("genus_processed_advanced.csv")


def compute_phylogenetic_distances(tree_path, taxa_list):
    """
    Compute pairwise phylogenetic distances from a Newick tree.
    Returns a dictionary mapping (taxon1, taxon2) -> distance.
    """
    try:
        tree = Phylo.read(tree_path, "newick")
        distances = {}
        
        for taxon1 in taxa_list:
            for taxon2 in taxa_list:
                try:
                    dist = tree.distance(taxon1, taxon2)
                    distances[(taxon1, taxon2)] = dist
                except:
                    # If taxa not found in tree, use max distance
                    distances[(taxon1, taxon2)] = float('inf')
        
        return distances
    except:
        print(f"Warning: Could not load tree from {tree_path}")
        return {}


def phylogenetic_imputation(df, trait_col, genus_tree_path):
    """
    Impute missing values for a trait using phylogenetic distances.
    For each genus with missing value, impute using weighted average of 
    phylogenetically close genera (inverse distance weighting).
    """
    # Get genera list
    all_genera = df['Genus'].unique().tolist()
    
    # Compute phylogenetic distances
    print(f"  Computing phylogenetic distances for {len(all_genera)} genera...")
    distances = compute_phylogenetic_distances(genus_tree_path, all_genera)
    
    if not distances:
        print(f"  Warning: No phylogenetic distances available, skipping phylo imputation for {trait_col}")
        return df
    
    # Find rows with missing values for this trait
    missing_mask = df[trait_col].isna()
    missing_genera = df.loc[missing_mask, 'Genus'].unique()
    
    print(f"  Imputing {missing_mask.sum()} missing values for {trait_col} using phylogenetic distances...")
    
    for genus in missing_genera:
        # Get phylogenetic distances from this genus to all others
        genus_distances = []
        genus_values = []
        
        for other_genus in all_genera:
            if genus == other_genus:
                continue
            
            # Get trait values for other genus
            other_values = df[(df['Genus'] == other_genus) & (~df[trait_col].isna())][trait_col]
            
            if len(other_values) > 0:
                dist = distances.get((genus, other_genus), float('inf'))
                if dist != float('inf') and dist > 0:
                    genus_distances.append(dist)
                    genus_values.append(other_values.mean())
        
        # Calculate weighted average (inverse distance weighting)
        if genus_distances:
            weights = 1.0 / np.array(genus_distances)
            weights = weights / weights.sum()
            imputed_value = np.sum(np.array(genus_values) * weights)
            
            # Fill missing values for this genus
            df.loc[(df['Genus'] == genus) & missing_mask, trait_col] = imputed_value
    
    return df


def add_phylogenetic_features(df, family_df, family_tree_path, genus_tree_path):
    """
    Add phylogenetic features to help with imputation:
    - Family-level aggregated features
    - Phylogenetic distance-based features
    """
    # Add family-level means for numeric columns
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    numeric_cols = [c for c in numeric_cols if c not in ['HybProp']]
    
    # Merge family-level data
    family_agg = {}
    for col in numeric_cols:
        if col in family_df.columns:
            family_means = family_df.groupby('Family')[col].mean().to_dict()
            df[f'Family_mean_{col}'] = df['Family'].map(family_means)
    
    return df


def main():
    # 1. Load raw data
    df = pd.read_csv(IN_PATH)
    print(f"Loaded {IN_PATH} with shape {df.shape}")
    
    # Load family data
    family_df = pd.read_csv(FAMILY_PATH)
    family_df = family_df.replace(".", np.nan)
    print(f"Loaded {FAMILY_PATH} with shape {family_df.shape}")

    # 2. Replace '.' with NaN
    df = df.replace(".", np.nan)

    # Genus is an identifier, Family and Order are categorical
    id_like_cols = ["Genus", "Family", "Order"]
    cat_cols = ["Family", "Order"]

    # Cast all non-id-like columns to numeric
    for col in df.columns:
        if col not in id_like_cols:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    
    for col in family_df.columns:
        if col not in ["Family", "Order"]:
            family_df[col] = pd.to_numeric(family_df[col], errors="coerce")

    # 3. Drop rows with missing target HybProp
    before = df.shape[0]
    df = df.dropna(subset=["HybProp"])
    after = df.shape[0]
    print(f"Dropped {before - after} rows with missing HybProp; remaining {after}")

    # 4. Impute Family/Order with most-frequent (if needed)
    cat_imp = SimpleImputer(strategy="most_frequent")
    df[cat_cols] = cat_imp.fit_transform(df[cat_cols])

    # 5. Phylogenetic distance-based imputation for trait columns
    print("\n=== Phylogenetic Distance-Based Imputation ===")
    trait_cols = [c for c in df.columns if c not in id_like_cols and c != 'HybProp']
    for trait_col in trait_cols:
        if df[trait_col].isna().sum() > 0:
            print(f"\nImputing {trait_col}...")
            df = phylogenetic_imputation(df, trait_col, GENUS_TREE_PATH)

    # 6. Add phylogenetic and family-level features for any remaining missing values
    print("\n=== Adding Family-Level Features for Remaining Imputation ===")
    df = add_phylogenetic_features(df, family_df, FAMILY_TREE_PATH, GENUS_TREE_PATH)

    # 7. Iterative imputation for any remaining missing values
    #    Exclude Genus, Family, Order from numeric imputation
    num_cols = [c for c in df.columns if c not in id_like_cols]
    
    remaining_missing = df[num_cols].isna().sum().sum()
    if remaining_missing > 0:
        print(f"\n=== Running IterativeImputer for {remaining_missing} remaining missing values ===")
        
        base_estimator = ExtraTreesRegressor(
            n_estimators=100,
            random_state=42,
            n_jobs=-1,
        )

        iter_imp = IterativeImputer(
            estimator=base_estimator,
            initial_strategy="median",
            max_iter=50,
            random_state=42,
        )

        imputed_values = iter_imp.fit_transform(df[num_cols])
        assert imputed_values.shape == df[num_cols].shape
        df[num_cols] = imputed_values
    else:
        print("\n=== No remaining missing values, skipping IterativeImputer ===")

    # 8. Drop temporary phylogenetic feature columns (keep only original features)
    phylo_cols = [c for c in df.columns if c.startswith('Family_mean_')]
    if phylo_cols:
        print(f"\nDropping {len(phylo_cols)} temporary phylogenetic feature columns")
        df = df.drop(columns=phylo_cols)

    # 9. Standardize numeric features (excluding target HybProp)
    feature_cols = [c for c in df.columns if c not in id_like_cols and c != 'HybProp']
    if feature_cols:
        print(f"\n=== Standardizing {len(feature_cols)} numeric feature columns ===")
        scaler = StandardScaler()
        df[feature_cols] = scaler.fit_transform(df[feature_cols])

    # 10. Sanity check: no missing values
    missing_total = df.isna().sum().sum()
    print(f"\nTotal remaining missing values: {missing_total}")

    # 11. Save processed data
    df.to_csv(OUT_PATH, index=False)
    print(f"Saved fully processed table to: {OUT_PATH}")


if __name__ == "__main__":
    main()
