"""
full_hybrid_pipeline.py

End-to-end pipeline:
 - Build deterministic biologically-informed pretrained genus embeddings (64-d)
 - Build pairwise dataset (trait-driven + embeddings + phylo)
 - Train RandomForest pair success model (Regressor -> probability)
 - Forecast trait improvements and generate actionable breeder recommendations
"""

import os
import math
import sys
import random
from pathlib import Path
import warnings
warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error
import joblib
from Bio import Phylo
from scipy.spatial.distance import pdist, squareform
import json
import itertools
from typing import List, Tuple

# ---------------------------
# Config
# ---------------------------
SEED = 42
random.seed(SEED); np.random.seed(SEED)

GENUS_CSV = "genus_processed.csv"
TREE_FILE = "genus.tre"                  # optional
PRETRAINED_EMB_FILE = "genus_pretrained_emb.npy"
EMB_META_FILE = "genus_emb_meta.csv"
PAIR_PIPE_FILE = "pair_pipe.pkl"         # contains model, scaler, meta
PAIR_MODEL_FILE = "pair_model_rf.pkl"

EMB_DIM = 64
MAX_PAIRS = 800000   # sample limit 
NEG_POS_RATIO = 1     # negative:positive ratio to gather
HARD_NEG_THRESHOLD = 0.6

RF_PARAMS = dict(
    n_estimators=300,
    max_depth=20,
    min_samples_leaf=5,
    n_jobs=-1,
    random_state=SEED
)

USE_PHYLO = os.path.exists(TREE_FILE)

# Traits that represent breeder-interest
FOCUS_TRAITS = ["drought_tol", "salinity_tol", "earliness", "disease_res", "yield_pot"]

# ---------------------------
# Helpers
# ---------------------------
def load_genus_table(path=GENUS_CSV):
    if not Path(path).exists():
        raise FileNotFoundError(f"Missing {path}. Place genus_processed.csv in working dir.")
    df = pd.read_csv(path)
    # ensuring 'Genus' column exists
    if 'Genus' not in df.columns:
        raise ValueError("genus_processed.csv must contain 'Genus' column")
    df = df.set_index('Genus')
    return df

def build_pretrained_embeddings(genus_df: pd.DataFrame, emb_dim=EMB_DIM, out_file=PRETRAINED_EMB_FILE, meta_file=EMB_META_FILE):
    """
    Deterministic embedding builder (no external models):
    - numeric trait PCA (captures trait covariance)
    - family/order one-hot embedding (if present)
    - HybProp/Hyb_Ratio included
    - final PCA to emb_dim
    Saves embeddings to disk.
    """
    print("Building pretrained biological genus embeddings (deterministic)...")
    # numeric traits
    numeric_cols = genus_df.select_dtypes(include=[np.number]).columns.tolist()
    # excluding HybProp/Hyb_Ratio for trait space optionally, but keeping them as features too
    trait_cols = [c for c in numeric_cols if c not in []]
    # If dataset is small, ensure we have some numeric features
    if len(trait_cols) == 0:
        raise ValueError("No numeric trait columns found in genus_processed.csv")
    traits = genus_df[trait_cols].fillna(0.0).astype(float).values
    scaler = StandardScaler()
    traits_scaled = scaler.fit_transform(traits)

    # optional taxonomy categorical columns
    cat_cols = []
    for c in ['Family', 'Order']:
        if c in genus_df.columns:
            cat_cols.append(c)
    cat_emb = None
    if len(cat_cols) > 0:
        cat_df = genus_df[cat_cols].fillna("NA").astype(str)
        enc = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
        cat_emb = enc.fit_transform(cat_df)
    else:
        cat_emb = np.zeros((traits_scaled.shape[0], 0))

    # Concatenate features: traits_scaled + cat_emb + HybProp/Hyb_Ratio if present
    extra_cols = []
    for c in ['HybProp', 'Hyb_Ratio']:
        if c in genus_df.columns:
            extra_cols.append(c)
    extras = genus_df[extra_cols].fillna(0.0).astype(float).values if len(extra_cols) > 0 else np.zeros((traits_scaled.shape[0], 0))

    combined = np.concatenate([traits_scaled, cat_emb, extras], axis=1)

    # Primary PCA to reduce noise (to max 128)
    pca1_dim = min(max(emb_dim*2, 32), combined.shape[1])
    pca1 = PCA(n_components=min(pca1_dim, combined.shape[1]), random_state=SEED)
    pca1_out = pca1.fit_transform(combined)

    # Final PCA to emb_dim
    pca2 = PCA(n_components=min(emb_dim, pca1_out.shape[1]), random_state=SEED)
    emb = pca2.fit_transform(pca1_out)

    # normalize to unit vectors (helps cosine similarity)
    norms = np.linalg.norm(emb, axis=1, keepdims=True) + 1e-10
    emb = emb / norms

    # Save
    np.save(out_file, emb)
    # Save meta (index align)
    meta_df = genus_df.reset_index()[['Genus'] + cat_cols + extra_cols]
    meta_df.to_csv(meta_file, index=False)
    print(f"Saved pretrained embeddings to {out_file} ({emb.shape[0]} x {emb.shape[1]}) and meta to {meta_file}")
    return emb, meta_df

def load_pretrained_embeddings(path=PRETRAINED_EMB_FILE):
    if not Path(path).exists():
        raise FileNotFoundError("Pretrained embeddings not found. Run embedding builder first.")
    emb = np.load(path)
    return emb

def phylogenetic_distance_matrix(genus_df: pd.DataFrame, tree_file: str = TREE_FILE):
    if not Path(tree_file).exists():
        print("No phylogeny file found; will fall back to taxonomic-order distances.")
        return None
    try:
        tree = Phylo.read(tree_file, "newick")
        # ensure tips correspond to genus names; tree.distance uses tip names present in tree
        names = list(genus_df.index)
        # compute pairwise distances in tree - if name missing, fallback to default
        n = len(names)
        mat = np.zeros((n,n), dtype=float)
        for i in range(n):
            for j in range(i+1,n):
                a = names[i]; b = names[j]
                try:
                    d = tree.distance(a, b)
                except Exception:
                    d = np.nan
                mat[i,j] = mat[j,i] = d if not math.isnan(d) else np.nan
        # replace nan with large value
        maxd = np.nanmax(mat[np.isfinite(mat)]) if np.any(np.isfinite(mat)) else 1.0
        mat = np.where(np.isfinite(mat), mat, maxd)
        # convert to similarity in [0,1]
        sim = 1.0 / (1.0 + mat)
        return sim
    except Exception as e:
        print("Could not compute tree distances:", e)
        return None

def taxonomic_distance_matrix(genus_df: pd.DataFrame):
    # simple fallback: distance 0 if same genus (not used), 0.5 if same family, 0.8 if same order, 1.0 otherwise -> convert to similarity
    print("Building taxonomic distance matrix from Family/Order (fallback).")
    n = len(genus_df)
    names = list(genus_df.index)
    family = genus_df['Family'] if 'Family' in genus_df.columns else pd.Series(["NA"]*n, index=genus_df.index)
    order = genus_df['Order'] if 'Order' in genus_df.columns else pd.Series(["NA"]*n, index=genus_df.index)
    mat = np.ones((n,n), dtype=float)
    for i in range(n):
        for j in range(n):
            if i == j:
                mat[i,j] = 0.0
            else:
                if family.iloc[i] == family.iloc[j]:
                    mat[i,j] = 0.3
                elif order.iloc[i] == order.iloc[j]:
                    mat[i,j] = 0.6
                else:
                    mat[i,j] = 1.0
    sim = 1.0 / (1.0 + mat)  # convert to similarity
    return sim

# ---------------------------
# Build pair dataset (features)
# ---------------------------

def realistic_success_probability(hybprop1, hybprop2, trait_sim, emb_sim, phy_sim):
    """Biologically realistic continuous target"""
    # Base success from similarity
    similarity_score = 0.4 * trait_sim + 0.3 * emb_sim + 0.3 * phy_sim
    
    # Individual propensity boost (tanh caps extremes)
    propensity_boost = 0.15 * max(0, np.tanh(hybprop1)) + 0.15 * max(0, np.tanh(hybprop2))
    
    # Realistic noise
    success_prob = np.clip(similarity_score + propensity_boost + np.random.normal(0, 0.12), 0, 1)
    return float(success_prob)

def build_pair_dataset(genus_df: pd.DataFrame, emb: np.ndarray, phylo_sim_mat: np.ndarray=None, max_pairs=MAX_PAIRS):
    print("Constructing pair feature table...")
    names = list(genus_df.index)
    n = len(names)

    # Precompute trait arrays (numeric)
    trait_cols = genus_df.select_dtypes(include=[np.number]).columns.tolist()
    # Keep HybProp/Hyb_Ratio in trait list to use as individual propensity signals
    traits_arr = genus_df[trait_cols].fillna(0.0).values

    # Precompute cosine similarities for embeddings
    emb_cos = 1 - squareform(pdist(emb, metric='cosine'))

    # phylo
    if phylo_sim_mat is None:
        phylo_sim_mat = taxonomic_distance_matrix(genus_df)

    # produce ALL pairs indices - but sample if too many
    all_pairs = list(itertools.combinations(range(n), 2))
    total_pairs = len(all_pairs)
    print(f"Total combinatorial pairs: {total_pairs}")
    if max_pairs is not None and total_pairs > max_pairs:
        print(f"Sampling {max_pairs} pairs (random seed {SEED})")
        sampled_idx = np.random.choice(range(total_pairs), size=max_pairs, replace=False)
        pair_idx = [all_pairs[i] for i in sampled_idx]
    else:
        pair_idx = all_pairs

    # build records
    records = []
    for i,j in pair_idx:
        # features:
        emb_sim = float(emb_cos[i,j])
        phy_sim = float(phylo_sim_mat[i,j]) if phylo_sim_mat is not None else 0.5
        # trait similarity: cosine on numeric trait vectors (avoid zero vector)
        try:
            t1 = traits_arr[i].reshape(1,-1)
            t2 = traits_arr[j].reshape(1,-1)
            trait_sim = float( (t1 @ t2.T) / ( (np.linalg.norm(t1) * np.linalg.norm(t2)) + 1e-10 ) )
        except Exception:
            # fallback to 0.5
            trait_sim = 0.5

        # individual genus HybProp / Hyb_Ratio signals if present
        hybprop1 = genus_df.iloc[i].get('HybProp', 0.0)
        hybprop2 = genus_df.iloc[j].get('HybProp', 0.0)
        hybratio1 = genus_df.iloc[i].get('Hyb_Ratio', 0.0)
        hybratio2 = genus_df.iloc[j].get('Hyb_Ratio', 0.0)

        # target
        label = realistic_success_probability(hybprop1, hybprop2, trait_sim, emb_sim, phy_sim)

        records.append({
            'g1': names[i], 'g2': names[j],
            'emb_sim': emb_sim, 'phy_sim': phy_sim, 'trait_sim': trait_sim,
            'hybprop1': float(hybprop1), 'hybprop2': float(hybprop2),
            'hybratio1': float(hybratio1), 'hybratio2': float(hybratio2),
            'label': label
        })
    df_pairs = pd.DataFrame.from_records(records)
    print(f"Built pair table with {len(df_pairs)} rows")
    return df_pairs

# ---------------------------
# Trainpair model
# ---------------------------
def train_pair_model(df_pairs: pd.DataFrame, feature_cols: List[str], target_col='label', test_size=0.2):
    print("Training RandomForest pair model...")
    X = df_pairs[feature_cols].values
    y = df_pairs[target_col].values.astype(float)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=SEED)

    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)

    rf = RandomForestRegressor(**RF_PARAMS)
    rf.fit(X_train_s, y_train)

    y_pred = rf.predict(X_test_s)
    r2 = r2_score(y_test, y_pred)
    rmse = math.sqrt(mean_squared_error(y_test, y_pred))
    print(f"RF model: R2={r2:.4f}, RMSE={rmse:.4f}")
    return {'model': rf, 'scaler': scaler, 'r2': r2, 'rmse': rmse, 'X_test': X_test, 'y_test': y_test, 'y_pred': y_pred}

# ---------------------------
# Forecasting trait improvements
# ---------------------------
def forecast_trait_improvements_pair(pair_pipe, traits_a: dict, traits_b: dict, focus_traits: List[str]=FOCUS_TRAITS,
                                     relative_delta=0.15, who_options=('A','B','both'), top_k=8):
    """
    pair_pipe = {'model':rf,'scaler':scaler,'feature_cols':feature_cols,'genus_df':genus_df,'emb':emb,'meta':meta}
    traits_* are dicts mapping trait name->value (must align with genus_df numeric columns)
    Returns top_k suggested trait improvements with predicted delta.
    """
    model = pair_pipe['model']; scaler = pair_pipe['scaler']
    feature_cols = pair_pipe['feature_cols']; genus_df = pair_pipe['genus_df']; emb = pair_pipe['emb']; meta = pair_pipe['meta']

    # Build trait vectors following trait numeric columns order from genus_df
    numeric_cols = genus_df.select_dtypes(include=[np.number]).columns.tolist()
    # Build arrays for current A/B traits
    def dict_to_vec(td):
        return np.array([float(td.get(c, genus_df[c].mean() if c in genus_df.columns else 0.0)) for c in numeric_cols], dtype=float)

    vecA = dict_to_vec(traits_a); vecB = dict_to_vec(traits_b)

    # compute baseline pair features (emb sim / phy sim / trait sim / hybprop signals)
    # if genus names unknown, hybprop signals zero
    emb_ar = emb
    # compute trait similarity using cosine directly on numeric traits; embedding-based suggestions are approximate.
    def cos_sim(x,y):
        num = float(np.dot(x,y))
        den = (np.linalg.norm(x)*np.linalg.norm(y)+1e-10)
        return num/den

    base_trait_sim = cos_sim(vecA, vecB)
    # phy sim fallback = 0.5
    base_phy = 0.5
    base_emb_sim = base_trait_sim  # approximate if no direct emb for new trait vector
    # construct model input
    baseline_X = np.array([[base_emb_sim, base_phy, base_trait_sim, 0.0, 0.0, 0.0, 0.0]])  # placeholder ordering; will be re-ordered below

    # We'll create function that computes feature vector given trait vectors and optional hybprop signals
    def features_from_traitvecs(vecA_local, vecB_local, hybprop1=0.0, hybprop2=0.0):
        emb_sim = cos_sim(vecA_local, vecB_local)  # use trait-based sim as proxy for new embeddings
        trait_sim = emb_sim
        phy_sim = 0.5
        return np.array([emb_sim, phy_sim, trait_sim, float(hybprop1), float(hybprop2), 0.0, 0.0])

    # determine feature ordering
    feature_cols = pair_pipe['feature_cols']

    # Evaluate baseline
    base_feat = features_from_traitvecs(vecA, vecB, 0.0, 0.0)
    Xb = base_feat.reshape(1,-1)
    Xb_s = scaler.transform(Xb)
    base_pred = model.predict(Xb_s)[0]

    results = []
    # iterate over focus traits (if present in numeric_cols)
    numeric_names = numeric_cols
    for t in focus_traits:
        if t not in numeric_names:
            continue
        idx = numeric_names.index(t)
        # compute delta relative to range
        tmin = float(genus_df[t].min()) if t in genus_df.columns else 0.0
        tmax = float(genus_df[t].max()) if t in genus_df.columns else 1.0
        trange = (tmax - tmin) if (tmax - tmin) > 0 else 1.0
        delta = relative_delta * trange
        # A improved
        a_mod = vecA.copy(); a_mod[idx] = min(vecA[idx] + delta, tmax)
        feat_a = features_from_traitvecs(a_mod, vecB)
        pred_a = model.predict(scaler.transform(feat_a.reshape(1,-1)))[0]
        # B improved
        b_mod = vecB.copy(); b_mod[idx] = min(vecB[idx] + delta, tmax)
        feat_b = features_from_traitvecs(vecA, b_mod)
        pred_b = model.predict(scaler.transform(feat_b.reshape(1,-1)))[0]
        # both improved
        feat_ab = features_from_traitvecs(a_mod, b_mod)
        pred_ab = model.predict(scaler.transform(feat_ab.reshape(1,-1)))[0]

        for who, s in [('A', pred_a), ('B', pred_b), ('both', pred_ab)]:
            delta_abs = s - base_pred
            pct = (delta_abs / base_pred * 100.0) if abs(base_pred) > 1e-8 else delta_abs*100.0
            results.append({
                'trait': t,
                'who': who,
                'orig_pred': base_pred,
                'new_pred': s,
                'delta_abs': delta_abs,
                'pct_delta': pct,
                'delta_amount': delta
            })
    results_sorted = sorted(results, key=lambda x: abs(x['delta_abs']), reverse=True)
    return results_sorted[:10], base_pred

# ---------------------------
# Actionable text recommendation
# ---------------------------
def actionable_recommendations(forecast_results, threshold_abs=0.005):
    recs = []
    for r in forecast_results:
        if r['delta_abs'] <= threshold_abs:
            continue
        trait = r['trait']; who = r['who']; d = r['delta_abs']; pct = r['pct_delta']; amt = r['delta_amount']
        recs.append(f"Improve {trait} in {who} by about {amt:.3f} units → estimated hybrid score Δ {d*100:.2f} pts ({pct:.2f}%).")
    if len(recs) == 0:
        recs = ["No trait improvement found exceeding threshold."]
    return recs

# ---------------------------
# MAIN runnable flow
# ---------------------------
def main():
    print("=== Full Hybrid Pipeline ===")
    genus_df = load_genus_table(GENUS_CSV)

    # 1) build or load pretrained biological embeddings
    if Path(PRETRAINED_EMB_FILE).exists():
        print("Loading existing pretrained embeddings...")
        emb = load_pretrained_embeddings(PRETRAINED_EMB_FILE)
        meta = pd.read_csv(EMB_META_FILE) if Path(EMB_META_FILE).exists() else genus_df.reset_index()[['Genus']]
    else:
        emb, meta = build_pretrained_embeddings(genus_df, emb_dim=EMB_DIM)

    # 2) phylogenetic similarity matrix
    phy_sim_mat = None
    if USE_PHYLO:
        try:
            phy_sim_mat = phylogenetic_distance_matrix(genus_df, TREE_FILE)
            print("Phylogenetic similarity matrix computed.")
        except Exception as e:
            print("Phylo computation failed:", e)
            phy_sim_mat = taxonomic_distance_matrix(genus_df)
    else:
        phy_sim_mat = taxonomic_distance_matrix(genus_df)

    # 3) Build pair dataset (sampled)
    df_pairs = build_pair_dataset(genus_df, emb, phy_sim_mat, max_pairs=MAX_PAIRS)
    # Show class balance
    print("Label distribution:", df_pairs['label'].value_counts().to_dict())

    # 4) create positives & negatives (hard negative sampling)
    positives = df_pairs[df_pairs['label'] > 0].copy()
    negatives = df_pairs[df_pairs['label'] == 0].copy()
    hard_negatives = negatives[(negatives['trait_sim'] >= HARD_NEG_THRESHOLD) | (negatives['emb_sim'] >= HARD_NEG_THRESHOLD)]
    # sample negatives to match ratio
    npos = len(positives)
    nneg_needed = min(len(negatives), int(npos * NEG_POS_RATIO))
    # combine hard negs + random negs
    selected_negs = pd.concat([hard_negatives, negatives.sample(max(0, nneg_needed - len(hard_negatives)), random_state=SEED)]).drop_duplicates().reset_index(drop=True)
    train_df = pd.concat([positives, selected_negs]).sample(frac=1.0, random_state=SEED).reset_index(drop=True)
    print(f"Training pairs: positives={len(positives)}, negatives_selected={len(selected_negs)}, total_train={len(train_df)}")

    # 5) Train RandomForest on pair features
    feature_cols = ['emb_sim','phy_sim','trait_sim','hybprop1','hybprop2','hybratio1','hybratio2']
    pair_result = train_pair_model(train_df, feature_cols, target_col='label', test_size=0.15)

    rf = pair_result['model']; scaler = pair_result['scaler']
    # Save model + scaler + meta
    pair_pipe = {'model': rf, 'scaler': scaler, 'feature_cols': feature_cols, 'genus_df': genus_df, 'emb': emb, 'meta': meta}
    joblib.dump(pair_pipe, PAIR_PIPE_FILE)
    joblib.dump(rf, PAIR_MODEL_FILE)
    print(f"Saved pair_pipe to {PAIR_PIPE_FILE} and raw model to {PAIR_MODEL_FILE}")

    # Forecast improvements
    forecast_results, base_pred = forecast_trait_improvements_pair(pair_pipe, traitsA, traitsB, focus_traits=[t for t in FOCUS_TRAITS if t in numeric_cols], relative_delta=0.15, top_k=10)
    print(f"Base predicted probability: {base_pred:.4f}")
    print("Top forecasted trait improvements:")
    for r in forecast_results[:8]:
        print(f" - {r['trait']} ({r['who']}): Δ {r['delta_abs']*100:.3f} pts → new {r['new_pred']*100:.2f}%")
    recs = actionable_recommendations(forecast_results)
    print("\nActionable recommendations:")
    for s in recs:
        print(" *", s)

if __name__ == "__main__":
    main()
