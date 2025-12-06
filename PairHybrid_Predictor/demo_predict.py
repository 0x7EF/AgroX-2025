import joblib
import pandas as pd
import numpy as np
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

print("🚀 Agro X 2025 - Hybridization Success Predictor")
print("=" * 60)

# LOAD PIPELINE ( It must exist )
bundle = joblib.load("pair_pipe.pkl")
model, scaler, feature_cols, genus_df, emb, meta = [bundle[k] for k in ['model','scaler','feature_cols','genus_df','emb','meta']]

print(f"Loaded: {len(genus_df)} genera, {len(feature_cols)} features")

def get_genus_info(genus_name):
    """Returns (traits_dict, embedding_array)"""
    # Traits
    traits = {}
    if 'Genus' in genus_df.columns:
        row = genus_df[genus_df['Genus'].str.strip() == genus_name.strip()]
        if not row.empty: traits = row.iloc[0].to_dict()
    elif genus_name in genus_df.index:
        traits = genus_df.loc[genus_name].to_dict()
    
    # Embedding  
    emb_idx = 0
    if 'Genus' in meta.columns:
        meta_row = meta[meta['Genus'].str.strip() == genus_name.strip()]
        if not meta_row.empty: emb_idx = int(meta_row.index[0])
    
    return traits, emb[emb_idx]

def predict_hybrid_success(traits_a, traits_b, emb_a, emb_b):
    """Predicts hybridization success"""
    emb_sim = np.dot(emb_a, emb_b) / (np.linalg.norm(emb_a) * np.linalg.norm(emb_b) + 1e-10)
    
    numeric_cols = ['perc_per', 'perc_wood', 'perc_ag', 'floral_symm', 'mating_system', 
                   'repro_syndrome', 'pollination_syndrome', 'RedList', 'tavg', 'C_value', 'CV_C_value']
    vec_a = np.array([traits_a.get(c, 0.0) for c in numeric_cols])
    vec_b = np.array([traits_b.get(c, 0.0) for c in numeric_cols])
    trait_sim = np.dot(vec_a, vec_b) / (np.linalg.norm(vec_a) * np.linalg.norm(vec_b) + 1e-10)
    
    features = np.array([
        emb_sim, 0.5, trait_sim,
        traits_a.get('HybProp', -0.18), traits_b.get('HybProp', -0.18),
        traits_a.get('Hyb_Ratio', 0.0), traits_b.get('Hyb_Ratio', 0.0)
    ], dtype=np.float32).reshape(1, -1)
    
    return float(np.clip(model.predict(scaler.transform(features))[0], 0, 1))

# TRAIT IMPROVEMENTS
BREEDER_TRAITS = {
    'perc_per': 'Perenniality', 'perc_wood': 'Woodiness', 
    'perc_ag': 'Agamospermy', 'HybProp': 'Hybrid Propensity', 'Hyb_Ratio': 'Hybrid Ratio'
}

def forecast_improvements(genus_a, genus_b):
    traits_a, emb_a = get_genus_info(genus_a)
    traits_b, emb_b = get_genus_info(genus_b)
    baseline = predict_hybrid_success(traits_a, traits_b, emb_a, emb_b)
    
    results = []
    for trait in BREEDER_TRAITS:
        if trait not in traits_a: continue
        
        # Parent A improvement
        traits_a_test = traits_a.copy()
        traits_a_test[trait] = min(1.0, traits_a_test[trait] * 1.3)
        prob_a = predict_hybrid_success(traits_a_test, traits_b, emb_a, emb_b)
        
        # Parent B improvement  
        traits_b_test = traits_b.copy()
        traits_b_test[trait] = min(1.0, traits_b_test[trait] * 1.3)
        prob_b = predict_hybrid_success(traits_a, traits_b_test, emb_a, emb_b)
        
        results.append({
            'Trait': BREEDER_TRAITS[trait],
            'Parent_A_Gain': f"+{(prob_a-baseline)*100:.1f}%",
            'Parent_B_Gain': f"+{(prob_b-baseline)*100:.1f}%"
        })
    
    return pd.DataFrame(results)

def find_top_pairs(n=10):
    genera = list(genus_df.index)[:30]  # Fast demo
    results = []
    
    print("🔍 Finding top pairs...", end=' ')
    for i in tqdm(range(len(genera)), desc="Scanning", ncols=80, leave=False):
        traits1, emb1 = get_genus_info(genera[i])
        for j in range(i+1, min(i+10, len(genera))):  # Limit comparisons
            traits2, emb2 = get_genus_info(genera[j])
            prob = predict_hybrid_success(traits1, traits2, emb1, emb2)
            if prob > 0.3:  # Only good pairs
                results.append({'Genus1': genera[i], 'Genus2': genera[j], 'Probability': prob})
    
    df = pd.DataFrame(results).nlargest(n, 'Probability')
    print("Done!")
    return df

# ===========================================
# DEMO
# ===========================================
if __name__ == "__main__":
    print("\n🎯 AGRO X 2025 - LIVE DEMO")
    print("-" * 45)
    
    # 1. BASELINE
    print("\n1️⃣ BASELINE")
    traits_a, emb_a = get_genus_info('Abies')
    traits_b, emb_b = get_genus_info('Abronia')
    base_prob = predict_hybrid_success(traits_a, traits_b, emb_a, emb_b)
    print(f"   Abies × Abronia: {base_prob:.3f}")
    
    # 2. OPTIMIZED
    print("\n2️⃣ OPTIMIZED BREEDING LINES")
    traits_a_opt = traits_a.copy()
    traits_b_opt = traits_b.copy()
    traits_a_opt['HybProp'] = 1.5
    traits_a_opt['perc_per'] = 1.0
    traits_b_opt['HybProp'] = 1.2  
    traits_b_opt['perc_per'] = 1.0
    opt_prob = predict_hybrid_success(traits_a_opt, traits_b_opt, emb_a, emb_b)
    print(f"   Optimized: {opt_prob:.3f}")
    print(f"   🎯 Gain: +{(opt_prob-base_prob):+.3f}")
    
    # 3. TRAIT IMPROVEMENTS
    print("\n3️⃣ BREEDER RECOMMENDATIONS")
    recs = forecast_improvements('Abies', 'Abronia')
    print(recs.round(3).to_string(index=False))
    
    # 4. TOP PAIRS
    print("\n4️⃣ TOP RECOMMENDED PAIRS")
    top_pairs = find_top_pairs(10)
    print(top_pairs[['Genus1', 'Genus2', 'Probability']].round(3).to_string(index=False))
    
    # SAVE
    top_pairs.to_csv('top_hybrid_pairs.csv', index=False)
    recs.to_csv('breeder_recommendations.csv', index=False)
    
    print("\nSAVED FILES:")
    print("top_hybrid_pairs.csv")
    print("breeder_recommendations.csv")
