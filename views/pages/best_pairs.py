#!/usr/bin/env python3
"""
best_pairs.py - Display top hybridization pairs
"""

import streamlit as st
import sys
import os
from pathlib import Path
import pandas as pd
import numpy as np
import warnings
import plotly.express as px
warnings.filterwarnings('ignore')

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent / 'Hyberidization_Propensity_Predictor'))

# Page config
st.set_page_config(
    page_title="Best Hybrid Pairs - Agro X",
    page_icon="🌿",
    layout="wide"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        margin-bottom: 1rem;
    }
    .st-emotion-cache-scp8yw {
        display: none !important;
    }
    .sub-header {
        font-size: 1.2rem;
        margin-bottom: 2rem;
    }
    .metric-card {
        background: linear-gradient(135deg, #E8F5E9 0%, #C8E6C9 100%);
        padding: 1.5rem;
        border-radius: 10px;
        border-left: 5px solid #4CAF50;
        margin: 1rem 0;
    }
    .stDataFrame {
        border-radius: 10px;
        overflow: hidden;
    }
</style>
""", unsafe_allow_html=True)

# Header
st.markdown('<div class="main-header">🌿 Best Hybridization Pairs</div>', unsafe_allow_html=True)

# Load model and data
@st.cache_resource
def load_model():
    """Load the pre-trained model and data"""
    try:
        import joblib
        bundle_path = Path(__file__).parent.parent.parent / 'Hyperidization_Propensity_Predictor' / 'pair_pipe.pkl'
        bundle = joblib.load(str(bundle_path))
        model = bundle['model']
        scaler = bundle['scaler']
        feature_cols = bundle['feature_cols']
        genus_df = bundle['genus_df']
        emb = bundle['emb']
        meta = bundle['meta']
        return model, scaler, feature_cols, genus_df, emb, meta
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None, None, None, None, None, None

model, scaler, feature_cols, genus_df, emb, meta = load_model()

if model is None:
    st.error("❌ Failed to load the prediction model. Please ensure pair_pipe.pkl exists in HybrPairSucc_Predictor directory.")
    st.stop()

# Helper functions
def get_genus_info(genus_name):
    """Returns (traits_dict, embedding_array)"""
    traits = {}
    if 'Genus' in genus_df.columns:
        row = genus_df[genus_df['Genus'].str.strip() == genus_name.strip()]
        if not row.empty: 
            traits = row.iloc[0].to_dict()
    elif genus_name in genus_df.index:
        traits = genus_df.loc[genus_name].to_dict()
    
    # Embedding  
    emb_idx = 0
    if 'Genus' in meta.columns:
        meta_row = meta[meta['Genus'].str.strip() == genus_name.strip()]
        if not meta_row.empty: 
            emb_idx = int(meta_row.index[0])
    
    return traits, emb[emb_idx]

def predict_hybrid_success(traits_a, traits_b, emb_a, emb_b):
    """Predicts hybridization success probability"""
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

def find_top_pairs(n=10, progress_callback=None):
    """Find top n hybridization pairs from all genera"""
    genera = list(genus_df.index)[:30]  # Fast demo - use first 30 genera like demo_predict.py
    results = []
    total = len(genera)
    
    for i in range(total):
        if progress_callback:
            progress_callback(i / total)
        traits1, emb1 = get_genus_info(genera[i])
        for j in range(i+1, min(i+10, len(genera))):  # Limit comparisons
            traits2, emb2 = get_genus_info(genera[j])
            prob = predict_hybrid_success(traits1, traits2, emb1, emb2)
            if prob > 0.3:  # Only good pairs
                results.append({
                    'Genus 1': genera[i], 
                    'Genus 2': genera[j], 
                    'Success Probability': prob,
                    'Percentage': f"{prob*100:.1f}%"
                })
    
    df = pd.DataFrame(results).nlargest(n, 'Success Probability')
    return df

def find_best_partners(target_genus, n=10, progress_callback=None):
    """Find top n best hybridization partners for a specific genus"""
    genera = list(genus_df.index)
    
    if target_genus not in genera:
        return None
    
    traits_target, emb_target = get_genus_info(target_genus)
    results = []
    
    # Sample genera for faster computation (200 candidates)
    target_idx = genera.index(target_genus)
    sample_genera = []
    
    # Include neighbors and random samples
    start_idx = max(0, target_idx - 50)
    end_idx = min(len(genera), target_idx + 50)
    neighbors = genera[start_idx:end_idx]
    
    # Add random samples from rest
    remaining = [g for g in genera if g not in neighbors and g != target_genus]
    if len(remaining) > 100:
        import random
        random.seed(42)
        remaining = random.sample(remaining, 100)
    
    sample_genera = [g for g in neighbors + remaining if g != target_genus][:200]
    total = len(sample_genera)
    
    for i, genus in enumerate(sample_genera):
        if progress_callback:
            progress_callback(i / total)
        traits, emb_g = get_genus_info(genus)
        prob = predict_hybrid_success(traits_target, traits, emb_target, emb_g)
        results.append({
            'Partner Genus': genus,
            'Success Probability': prob,
            'Percentage': f"{prob*100:.1f}%"
        })
    
    df = pd.DataFrame(results).nlargest(n, 'Success Probability')
    return df

# Main content - Controls
st.markdown("### ⚙️ Configuration")

col1, col2 = st.columns(2)

with col1:
    mode = st.radio(
        "Select Mode:",
        ["Top Pairs Overall", "Best Partners for Specific Genus"],
        help="Choose to see overall top pairs or find best partners for a specific genus"
    )

with col2:
    n_results = st.slider(
        "Number of Results:",
        min_value=5,
        max_value=30,
        value=10,
        step=5,
        help="How many results to display"
    )

st.markdown("---")

# Results section
if mode == "Top Pairs Overall":
    st.markdown("### 🏆 Top Hybridization Pairs Across All Genera")
    
    if st.button("🔍 Find Top Pairs", type="primary", use_container_width=True):
        with st.spinner("Analyzing genus combinations..."):
            progress_bar = st.progress(0)
            results = find_top_pairs(n=n_results, progress_callback=lambda p: progress_bar.progress(p))
            progress_bar.empty()
            
            if not results.empty:
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Best Probability", f"{results['Success Probability'].max():.1%}")
                with col2:
                    st.metric("Average Probability", f"{results['Success Probability'].mean():.1%}")
                with col3:
                    st.metric("Pairs Analyzed", len(results))
                
                st.markdown("#### 📊 Results")
                st.dataframe(
                    results[['Genus 1', 'Genus 2', 'Percentage']].reset_index(drop=True),
                    use_container_width=True,
                    hide_index=True
                )
                
                csv = results.to_csv(index=False)
                st.download_button(
                    label="📥 Download Results as CSV",
                    data=csv,
                    file_name=f"top_{n_results}_hybrid_pairs.csv",
                    mime="text/csv",
                    use_container_width=True
                )
            else:
                st.warning("No pairs found meeting the criteria.")

else:  # Best Partners for Specific Genus
    st.markdown("### 🎯 Find Best Hybridization Partners")
    
    # Get list of all genera
    all_genera = sorted(list(genus_df.index))
    
    # Genus selection
    col1, col2 = st.columns([3, 1])
    with col1:
        selected_genus = st.selectbox(
            "Select Target Genus:",
            options=all_genera,
            help="Choose the genus you want to find partners for"
        )
    
    with col2:
        st.markdown("<br>", unsafe_allow_html=True)
        search_button = st.button("🔍 Find Partners", type="primary", use_container_width=True)
    
    if search_button:
        if selected_genus:
            with st.spinner(f"Finding best partners for {selected_genus}..."):
                progress_bar = st.progress(0)
                results = find_best_partners(selected_genus, n=n_results, progress_callback=lambda p: progress_bar.progress(p))
                progress_bar.empty()
                
                if results is not None and not results.empty:
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("Target Genus", selected_genus)
                    with col2:
                        st.metric("Best Match Probability", f"{results['Success Probability'].max():.1%}")
                    with col3:
                        st.metric("Partners Found", len(results))
                    
                    st.markdown("#### 📊 Best Hybridization Partners")
                    
                    # Add color styling based on probability
                    def color_probability(val):
                        if isinstance(val, str):
                            prob = float(val.strip('%')) / 100
                        else:
                            prob = val
                        if prob >= 0.7:
                            return 'background-color: #1B5E20; color: #A5D6A7'  # Dark green bg, light green text
                        elif prob >= 0.5:
                            return 'background-color: #F57F17; color: #FFF59D'  # Dark amber bg, light yellow text
                        else:
                            return 'background-color: #BF360C; color: #FFAB91'  # Dark red bg, light orange text
                    
                    styled_df = results[['Partner Genus', 'Percentage']].style.applymap(
                        color_probability, 
                        subset=['Percentage']
                    )
                    
                    st.dataframe(
                        styled_df,
                        hide_index=True,
                        use_container_width=True,
                    )
                    
                    # Probability distribution
                    st.markdown("#### 📈 Probability Distribution")
                    fig = px.bar(
                        results.head(n_results),
                        x='Partner Genus',
                        y='Success Probability',
                        color='Success Probability',
                        color_continuous_scale='Greens',
                        labels={'Success Probability': 'Success Probability'},
                        title=f'Top {n_results} Partners for {selected_genus}'
                    )
                    fig.update_layout(showlegend=False)
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Download button
                    csv = results.to_csv(index=False)
                    st.download_button(
                        label=f"📥 Download Partners for {selected_genus}",
                        data=csv,
                        file_name=f"best_partners_{selected_genus}.csv",
                        mime="text/csv",
                        use_container_width=True
                    )
                else:
                    st.error("Could not find partners for the selected genus.")
        else:
            st.warning("Please select a genus first.")

