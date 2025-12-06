import streamlit as st
import pandas as pd
import os
import numpy as np
import joblib
import warnings

warnings.filterwarnings('ignore')

st.set_page_config(
    page_title="Hybridization Test",
    page_icon="🌿",
    layout="wide"
)

st.markdown("""
<style>
    /* Selectbox styling */
    div[data-baseweb="select"] > div {
        border-color: #d3d3d3 !important;
        border-radius: 8px;
    }
    
    .st-emotion-cache-scp8yw {
        display: none !important;
    }
    
    div[data-baseweb="select"] > div:focus-within {
        border-color: #4CAF50 !important;
        box-shadow: 0 0 0 2px rgba(76, 175, 80, 0.2) !important;
    }
    
    div[data-baseweb="select"] > div:hover {
        border-color: #4CAF50 !important;
    }
    
    /* Button styling */
    .stButton > button {
        width: 100%;
        border-radius: 8px;
        height: 3em;
        font-weight: 600;
    }
    
    /* Card-like containers */
    .block-container {
        padding-top: 2rem;
    }
    
    /* Input styling */
    .stNumberInput > div > div > input {
        border-radius: 8px;
    }
    
    /* Section headers */
    .trait-section {
        background-color: #f8f9fa;
        padding: 1rem;
        border-radius: 8px;
        margin: 1rem 0;
        border-left: 4px solid #4CAF50;
    }
</style>
""", unsafe_allow_html=True)


@st.cache_data
def load_data():
    """Load and cache the genus data"""
    current_dir = os.path.dirname(os.path.abspath(__file__))
    data_path = os.path.join(current_dir, '..', '..', 'PairHybrid_Predictor', 'genus_processed.csv')
    return pd.read_csv(data_path)


@st.cache_resource
def load_model():
    """Load the trained hybridization prediction model and embeddings"""
    try:
        current_dir = os.path.dirname(os.path.abspath(__file__))
        model_path = os.path.join(current_dir, '..', '..', 'PairHybrid_Predictor', 'pair_pipe.pkl')
        
        if not os.path.exists(model_path):
            st.warning("⚠️ Model file not found. Prediction features will be disabled.")
            return None
        
        bundle = joblib.load(model_path)
        model = bundle['model']
        scaler = bundle['scaler']
        feature_cols = bundle['feature_cols']
        genus_df = bundle['genus_df']
        emb = bundle['emb']
        meta = bundle['meta']
        
        return {
            'model': model,
            'scaler': scaler,
            'feature_cols': feature_cols,
            'genus_df': genus_df,
            'emb': emb,
            'meta': meta
        }
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        return None


# Trait columns definition with descriptions
TRAIT_COLUMNS = [
    'perc_per', 'perc_wood', 'perc_ag', 'floral_symm', 'mating_system',
    'repro_syndrome', 'pollination_syndrome', 'RedList', 'tavg', 'C_value', 'CV_C_value'
]

TRAIT_LABELS = {
    'perc_per': 'Perenniality (%)',
    'perc_wood': 'Woodiness (%)',
    'perc_ag': 'Agamospermy (%)',
    'floral_symm': 'Floral Symmetry',
    'mating_system': 'Mating System',
    'repro_syndrome': 'Reproductive Syndrome',
    'pollination_syndrome': 'Pollination Syndrome',
    'RedList': 'Red List Status',
    'tavg': 'Average Temperature',
    'C_value': 'Genome Size (C-value)',
    'CV_C_value': 'CV of C-value'
}

# Breeder-relevant traits for improvement analysis
BREEDER_TRAITS = {
    'perc_per': 'Perenniality',
    'perc_wood': 'Woodiness',
    'perc_ag': 'Agamospermy',
    'HybProp': 'Hybrid Propensity',
    'Hyb_Ratio': 'Hybrid Ratio'
}


def get_genus_embedding(genus_name, model_bundle):
    """Get embedding for a specific genus"""
    if model_bundle is None:
        return None
    
    meta = model_bundle['meta']
    emb = model_bundle['emb']
    
    emb_idx = 0
    if 'Genus' in meta.columns:
        meta_row = meta[meta['Genus'].str.strip() == genus_name.strip()]
        if not meta_row.empty:
            emb_idx = int(meta_row.index[0])
    
    return emb[emb_idx]


def predict_hybrid_success(traits_a, traits_b, emb_a, emb_b, model_bundle):
    """
    Predict hybridization success probability between two genera
    
    Parameters:
    -----------
    traits_a : dict
        Trait dictionary for genus A
    traits_b : dict
        Trait dictionary for genus B
    emb_a : numpy.ndarray
        Embedding vector for genus A
    emb_b : numpy.ndarray
        Embedding vector for genus B
    model_bundle : dict
        Model and scaler bundle
    
    Returns:
    --------
    float : Hybridization success probability (0-1)
    """
    if model_bundle is None or emb_a is None or emb_b is None:
        return None
    
    model = model_bundle['model']
    scaler = model_bundle['scaler']
    
    # Calculate embedding similarity
    emb_sim = np.dot(emb_a, emb_b) / (np.linalg.norm(emb_a) * np.linalg.norm(emb_b) + 1e-10)
    
    # Calculate trait similarity
    numeric_cols = ['perc_per', 'perc_wood', 'perc_ag', 'floral_symm', 'mating_system',
                   'repro_syndrome', 'pollination_syndrome', 'RedList', 'tavg', 'C_value', 'CV_C_value']
    
    vec_a = np.array([traits_a.get(c, 0.0) for c in numeric_cols])
    vec_b = np.array([traits_b.get(c, 0.0) for c in numeric_cols])
    trait_sim = np.dot(vec_a, vec_b) / (np.linalg.norm(vec_a) * np.linalg.norm(vec_b) + 1e-10)
    
    # Create feature vector
    features = np.array([
        emb_sim, 0.5, trait_sim,
        traits_a.get('HybProp', -0.18), traits_b.get('HybProp', -0.18),
        traits_a.get('Hyb_Ratio', 0.0), traits_b.get('Hyb_Ratio', 0.0)
    ], dtype=np.float32).reshape(1, -1)
    
    # Predict
    prediction = model.predict(scaler.transform(features))[0]
    return float(np.clip(prediction, 0, 1))


def forecast_improvements(genus_a_name, genus_b_name, traits_a, traits_b, emb_a, emb_b, model_bundle):
    """
    Forecast trait improvement recommendations for hybridization success
    
    Parameters:
    -----------
    genus_a_name : str
        Name of genus A
    genus_b_name : str
        Name of genus B
    traits_a : dict
        Trait dictionary for genus A
    traits_b : dict
        Trait dictionary for genus B
    emb_a : numpy.ndarray
        Embedding vector for genus A
    emb_b : numpy.ndarray
        Embedding vector for genus B
    model_bundle : dict
        Model and scaler bundle
    
    Returns:
    --------
    tuple : (baseline_probability, improvements_dataframe)
    """
    if model_bundle is None:
        return None, None
    
    # Calculate baseline
    baseline = predict_hybrid_success(traits_a, traits_b, emb_a, emb_b, model_bundle)
    
    if baseline is None:
        return None, None
    
    results = []
    for trait_key, trait_label in BREEDER_TRAITS.items():
        if trait_key not in traits_a:
            continue
        
        # Test Parent A improvement
        traits_a_test = traits_a.copy()
        traits_a_test[trait_key] = min(1.0, traits_a_test[trait_key] * 1.3)
        prob_a = predict_hybrid_success(traits_a_test, traits_b, emb_a, emb_b, model_bundle)
        
        # Test Parent B improvement
        traits_b_test = traits_b.copy()
        traits_b_test[trait_key] = min(1.0, traits_b_test[trait_key] * 1.3)
        prob_b = predict_hybrid_success(traits_a, traits_b_test, emb_a, emb_b, model_bundle)
        
        gain_a = (prob_a - baseline) * 100
        gain_b = (prob_b - baseline) * 100
        
        results.append({
            'Trait': trait_label,
            f'{genus_a_name} Gain (%)': gain_a,
            f'{genus_b_name} Gain (%)': gain_b,
            'Best Parent': genus_a_name if gain_a > gain_b else genus_b_name,
            'Max Gain (%)': max(gain_a, gain_b)
        })
    
    df = pd.DataFrame(results)
    df = df.sort_values('Max Gain (%)', ascending=False)
    
    return baseline, df


def get_genus_traits(df, genus_name):
    """Extract trait data for a specific genus"""
    genus_row = df[df['Genus'] == genus_name]
    if genus_row.empty:
        return None
    return genus_row.iloc[0]


def render_trait_inputs(genus_name, traits, prefix):
    """Render editable trait input fields for a genus"""
    st.markdown(f"### 🧬 {genus_name} Traits")
    
    updated_traits = {}
    
    # Create 2 columns for better layout
    col1, col2 = st.columns(2)
    
    for idx, trait in enumerate(TRAIT_COLUMNS):
        col = col1 if idx % 2 == 0 else col2
        
        with col:
            default_value = float(traits[trait]) if pd.notna(traits[trait]) else 0.0
            updated_traits[trait] = st.number_input(
                label=TRAIT_LABELS[trait],
                value=default_value,
                format="%.6f",
                key=f"{prefix}_{trait}",
                help=f"Edit {TRAIT_LABELS[trait]} value for {genus_name}"
            )
    
    st.markdown("</div>", unsafe_allow_html=True)
    return updated_traits


def create_comparison_dataframe(genus_a_name, traits_a, genus_b_name, traits_b):
    """Create a comparison dataframe for export"""
    data = {
        'Trait': [TRAIT_LABELS[trait] for trait in TRAIT_COLUMNS],
        'Trait_Code': TRAIT_COLUMNS,
        f'{genus_a_name}': [traits_a[trait] for trait in TRAIT_COLUMNS],
        f'{genus_b_name}': [traits_b[trait] for trait in TRAIT_COLUMNS]
    }
    return pd.DataFrame(data)


def export_to_csv(df, filename):
    """Export dataframe to CSV"""
    csv = df.to_csv(index=False)
    return csv


# Main App
st.title("🌿 Pair Hybridization Test")
st.markdown("### Compare and edit trait characteristics between two plant genera")
st.markdown("---")

# Load data
try:
    df = load_data()
    model_bundle = load_model()
    genus_list = sorted(df['Genus'].unique().tolist())
    
    if model_bundle is None:
        st.warning("⚠️ Model not loaded - Only trait comparison available")
        
except Exception as e:
    st.error(f"Error loading data: {str(e)}")
    st.stop()

# Genus Selection Section
st.subheader("📊 Select Genera for Comparison")
col1, col2, col3 = st.columns([1, 0.1, 1])

with col1:
    genus_A = st.selectbox(
        '**Genus A**',
        options=genus_list,
        key='genus_a_select',
        help="Select the first genus for comparison"
    )

with col3:
    genus_B = st.selectbox(
        '**Genus B**',
        options=genus_list,
        index=min(1, len(genus_list)-1),  # Default to second genus
        key='genus_b_select',
        help="Select the second genus for comparison"
    )

st.markdown("---")

# Load button
col_btn1, col_btn2, col_btn3 = st.columns([2, 1, 2])
with col_btn2:
    load_button = st.button("📥 Load Traits", type="primary", use_container_width=True)

if load_button or 'traits_loaded' in st.session_state:
    # Fetch genus traits
    traits_A_original = get_genus_traits(df, genus_A)
    traits_B_original = get_genus_traits(df, genus_B)
    
    if traits_A_original is None:
        st.error(f"❌ No trait data found for genus: **{genus_A}**")
        st.stop()
    
    if traits_B_original is None:
        st.error(f"❌ No trait data found for genus: **{genus_B}**")
        st.stop()
    
    # Mark traits as loaded
    st.session_state['traits_loaded'] = True
    
    # Render editable trait inputs
    col_a, col_b = st.columns(2)
    
    with col_a:
        updated_traits_A = render_trait_inputs(genus_A, traits_A_original, "genus_a")
    
    with col_b:
        updated_traits_B = render_trait_inputs(genus_B, traits_B_original, "genus_b")
    
    st.markdown("---")
    
    # Create comparison dataframe
    comparison_df = create_comparison_dataframe(genus_A, updated_traits_A, genus_B, updated_traits_B)
    
    # Display comparison table
    st.markdown("#### 📊 Trait Comparison Table")
    st.dataframe(
        comparison_df[['Trait_Code', genus_A, genus_B]].style.highlight_max(axis=1, subset=[genus_A, genus_B], color='#1e4620')
                          .highlight_min(axis=1, subset=[genus_A, genus_B], color='#4a1515'),
        use_container_width=True,
        hide_index=True
    )
    
    st.markdown("---")
    
    # ============================================
    # PREDICTION SECTION
    # ============================================
    if model_bundle is not None:
        st.subheader("🤖 Hybridization Success Prediction")
        
        col_pred1, col_pred2, col_pred3 = st.columns([1, 3, 1])
        
        with col_pred2:
            predict_button = st.button("🔮 Predict Success", type="primary", use_container_width=True)
        
        if predict_button or 'prediction_made' in st.session_state:
            st.session_state['prediction_made'] = True
            
            with st.spinner("🧬 Analyzing hybridization potential..."):
                # Get embeddings
                emb_a = get_genus_embedding(genus_A, model_bundle)
                emb_b = get_genus_embedding(genus_B, model_bundle)
                
                # Convert updated traits to dict format with all necessary fields
                full_traits_a = traits_A_original.to_dict()
                full_traits_a.update(updated_traits_A)
                
                full_traits_b = traits_B_original.to_dict()
                full_traits_b.update(updated_traits_B)
                
                # Predict success
                success_prob = predict_hybrid_success(
                    full_traits_a, full_traits_b, emb_a, emb_b, model_bundle
                )
                
                # Forecast improvements
                baseline_prob, improvements_df = forecast_improvements(
                    genus_A, genus_B, full_traits_a, full_traits_b, emb_a, emb_b, model_bundle
                )
                
                if success_prob is not None:
                    # Display prediction results
                    st.markdown("---")
                    st.markdown("### 📈 Prediction Results")
                    
                    # Main metrics
                    metric_col1, metric_col2, metric_col3, metric_col4 = st.columns(4)
                    
                    with metric_col1:
                        st.metric(
                            label="Hybridization Success",
                            value=f"{success_prob*100:.2f}%",
                            help="Predicted probability of successful hybridization"
                        )
                    
                    with metric_col2:
                        success_level = "High" if success_prob > 0.7 else "Medium" if success_prob > 0.4 else "Low"
                        color = "🟢" if success_prob > 0.7 else "🟡" if success_prob > 0.4 else "🔴"
                        st.metric(
                            label="Success Level",
                            value=f"{color} {success_level}",
                            help="Qualitative assessment of hybridization potential"
                        )
                    
                    with metric_col3:
                        # Calculate trait similarity
                        numeric_cols = TRAIT_COLUMNS
                        vec_a = np.array([updated_traits_A[c] for c in numeric_cols])
                        vec_b = np.array([updated_traits_B[c] for c in numeric_cols])
                        trait_sim = np.dot(vec_a, vec_b) / (np.linalg.norm(vec_a) * np.linalg.norm(vec_b) + 1e-10)
                        
                        st.metric(
                            label="Trait Similarity",
                            value=f"{trait_sim*100:.2f}%",
                            help="Cosine similarity between trait vectors"
                        )
                    
                    with metric_col4:
                        # Calculate embedding similarity
                        emb_sim = np.dot(emb_a, emb_b) / (np.linalg.norm(emb_a) * np.linalg.norm(emb_b) + 1e-10)
                        st.metric(
                            label="Genetic Similarity",
                            value=f"{emb_sim*100:.2f}%",
                            help="Embedding-based genetic similarity"
                        )
                    
                    st.markdown("---")
                    
                    # Improvement recommendations
                    if improvements_df is not None and not improvements_df.empty:
                        st.markdown("### 🔧 Trait Improvement Recommendations")
                        st.markdown(f"**Baseline Success Rate:** {baseline_prob*100:.2f}%")
                        
                        # Format the dataframe with styled display
                        formatted_df = improvements_df.copy()
                        formatted_df[f'{genus_A} Gain (%)'] = formatted_df[f'{genus_A} Gain (%)'].apply(lambda x: f"{x:+.2f}")
                        formatted_df[f'{genus_B} Gain (%)'] = formatted_df[f'{genus_B} Gain (%)'].apply(lambda x: f"{x:+.2f}")
                        formatted_df['Max Gain (%)'] = formatted_df['Max Gain (%)'].apply(lambda x: f"{x:+.2f}")
                        
                        # Add color indicators
                        def add_color_indicator(val_str):
                            try:
                                val = float(val_str)
                                if val > 5:
                                    return f"🟢 {val_str}"
                                elif val > 1:
                                    return f"🟡 {val_str}"
                                elif val > 0:
                                    return f"⚪ {val_str}"
                                else:
                                    return f"🔴 {val_str}"
                            except:
                                return val_str
                        
                        formatted_df['Max Gain (%)'] = formatted_df['Max Gain (%)'].apply(add_color_indicator)
                        
                        st.dataframe(
                            formatted_df.reset_index(drop=True),
                            use_container_width=True,
                            hide_index=True
                        )
                        
                        # Top recommendation
                        top_trait = improvements_df.iloc[0]
                        if top_trait['Max Gain (%)'] > 0:
                            st.success(
                                f"✨ **Top Recommendation:** Improve **{top_trait['Trait']}** in "
                                f"**{top_trait['Best Parent']}** for up to **+{top_trait['Max Gain (%)']:.2f}%** gain"
                            )
                            
                            # Link top recommendation to practical benefits
                            st.markdown("#### 🎯 Expected Agricultural Outcomes:")
                            
                            outcome_mapping = {
                                'Perenniality': [
                                    '**Enhanced Earliness:** Faster crop establishment and earlier harvest windows',
                                    '**Sustained Yield Potential:** Multi-season production without replanting',
                                ],
                                'Woodiness': [
                                    '**Superior Drought Tolerance:** Deep root systems and enhanced water retention',
                                    '**Improved Disease Resistance:** Stronger cell walls and natural pathogen barriers',
                                ],
                                'Agamospermy': [
                                    '**Consistent Maturity Timing:** Uniform crop development across generations',
                                    '**Stable Yield Performance:** Predictable production year after year',
                                ],
                                'Hybrid Propensity': [
                                    '**Increased Salinity Tolerance:** Better osmotic adjustment in saline conditions',
                                    '**Heterosis Effects:** Hybrid vigor leading to enhanced overall performance',
                                ],
                                'Hybrid Ratio': [
                                    '**Improved Stress Adaptation:** Greater resilience to environmental challenges',
                                    '**Enhanced Yield Stability:** More consistent production under variable conditions',
                                ]
                            }
                            
                            if top_trait['Trait'] in outcome_mapping:
                                st.markdown(f"By improving **{top_trait['Trait']}**, you can expect:")
                                for outcome in outcome_mapping[top_trait['Trait']]:
                                    st.markdown(f"• {outcome}")
                        else:
                            st.info("💡 Current trait values are near optimal for this combination")
                        
                
                else:
                    st.error("❌ Prediction failed. Please check your inputs.")
    
    st.markdown("---")
    
    col_export1, col_export2, col_export3 = st.columns([1, 1, 1])
    
    with col_export2:
        if st.button("🔄 Reset All", use_container_width=True):
            st.session_state.clear()
            st.rerun()
