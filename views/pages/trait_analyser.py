import streamlit as st
import pandas as pd
import os
import joblib

st.set_page_config(
    page_title="Trait Analyzer",
    page_icon="🔬",
    layout="wide"
)

st.markdown("""
    <style>
        /* Hide increment/decrement buttons */
        button[data-testid="stNumberInputStepUp"],
        button[data-testid="stNumberInputStepDown"] {
            display: none;
        }
            
            .st-emotion-cache-scp8yw {
            display: none !important; }
    
        
        /* Input field styling */
        input[type="number"] {
            border-radius: 8px;
            border-color: #d3d3d3 !important;
        }
        
        input[type="number"]:focus {
            border-color: #4CAF50 !important;
            box-shadow: 0 0 0 2px rgba(76, 175, 80, 0.2) !important;
        }
        
        /* Button styling */
        .stButton > button {
            border-radius: 8px;
            font-weight: 600;
        }
        
        /* Section headers */
        .trait-section {
            background-color: rgba(76, 175, 80, 0.1);
            padding: .8rem;
            border-radius: 8px;
            margin-bottom: 1rem;
            font-weight: 600;
        }
    </style>
""", unsafe_allow_html=True)

current_dir = os.path.dirname(os.path.abspath(__file__))
genus_data_path = os.path.join(current_dir, '..', '..', 'Hybridization_Propensity_Predictor', 'genus_processed.csv')
model_path = os.path.join(current_dir, '..', '..', 'Hybridization_Propensity_Predictor', 'genus_hybprop_model.pkl')
@st.cache_data
def load_genus_data():
    """Load and cache genus trait data"""
    return pd.read_csv(genus_data_path)

@st.cache_resource
def load_model():
    """Load and cache the trained model"""
    return joblib.load(model_path)

TRAIT_CONFIG = {
    'perc_per': {
        'label': '🌱 Perennial Percentage',
        'min': 0.0, 'max': 1.0, 'step': 0.01, 'format': 4,
        'help': 'Percentage of perennial species in genus'
    },
    'perc_wood': {
        'label': '🌳 Woody Percentage',
        'min': 0.0, 'max': 1.0, 'step': 0.01, 'format': 4,
        'help': 'Percentage of woody species in genus'
    },
    'perc_ag': {
        'label': '🌾 Agricultural Percentage',
        'min': 0.0, 'max': 1.0, 'step': 0.001, 'format': 4,
        'help': 'Percentage of agricultural species in genus'
    },
    'floral_symm': {
        'label': '🌸 Floral Symmetry',
        'min': 0.0, 'max': None, 'step': 0.1, 'format': 1,
        'help': 'Floral symmetry measurement'
    },
    'mating_system': {
        'label': '💐 Mating System',
        'min': 0.0, 'max': None, 'step': 0.1, 'format': 1,
        'help': 'Mating system classification'
    },
    'repro_syndrome': {
        'label': '🌺 Reproduction Syndrome',
        'min': 0.0, 'max': None, 'step': 0.1, 'format': 4,
        'help': 'Reproduction syndrome type'
    },
    'pollination_syndrome': {
        'label': '🐝 Pollination Syndrome',
        'min': 0.0, 'max': None, 'step': 0.1, 'format': 4,
        'help': 'Pollination syndrome classification'
    },
    'RedList': {
        'label': '⚠️ Red List Status',
        'min': 0.0, 'max': 1.0, 'step': 0.01, 'format': 4,
        'help': 'Conservation status from Red List'
    },
    'tavg': {
        'label': '🌡️ Average Temperature',
        'min': None, 'max': None, 'step': 0.1, 'format': 4,
        'help': 'Average temperature in habitat'
    },
    'C_value': {
        'label': '🧬 C-Value',
        'min': 0.0, 'max': None, 'step': 0.01, 'format': 4,
        'help': 'Genome size measurement'
    },
    'CV_C_value': {
        'label': '📊 CV C-Value',
        'min': 0.0, 'max': None, 'step': 0.01, 'format': 2,
        'help': 'Coefficient of variation of C-value'
    }
}


def create_input_field(key, config, default_value=0.0, genus_suffix=''):
    """
    Create a styled number input field
    
    Parameters:
    -----------
    key : str
        Unique identifier for the input field
    config : dict
        Configuration containing label, min, max, step, format (decimal places), and help text
    default_value : float
        Default value to populate the input field
    genus_suffix : str
        Genus name to append to key for unique identification
    """
    kwargs = {
        'label': config['label'],
        'value': default_value,
        'step': config['step'],
        'help': config['help'],
        'key': f"{key}_{genus_suffix}",
        'format': f"%.{config.get('format', 2)}f"  # Default to 2 decimal places
    }
    
    return st.number_input(**kwargs)


def collect_trait_data(trait_values):
    """Organize trait data into a DataFrame"""
    return pd.DataFrame([trait_values])

df = load_genus_data()

st.title("🔬 Trait Analyzer")
st.markdown("### Enter custom trait values for analysis")
st.markdown("---")


st.markdown("### 📝 Enter Trait Values")

# Genus selection with automatic Family and Order lookup
col_genus, col_family, col_order = st.columns(3)

with col_genus:
    selected_genus = st.selectbox(
        'Select Genus',
        options=df['Genus'].unique().tolist(),
        key='genus_reference_select',
        help="Select a genus from the dataset"
    )

genus_row = df[df['Genus'] == selected_genus].iloc[0]
selected_family = genus_row['Family']
selected_order = genus_row['Order']

with col_family:
    st.text_input(
        'Family ',
        value=selected_family,
        disabled=True,
        key=f'family_field_{selected_genus}'
    )

with col_order:
    st.text_input(
        'Order ',
        value=selected_order,
        disabled=True,
        key=f'order_field_{selected_genus}'
    )

st.markdown("---")

# Get default values from the selected genus row
default_trait_values = {}
for trait in TRAIT_CONFIG.keys():
    if trait in genus_row.index:
        default_trait_values[trait] = float(genus_row[trait])
    else:
        default_trait_values[trait] = 0.0

col1, col2 = st.columns(2)

trait_values = {
    'Genus': selected_genus,
    'Family': selected_family,
    'Order': selected_order,
}

traits_list = list(TRAIT_CONFIG.keys())
mid_point = len(traits_list) // 2

with col1:
    st.markdown('<div class="trait-section">🌿 Morphological & Life History Traits</div', unsafe_allow_html=True)
    for trait in traits_list[:mid_point]:
        trait_values[trait] = create_input_field(trait, TRAIT_CONFIG[trait], default_trait_values[trait], selected_genus)
    st.markdown('</div>', unsafe_allow_html=True)

with col2:
    st.markdown('<div class="trait-section">🌍 Ecological & Genetic Traits</div', unsafe_allow_html=True)
    for trait in traits_list[mid_point:]:
        trait_values[trait] = create_input_field(trait, TRAIT_CONFIG[trait], default_trait_values[trait], selected_genus)
    st.markdown('</div>', unsafe_allow_html=True)

st.markdown("---")

col_btn1, col_btn2, col_btn3, col_btn4 = st.columns([1, 1, 1, 1])

with col_btn2:
    analyze_button = st.button("🔬 Analyze Traits", type="primary", use_container_width=True)

with col_btn3:
    reset_button = st.button("🔄 Reset All", type="secondary", use_container_width=True)

if reset_button:
    st.rerun()

if analyze_button:
    with st.spinner("Processing trait data and making prediction..."):
        try:
            # Create DataFrame with correct column order for model
            # Model expects: Genus, Family, Order, then all numeric traits
            input_data = {
                'Genus': [selected_genus],
                'Family': [selected_family],
                'Order': [selected_order],
            }
            
            # Add numeric traits in the order they were defined
            for trait in TRAIT_CONFIG.keys():
                input_data[trait] = [trait_values[trait]]
            
            # Create DataFrame
            trait_df = pd.DataFrame(input_data)
            
            # Load model and make prediction
            model = load_model()
            prediction = model.predict(trait_df)[0]
            
            # Store in session state
            st.session_state['trait_data'] = trait_df
            st.session_state['trait_values'] = trait_values
            st.session_state['prediction'] = prediction
            
            
            # Display prediction result prominently
            st.markdown("### 🎯 Prediction Result")
            col_pred1, col_pred2, col_pred3 = st.columns([1, 2, 1])
            with col_pred2:
                st.metric(
                    label="Predicted HybProp (Hybridization Propensity)",
                    value=f"{prediction:.4f}",
                    help="Higher values indicate greater hybridization propensity"
                )
            
            st.markdown("---")
            
            # Display taxonomic information
            st.markdown("### 📋 Input Values")
            st.markdown("**Taxonomic Information:**")
            tax_col1, tax_col2, tax_col3 = st.columns(3)
            with tax_col1:
                st.info(f"**Genus:** {selected_genus}")
            with tax_col2:
                st.info(f"**Family:** {selected_family}")
            with tax_col3:
                st.info(f"**Order:** {selected_order}")
            
            # Display trait values
            st.markdown("**Trait Values:**")
            display_data = pd.DataFrame({
                'Trait': [TRAIT_CONFIG[k]['label'] for k in TRAIT_CONFIG.keys()],
                'Value': [trait_values[k] for k in TRAIT_CONFIG.keys()]
            })
            
            st.dataframe(
                display_data.style.format({'Value': '{:.4f}'}),
                use_container_width=True,
                height=400
            )
            
        except Exception as e:
            st.error(f"❌ Prediction failed: {str(e)}")
            st.info("Please check that all values are entered correctly and the model file exists.")
        
        
        # # Display data table
        # st.markdown("### � Entered Values")
        
        # # Create display DataFrame with better formatting
        # display_data = pd.DataFrame({
        #     'Trait': [TRAIT_CONFIG[k]['label'] for k in trait_values.keys()],
        #     'Value': list(trait_values.values())
        # })
        
        # st.dataframe(
        #     display_data.style.format({'Value': '{:.4f}'}),
        #     use_container_width=True,
        #     height=400
        # )
        



