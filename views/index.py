import streamlit as st
import os
# Page configuration
st.set_page_config(
    page_title="AgroX - Plant Analysis Platform",
    page_icon="🌱",
    layout="wide",
    initial_sidebar_state="collapsed"
)



st.markdown("""
<style>
    /* Global styling */
    .main {
        background: linear-gradient(to bottom, #f8fffe 0%, #e8f5e9 100%);
    }
            
            .st-emotion-cache-scp8yw {
            display: none !important; }
    
    .main-header {
        text-align: center;
        padding: 3rem 2rem;
        background: linear-gradient(135deg, #2e7d32 0%, #66bb6a 50%, #81c784 100%);
        color: white;
        border-radius: 20px;
        margin-bottom: 3rem;
        box-shadow: 0 10px 30px rgba(46, 125, 50, 0.3);
        position: relative;
        overflow: hidden;
    }
    
    .main-header::before {
        content: '';
        position: absolute;
        top: -50%;
        left: -50%;
        width: 200%;
        height: 200%;
        background: radial-gradient(circle, rgba(255,255,255,0.1) 0%, transparent 70%);
        animation: pulse 4s ease-in-out infinite;
    }
    
    @keyframes pulse {
        0%, 100% { transform: scale(1); opacity: 0.5; }
        50% { transform: scale(1.1); opacity: 0.8; }
    }
    
    .header-title {
        font-size: 3rem;
        font-weight: 800;
        margin-bottom: 0.5rem;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.2);
        position: relative;
        z-index: 1;
    }
    
    .header-subtitle {
        font-size: 1.3rem;
        margin-top: 0.5rem;
        opacity: 0.95;
        position: relative;
        z-index: 1;
    }
    
    .intro-text {
        text-align: center;
        font-size: 1.2rem;
        color: #2e7d32;
        margin-bottom: 2rem;
        font-weight: 500;
    }
    
    .card {
        background: linear-gradient(135deg, #ffffff 0%, #f1f8f4 100%);
        padding: 2.5rem;
        border-radius: 20px;
        box-shadow: 0 8px 20px rgba(0, 0, 0, 0.08);
        transition: all 0.4s cubic-bezier(0.175, 0.885, 0.32, 1.275);
        border: 2px solid transparent;
        height: 100%;
        margin-bottom: 1rem;
        position: relative;
        overflow: hidden;
    }
    
    .card::before {
        content: '';
        position: absolute;
        top: 0;
        left: 0;
        right: 0;
        height: 5px;
        background: linear-gradient(90deg, #2e7d32, #66bb6a, #81c784);
        transform: scaleX(0);
        transition: transform 0.4s ease;
    }
    
    .card:hover::before {
        transform: scaleX(1);
    }
    
    .card:hover {
        transform: translateY(-10px) scale(1.02);
        box-shadow: 0 15px 40px rgba(46, 125, 50, 0.15);
        border-color: #66bb6a;
    }
    
    .card-icon {
        font-size: 4rem;
        margin-bottom: 1.5rem;
        display: inline-block;
        animation: float 3s ease-in-out infinite;
    }
    
    @keyframes float {
        0%, 100% { transform: translateY(0px); }
        50% { transform: translateY(-10px); }
    }
    
    .card-title {
        font-size: 1.8rem;
        font-weight: 700;
        background: linear-gradient(135deg, #1b5e20, #4caf50);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        margin-bottom: 1rem;
    }
    
    .card-description {
        color: #555;
        line-height: 1.8;
        margin-bottom: 2rem;
        font-size: 1.05rem;
    }
    
    .feature-badge {
        display: inline-block;
        background: linear-gradient(135deg, #4caf50, #66bb6a);
        color: white;
        padding: 0.4rem 1rem;
        border-radius: 20px;
        font-size: 0.85rem;
        font-weight: 600;
        margin-top: 1rem;
        box-shadow: 0 2px 8px rgba(76, 175, 80, 0.3);
    }
    
    .stButton > button {
        width: 100%;
        background: linear-gradient(135deg, #2e7d32 0%, #4caf50 100%);
        color: white;
        border: none;
        border-radius: 12px;
        padding: 1rem 2rem;
        font-size: 1.1rem;
        font-weight: 700;
        transition: all 0.3s ease;
        margin-inline: auto;
        box-shadow: 0 4px 15px rgba(46, 125, 50, 0.3);
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }
    
    .stButton > button:hover {
        background: linear-gradient(135deg, #1b5e20 0%, #2e7d32 100%);
        transform: translateY(-2px);
        box-shadow: 0 6px 20px rgba(46, 125, 50, 0.4);
    }
    
    .stButton > button:active {
        transform: translateY(0px);
    }
    
    .footer {
        text-align: center;
        color: #666;
        padding: 2rem 0;
        margin-top: 3rem;
        border-top: 2px solid rgba(46, 125, 50, 0.1);
    }
    
    .footer-text {
        font-size: 0.95rem;
        color: #2e7d32;
        font-weight: 500;
    }
    
    /* Hide Streamlit branding */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
</style>
""", unsafe_allow_html=True)

st.markdown("""
<div class="main-header">
    <h1 class="header-title">🌱 AgroX Platform</h1>
    <p class="header-subtitle">Advanced Plant Analysis & Hybridization Intelligence</p>
</div>
""", unsafe_allow_html=True)

st.markdown("""
<p class="intro-text">
    ✨ Welcome to the future of plant science! Choose your analytical tool to begin your journey.
</p>
""", unsafe_allow_html=True)

col1, col2, col3 = st.columns(3, gap="large")

with col1:
    st.markdown("""
    <div class="card">
        <div class="card-icon">🌿</div>
        <div class="card-title">Hybridization Test</div>
        <div class="card-description">
            Discover the compatibility between plant pairs with our advanced prediction model. 
            Select parent genera and unlock insights into hybrid formation potential using 
            machine learning algorithms trained on extensive botanical data.
        </div>
        <div class="feature-badge">🎯 AI-Powered Predictions</div>
    </div>
    """, unsafe_allow_html=True)
    
    if st.button("🚀 Launch Hybridization Test", key="hyb_test"):
       st.switch_page(os.path.join("pages", "pair_hyb_test.py")) 

with col2:
    st.markdown("""
    <div class="card">
        <div class="card-icon">🔬</div>
        <div class="card-title">Trait Analyzer</div>
        <div class="card-description">
            Deep dive into plant characteristics with our comprehensive trait analysis system. 
            Input morphological, ecological, and physiological traits to generate detailed 
            hybridization forecasts backed by scientific data.
        </div>
        <div class="feature-badge">📊 Multi-Trait Analysis</div>
    </div>
    """, unsafe_allow_html=True)
    
    if st.button("🚀 Launch Trait Analyzer", key="trait_analyzer"):
        st.switch_page(os.path.join("pages", "trait_analyser.py"))

with col3:
    st.markdown("""
    <div class="card">
        <div class="card-icon">🏆</div>
        <div class="card-title">Best Hybrid Pairs</div>
        <div class="card-description">
            Explore top-performing hybridization combinations across all genera. 
            Discover the most promising plant pairs or find the best partners for a specific 
            genus using our comprehensive success prediction algorithms.
        </div>
        <div class="feature-badge">🌟 Top Recommendations</div>
    </div>
    """, unsafe_allow_html=True)
    
    if st.button("🚀 Launch Best Pairs", key="best_pairs"):
        st.switch_page(os.path.join("pages", "best_pairs.py"))

st.markdown("<br>", unsafe_allow_html=True)

# Second row for Algeria scenario designer
col4, col5, col6 = st.columns(3, gap="large")

with col4:
    st.markdown("""
    <div class="card">
        <div class="card-icon">🌍</div>
        <div class="card-title">Algeria Climate Designer</div>
        <div class="card-description">
            Design custom hybridization scenarios tailored to Algerian climate zones. 
            Adjust stress weights (drought, salinity, frost) and zone-specific parameters 
            to identify optimal hybrid pairs for Coastal, High Plateau, and Saharan regions.
        </div>
        <div class="feature-badge">🗺️ Regional Optimization</div>
    </div>
    """, unsafe_allow_html=True)
    
    if st.button("🚀 Launch Algeria Designer", key="algeria"):
        st.switch_page(os.path.join("pages", "algeria.py"))

# Footer
st.markdown("<br><br>", unsafe_allow_html=True)
st.markdown("""
<div class="footer">
    <p class="footer-text">🌱 AgroX Platform © 2025 - Empowering Botanical Research Through Technology</p>
    <p style="color: #999; font-size: 0.85rem; margin-top: 0.5rem;">
        Powered by Machine Learning & Botanical Science
    </p>
</div>
""", unsafe_allow_html=True)
