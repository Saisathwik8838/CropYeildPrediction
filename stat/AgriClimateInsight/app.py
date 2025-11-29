"""
AgriClimate Insight Portal - Main Streamlit Application
Interactive dashboard for climate-agriculture data analysis and modeling
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.express as px
from datetime import datetime, date
import sys
import os

# Add utils directory to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'utils'))

# Import our custom modules
from utils.data_preprocessing import DataPreprocessor
from utils.analysis import StatisticalAnalyzer
from utils.modeling import MLModeler
from utils.visualization import DataVisualizer

# Page configuration
st.set_page_config(
    page_title="AgriClimate Insight Portal",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Premium Design System CSS
st.markdown("""
<style>
    /* Import Google Fonts */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');
    
    /* Root Variables */
    :root {
        /* Color Palette */
        --primary-50: #f0f9ff;
        --primary-100: #e0f2fe;
        --primary-200: #bae6fd;
        --primary-300: #7dd3fc;
        --primary-400: #38bdf8;
        --primary-500: #0ea5e9;
        --primary-600: #0284c7;
        --primary-700: #0369a1;
        --primary-800: #075985;
        --primary-900: #0c4a6e;
        
        --secondary-50: #f8fafc;
        --secondary-100: #f1f5f9;
        --secondary-200: #e2e8f0;
        --secondary-300: #cbd5e1;
        --secondary-400: #94a3b8;
        --secondary-500: #64748b;
        --secondary-600: #475569;
        --secondary-700: #334155;
        --secondary-800: #1e293b;
        --secondary-900: #0f172a;
        
        /* Semantic Colors */
        --rainfall-50: #eff6ff;
        --rainfall-100: #dbeafe;
        --rainfall-200: #bfdbfe;
        --rainfall-300: #93c5fd;
        --rainfall-400: #60a5fa;
        --rainfall-500: #3b82f6;
        --rainfall-600: #2563eb;
        --rainfall-700: #1d4ed8;
        --rainfall-800: #1e40af;
        --rainfall-900: #1e3a8a;
        
        --temperature-50: #fef2f2;
        --temperature-100: #fee2e2;
        --temperature-200: #fecaca;
        --temperature-300: #fca5a5;
        --temperature-400: #f87171;
        --temperature-500: #ef4444;
        --temperature-600: #dc2626;
        --temperature-700: #b91c1c;
        --temperature-800: #991b1b;
        --temperature-900: #7f1d1d;
        
        --crop-50: #f0fdf4;
        --crop-100: #dcfce7;
        --crop-200: #bbf7d0;
        --crop-300: #86efac;
        --crop-400: #4ade80;
        --crop-500: #22c55e;
        --crop-600: #16a34a;
        --crop-700: #15803d;
        --crop-800: #166534;
        --crop-900: #14532d;
        
        /* Typography */
        --font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
        --font-size-xs: 0.75rem;
        --font-size-sm: 0.875rem;
        --font-size-base: 1rem;
        --font-size-lg: 1.125rem;
        --font-size-xl: 1.25rem;
        --font-size-2xl: 1.5rem;
        --font-size-3xl: 1.875rem;
        --font-size-4xl: 2.25rem;
        --font-size-5xl: 3rem;
        
        /* Spacing (8px grid) */
        --space-1: 0.25rem;
        --space-2: 0.5rem;
        --space-3: 0.75rem;
        --space-4: 1rem;
        --space-5: 1.25rem;
        --space-6: 1.5rem;
        --space-8: 2rem;
        --space-10: 2.5rem;
        --space-12: 3rem;
        --space-16: 4rem;
        --space-20: 5rem;
        
        /* Border Radius */
        --radius-sm: 0.375rem;
        --radius-md: 0.5rem;
        --radius-lg: 0.75rem;
        --radius-xl: 1rem;
        --radius-2xl: 1.5rem;
        
        /* Shadows */
        --shadow-sm: 0 1px 2px 0 rgb(0 0 0 / 0.05);
        --shadow-md: 0 4px 6px -1px rgb(0 0 0 / 0.1), 0 2px 4px -2px rgb(0 0 0 / 0.1);
        --shadow-lg: 0 10px 15px -3px rgb(0 0 0 / 0.1), 0 4px 6px -4px rgb(0 0 0 / 0.1);
        --shadow-xl: 0 20px 25px -5px rgb(0 0 0 / 0.1), 0 8px 10px -6px rgb(0 0 0 / 0.1);
        --shadow-2xl: 0 25px 50px -12px rgb(0 0 0 / 0.25);
    }
    
    /* Global Styles */
    * {
        font-family: var(--font-family);
    }
    
    .main-header {
        font-size: var(--font-size-5xl);
        font-weight: 800;
        color: var(--secondary-900);
        text-align: center;
        margin-bottom: var(--space-12);
        background: linear-gradient(135deg, var(--primary-600), var(--primary-800));
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        letter-spacing: -0.025em;
        line-height: 1.1;
    }
    
    .sub-header {
        font-size: var(--font-size-2xl);
        font-weight: 700;
        color: var(--secondary-800);
        margin-top: var(--space-8);
        margin-bottom: var(--space-6);
        padding-bottom: var(--space-3);
        border-bottom: 3px solid var(--primary-200);
        position: relative;
    }
    
    .sub-header::after {
        content: '';
        position: absolute;
        bottom: -3px;
        left: 0;
        width: 60px;
        height: 3px;
        background: linear-gradient(90deg, var(--primary-500), var(--primary-300));
        border-radius: var(--radius-sm);
    }
    
    /* Premium Metric Cards */
    .metric-card {
        background: linear-gradient(135deg, rgba(255,255,255,0.9), rgba(248,250,252,0.8));
        backdrop-filter: blur(10px);
        padding: var(--space-6);
        border-radius: var(--radius-xl);
        border: 1px solid rgba(255,255,255,0.2);
        margin: var(--space-3) 0;
        box-shadow: var(--shadow-lg);
        transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
        position: relative;
        overflow: hidden;
    }
    
    .metric-card::before {
        content: '';
        position: absolute;
        top: 0;
        left: 0;
        right: 0;
        height: 4px;
        background: linear-gradient(90deg, var(--primary-500), var(--primary-300));
    }
    
    .metric-card:hover {
        transform: translateY(-2px);
        box-shadow: var(--shadow-xl);
    }
    
    .metric-title {
        font-size: var(--font-size-sm);
        font-weight: 600;
        color: var(--secondary-600);
        text-transform: uppercase;
        letter-spacing: 0.05em;
        margin-bottom: var(--space-2);
    }
    
    .metric-value {
        font-size: var(--font-size-3xl);
        font-weight: 800;
        color: var(--secondary-900);
        line-height: 1;
    }
    
    .metric-delta {
        font-size: var(--font-size-sm);
        font-weight: 600;
        margin-top: var(--space-2);
        padding: var(--space-1) var(--space-2);
        border-radius: var(--radius-sm);
        display: inline-block;
    }
    
    .metric-delta.positive {
        color: var(--crop-700);
        background-color: var(--crop-50);
    }
    
    .metric-delta.negative {
        color: var(--temperature-700);
        background-color: var(--temperature-50);
    }
    
    /* Premium Info Boxes */
    .info-box {
        background: linear-gradient(135deg, var(--rainfall-50), rgba(255,255,255,0.8));
        backdrop-filter: blur(10px);
        padding: var(--space-6);
        border-radius: var(--radius-xl);
        border: 1px solid var(--rainfall-200);
        margin: var(--space-4) 0;
        box-shadow: var(--shadow-md);
        position: relative;
        overflow: hidden;
    }
    
    .info-box::before {
        content: '';
        position: absolute;
        top: 0;
        left: 0;
        right: 0;
        height: 4px;
        background: linear-gradient(90deg, var(--rainfall-500), var(--rainfall-300));
    }
    
    .warning-box {
        background: linear-gradient(135deg, #fef3c7, rgba(255,255,255,0.8));
        backdrop-filter: blur(10px);
        padding: var(--space-6);
        border-radius: var(--radius-xl);
        border: 1px solid #fbbf24;
        margin: var(--space-4) 0;
        box-shadow: var(--shadow-md);
        position: relative;
        overflow: hidden;
    }
    
    .warning-box::before {
        content: '';
        position: absolute;
        top: 0;
        left: 0;
        right: 0;
        height: 4px;
        background: linear-gradient(90deg, #f59e0b, #fbbf24);
    }
    
    .success-box {
        background: linear-gradient(135deg, var(--crop-50), rgba(255,255,255,0.8));
        backdrop-filter: blur(10px);
        padding: var(--space-6);
        border-radius: var(--radius-xl);
        border: 1px solid var(--crop-200);
        margin: var(--space-4) 0;
        box-shadow: var(--shadow-md);
        position: relative;
        overflow: hidden;
    }
    
    .success-box::before {
        content: '';
        position: absolute;
        top: 0;
        left: 0;
        right: 0;
        height: 4px;
        background: linear-gradient(90deg, var(--crop-500), var(--crop-300));
    }
    
    /* Sidebar Styling */
    .sidebar .sidebar-content {
        background: linear-gradient(180deg, var(--secondary-50), rgba(255,255,255,0.9));
        backdrop-filter: blur(20px);
        border-right: 1px solid var(--secondary-200);
    }
    
    .sidebar .sidebar-content .block-container {
        padding: var(--space-6);
    }
    
    /* Form Controls */
    .stSelectbox > div > div {
        background: rgba(255,255,255,0.9);
        backdrop-filter: blur(10px);
        border: 1px solid var(--secondary-200);
        border-radius: var(--radius-lg);
        box-shadow: var(--shadow-sm);
        transition: all 0.2s ease;
    }
    
    .stSelectbox > div > div:hover {
        border-color: var(--primary-300);
        box-shadow: var(--shadow-md);
    }
    
    .stSlider > div > div > div {
        background: var(--primary-200);
        border-radius: var(--radius-lg);
    }
    
    .stSlider > div > div > div > div {
        background: linear-gradient(90deg, var(--primary-500), var(--primary-300));
    }
    
    /* Premium Buttons */
    .stButton > button {
        background: linear-gradient(135deg, var(--primary-500), var(--primary-600));
        color: white;
        border-radius: var(--radius-lg);
        border: none;
        padding: var(--space-3) var(--space-6);
        font-weight: 600;
        font-size: var(--font-size-sm);
        box-shadow: var(--shadow-md);
        transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
        position: relative;
        overflow: hidden;
    }
    
    .stButton > button::before {
        content: '';
        position: absolute;
        top: 0;
        left: -100%;
        width: 100%;
        height: 100%;
        background: linear-gradient(90deg, transparent, rgba(255,255,255,0.2), transparent);
        transition: left 0.5s;
    }
    
    .stButton > button:hover {
        background: linear-gradient(135deg, var(--primary-600), var(--primary-700));
        transform: translateY(-1px);
        box-shadow: var(--shadow-lg);
    }
    
    .stButton > button:hover::before {
        left: 100%;
    }
    
    .stButton > button:active {
        transform: translateY(0);
        box-shadow: var(--shadow-md);
    }
    
    /* Premium Tabs */
    .stTabs [data-baseweb="tab-list"] {
        gap: var(--space-2);
        background: var(--secondary-100);
        padding: var(--space-2);
        border-radius: var(--radius-xl);
        box-shadow: var(--shadow-sm);
    }
    
    .stTabs [data-baseweb="tab"] {
        background: transparent;
        border-radius: var(--radius-lg);
        padding: var(--space-3) var(--space-6);
        font-weight: 600;
        font-size: var(--font-size-sm);
        transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
        position: relative;
        overflow: hidden;
    }
    
    .stTabs [data-baseweb="tab"]:hover {
        background: rgba(255,255,255,0.5);
        transform: translateY(-1px);
    }
    
    .stTabs [aria-selected="true"] {
        background: linear-gradient(135deg, var(--primary-500), var(--primary-600));
        color: white;
        box-shadow: var(--shadow-md);
        transform: translateY(-2px);
    }
    
    .stTabs [aria-selected="true"]::before {
        content: '';
        position: absolute;
        top: 0;
        left: 0;
        right: 0;
        bottom: 0;
        background: linear-gradient(45deg, transparent 30%, rgba(255,255,255,0.1) 50%, transparent 70%);
        animation: shimmer 2s infinite;
    }
    
    @keyframes shimmer {
        0% { transform: translateX(-100%); }
        100% { transform: translateX(100%); }
    }
    
    /* Chart Containers */
    .stPlotlyChart {
        border-radius: var(--radius-xl);
        overflow: hidden;
        box-shadow: var(--shadow-lg);
        background: white;
        margin: var(--space-4) 0;
    }
    
    /* Data Tables */
    .stDataFrame {
        border-radius: var(--radius-xl);
        overflow: hidden;
        box-shadow: var(--shadow-md);
        background: white;
        margin: var(--space-4) 0;
    }
    
    /* Loading Spinner */
    .stSpinner > div {
        border-color: var(--primary-200);
        border-top-color: var(--primary-500);
    }
    
    /* Responsive Design */
    @media (max-width: 768px) {
        .main-header {
            font-size: var(--font-size-4xl);
            margin-bottom: var(--space-8);
        }
        
        .sub-header {
            font-size: var(--font-size-xl);
            margin-top: var(--space-6);
            margin-bottom: var(--space-4);
        }
        
        .metric-card {
            padding: var(--space-4);
        }
        
        .metric-value {
            font-size: var(--font-size-2xl);
        }
    }
    
    /* Accessibility */
    @media (prefers-reduced-motion: reduce) {
        * {
            animation-duration: 0.01ms !important;
            animation-iteration-count: 1 !important;
            transition-duration: 0.01ms !important;
        }
    }
    
    /* High contrast mode */
    @media (prefers-contrast: high) {
        :root {
            --primary-500: #0000ff;
            --secondary-900: #000000;
            --secondary-600: #333333;
        }
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'data_loaded' not in st.session_state:
    st.session_state.data_loaded = False
if 'processed_data' not in st.session_state:
    st.session_state.processed_data = None
if 'analysis_results' not in st.session_state:
    st.session_state.analysis_results = None
if 'model_results' not in st.session_state:
    st.session_state.model_results = None

@st.cache_data
def load_and_preprocess_data():
    """Load and preprocess data with caching"""
    try:
        preprocessor = DataPreprocessor()
        processed_data = preprocessor.preprocess_all()
        return processed_data, True
    except Exception as e:
        st.error(f"Error loading data: {e}")
        return None, False

@st.cache_data
def run_statistical_analysis(df):
    """Run statistical analysis with caching"""
    try:
        analyzer = StatisticalAnalyzer(df)
        results = analyzer.run_complete_analysis()
        return results, True
    except Exception as e:
        st.error(f"Error in statistical analysis: {e}")
        return None, False

@st.cache_data
def run_ml_modeling(df):
    """Run ML modeling with caching"""
    try:
        modeler = MLModeler(df)
        results = modeler.run_complete_modeling()
        return results, True
    except Exception as e:
        st.error(f"Error in ML modeling: {e}")
        return None, False

def main():
    """Main application function"""
    
    # Header
    st.markdown('<h1 class="main-header">AgriClimate Insight Portal</h1>', unsafe_allow_html=True)
    st.markdown("""
    <div class="info-box">
        <strong>Welcome to AgriClimate Insight Portal!</strong><br>
        This interactive dashboard provides comprehensive analysis of climate-agriculture relationships in India.
        Explore correlations, trends, and predictive models to understand how climate factors impact agricultural yields.
    </div>
    """, unsafe_allow_html=True)
    
    # Premium Sidebar
    st.sidebar.markdown("""
    <div style="text-align: center; margin-bottom: 2rem;">
        <h2 style="color: var(--secondary-800); font-weight: 700; margin: 0;">🎛️ Control Panel</h2>
        <p style="color: var(--secondary-600); font-size: 0.9rem; margin: 0.5rem 0 0 0;">Manage your analysis</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Data loading section with premium styling
    st.sidebar.markdown("""
    <div style="background: linear-gradient(135deg, var(--rainfall-50), rgba(255,255,255,0.8)); 
                padding: 1rem; border-radius: var(--radius-lg); margin-bottom: 1.5rem; 
                border: 1px solid var(--rainfall-200);">
        <h3 style="color: var(--rainfall-800); margin: 0 0 0.5rem 0; font-size: 1rem;">📊 Data Management</h3>
        <p style="color: var(--rainfall-700); font-size: 0.8rem; margin: 0;">Load and preprocess your dataset</p>
    </div>
    """, unsafe_allow_html=True)
    
    if st.sidebar.button("🚀 Load & Preprocess Data", type="primary", use_container_width=True):
        with st.spinner("🔄 Loading and preprocessing data..."):
            processed_data, success = load_and_preprocess_data()
            if success:
                st.session_state.processed_data = processed_data
                st.session_state.data_loaded = True
                st.sidebar.success("✅ Data loaded successfully!")
            else:
                st.sidebar.error("❌ Failed to load data")
    
    # Analysis controls with premium styling
    if st.session_state.data_loaded:
        st.sidebar.markdown("""
        <div style="background: linear-gradient(135deg, var(--crop-50), rgba(255,255,255,0.8)); 
                    padding: 1rem; border-radius: var(--radius-lg); margin-bottom: 1.5rem; 
                    border: 1px solid var(--crop-200);">
            <h3 style="color: var(--crop-800); margin: 0 0 0.5rem 0; font-size: 1rem;">🔬 Analysis Controls</h3>
            <p style="color: var(--crop-700); font-size: 0.8rem; margin: 0;">Run statistical and ML analysis</p>
        </div>
        """, unsafe_allow_html=True)
        
        col1, col2 = st.sidebar.columns(2)
        
        with col1:
            if st.button("📈 Stats", use_container_width=True):
                with st.spinner("🔄 Running statistical analysis..."):
                    results, success = run_statistical_analysis(st.session_state.processed_data)
                    if success:
                        st.session_state.analysis_results = results
                        st.sidebar.success("✅ Statistical analysis completed!")
        
        with col2:
            if st.button("🤖 ML", use_container_width=True):
                with st.spinner("🔄 Training ML models..."):
                    results, success = run_ml_modeling(st.session_state.processed_data)
                    if success:
                        st.session_state.model_results = results
                        st.sidebar.success("✅ ML modeling completed!")
        
        # Filters with premium styling
        st.sidebar.markdown("""
        <div style="background: linear-gradient(135deg, var(--temperature-50), rgba(255,255,255,0.8)); 
                    padding: 1rem; border-radius: var(--radius-lg); margin-bottom: 1.5rem; 
                    border: 1px solid var(--temperature-200);">
            <h3 style="color: var(--temperature-800); margin: 0 0 0.5rem 0; font-size: 1rem;">🎯 Filters</h3>
            <p style="color: var(--temperature-700); font-size: 0.8rem; margin: 0;">Refine your data view</p>
        </div>
        """, unsafe_allow_html=True)
        
        # State filter
        states = ['All'] + sorted(st.session_state.processed_data['State'].unique().tolist())
        selected_state = st.sidebar.selectbox("🏛️ Select State", states)
        
        # Crop filter
        crops = ['All'] + sorted(st.session_state.processed_data['Crop'].unique().tolist())
        selected_crop = st.sidebar.selectbox("🌾 Select Crop", crops)
        
        # Year range filter
        if 'Year' in st.session_state.processed_data.columns:
            years = st.session_state.processed_data['Year'].dt.year.unique()
            min_year, max_year = int(years.min()), int(years.max())
            year_range = st.sidebar.slider(
                "📅 Select Year Range",
                min_value=min_year,
                max_value=max_year,
                value=(min_year, max_year)
            )
        
        # Apply filters
        filtered_data = st.session_state.processed_data.copy()
        
        if selected_state != 'All':
            filtered_data = filtered_data[filtered_data['State'] == selected_state]
        
        if selected_crop != 'All':
            filtered_data = filtered_data[filtered_data['Crop'] == selected_crop]
        
        if 'Year' in filtered_data.columns:
            filtered_data = filtered_data[
                (filtered_data['Year'].dt.year >= year_range[0]) & 
                (filtered_data['Year'].dt.year <= year_range[1])
            ]
    
    else:
        st.markdown("""
        <div class="warning-box">
            <strong>No Data Loaded</strong><br>
            Please click "Load & Preprocess Data" in the sidebar to begin analysis.
        </div>
        """, unsafe_allow_html=True)
        return
    
    # Main content tabs with premium styling
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "📊 Overview", 
        "🔗 Correlations", 
        "🔮 Forecast", 
        "⚖️ Model Comparison", 
        "📋 Summary"
    ])
    
    with tab1:
        st.markdown('<h2 class="sub-header">Data Overview</h2>', unsafe_allow_html=True)
        
        # Key metrics with premium styling
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            delta_value = len(filtered_data) - len(st.session_state.processed_data)
            delta_class = "positive" if delta_value >= 0 else "negative"
            delta_symbol = "↗" if delta_value >= 0 else "↘"
            
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-title">Total Records</div>
                <div class="metric-value">{len(filtered_data):,}</div>
                <div class="metric-delta {delta_class}">
                    {delta_symbol} {abs(delta_value):,} vs total
                </div>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            if 'Yield' in filtered_data.columns:
                avg_yield = filtered_data['Yield'].mean()
                baseline_yield = st.session_state.processed_data['Yield'].mean()
                delta_value = avg_yield - baseline_yield
                delta_class = "positive" if delta_value >= 0 else "negative"
                delta_symbol = "↗" if delta_value >= 0 else "↘"
                
                st.markdown(f"""
                <div class="metric-card">
                    <div class="metric-title">Average Yield</div>
                    <div class="metric-value">{avg_yield:.1f}</div>
                    <div class="metric-delta {delta_class}">
                        {delta_symbol} {abs(delta_value):.1f} kg/ha
                    </div>
                </div>
                """, unsafe_allow_html=True)
        
        with col3:
            if 'Rainfall' in filtered_data.columns:
                avg_rainfall = filtered_data['Rainfall'].mean()
                baseline_rainfall = st.session_state.processed_data['Rainfall'].mean()
                delta_value = avg_rainfall - baseline_rainfall
                delta_class = "positive" if delta_value >= 0 else "negative"
                delta_symbol = "↗" if delta_value >= 0 else "↘"
                
                st.markdown(f"""
                <div class="metric-card">
                    <div class="metric-title">Average Rainfall</div>
                    <div class="metric-value">{avg_rainfall:.1f}</div>
                    <div class="metric-delta {delta_class}">
                        {delta_symbol} {abs(delta_value):.1f} mm
                    </div>
                </div>
                """, unsafe_allow_html=True)
        
        with col4:
            if 'Temperature' in filtered_data.columns:
                avg_temp = filtered_data['Temperature'].mean()
                baseline_temp = st.session_state.processed_data['Temperature'].mean()
                delta_value = avg_temp - baseline_temp
                delta_class = "positive" if delta_value >= 0 else "negative"
                delta_symbol = "↗" if delta_value >= 0 else "↘"
                
                st.markdown(f"""
                <div class="metric-card">
                    <div class="metric-title">Average Temperature</div>
                    <div class="metric-value">{avg_temp:.1f}</div>
                    <div class="metric-delta {delta_class}">
                        {delta_symbol} {abs(delta_value):.1f}°C
                    </div>
                </div>
                """, unsafe_allow_html=True)
        
        # Time series plot
        st.markdown('<h3 class="sub-header">Time Series Analysis</h3>', unsafe_allow_html=True)
        
        if 'Year' in filtered_data.columns:
            visualizer = DataVisualizer(filtered_data)
            
            # Yield trends
            if 'Yield' in filtered_data.columns:
                yield_fig = visualizer.create_line_chart(
                    'Year', 'Yield', 'State',
                    'Crop Yield Trends by State'
                )
                st.plotly_chart(yield_fig, use_container_width=True)
            
            # Climate trends
            col1, col2 = st.columns(2)
            
            with col1:
                if 'Rainfall' in filtered_data.columns:
                    rainfall_fig = visualizer.create_line_chart(
                        'Year', 'Rainfall', 'State',
                        'Rainfall Trends by State'
                    )
                    st.plotly_chart(rainfall_fig, use_container_width=True)
            
            with col2:
                if 'Temperature' in filtered_data.columns:
                    temp_fig = visualizer.create_line_chart(
                        'Year', 'Temperature', 'State',
                        'Temperature Trends by State'
                    )
                    st.plotly_chart(temp_fig, use_container_width=True)
        
        # Distribution plots
        st.markdown('<h3 class="sub-header">Data Distributions</h3>', unsafe_allow_html=True)
        
        col1, col2 = st.columns(2)
        
        with col1:
            if 'Yield' in filtered_data.columns:
                yield_hist = visualizer.create_histogram('Yield', title="Yield Distribution")
                st.plotly_chart(yield_hist, use_container_width=True)
        
        with col2:
            if 'Rainfall' in filtered_data.columns:
                rainfall_hist = visualizer.create_histogram('Rainfall', title="Rainfall Distribution")
                st.plotly_chart(rainfall_hist, use_container_width=True)
    
    with tab2:
        st.markdown('<h2 class="sub-header">Correlation Analysis</h2>', unsafe_allow_html=True)
        
        if st.session_state.analysis_results:
            # Correlation heatmap
            corr_results = st.session_state.analysis_results.get('correlations', {})
            
            if 'overall' in corr_results:
                st.markdown("### Correlation Matrix")
                
                # Create correlation heatmap
                corr_matrix = corr_results['overall']
                fig = go.Figure(data=go.Heatmap(
                    z=corr_matrix.values,
                    x=corr_matrix.columns,
                    y=corr_matrix.columns,
                    colorscale='RdBu',
                    zmid=0,
                    text=np.round(corr_matrix.values, 2),
                    texttemplate="%{text}",
                    textfont={"size": 10}
                ))
                
                fig.update_layout(
                    title="Correlation Matrix",
                    template='plotly_white',
                    height=600
                )
                
                st.plotly_chart(fig, use_container_width=True)
            
            # Climate-Yield correlations
            if 'climate_yield' in corr_results:
                st.markdown("### Climate-Yield Correlations")
                
                climate_corr = corr_results['climate_yield']
                corr_df = pd.DataFrame(list(climate_corr.items()), columns=['Variable', 'Correlation'])
                corr_df['Strength'] = corr_df['Correlation'].apply(
                    lambda x: 'Strong' if abs(x) > 0.7 else 'Moderate' if abs(x) > 0.3 else 'Weak'
                )
                
                st.dataframe(corr_df, use_container_width=True)
                
                # Correlation bar chart
                fig = px.bar(
                    corr_df,
                    x='Variable',
                    y='Correlation',
                    color='Correlation',
                    color_continuous_scale='RdBu',
                    title="Climate Variables vs Yield Correlation"
                )
                st.plotly_chart(fig, use_container_width=True)
        
        else:
            st.info("Run statistical analysis to see correlation results.")
    
    with tab3:
        st.markdown('<h2 class="sub-header">Forecasting</h2>', unsafe_allow_html=True)
        
        if st.session_state.model_results:
            # Time series forecasts
            if 'arima' in st.session_state.model_results:
                st.markdown("### ARIMA Forecast")
                
                arima_results = st.session_state.model_results['arima']
                if 'error' not in arima_results:
                    # Create forecast plot
                    fig = go.Figure()
                    
                    # Historical data
                    fig.add_trace(go.Scatter(
                        x=list(range(len(arima_results['fitted_values']))),
                        y=arima_results['fitted_values'],
                        mode='lines',
                        name='Fitted Values',
                        line=dict(color='blue')
                    ))
                    
                    # Forecast
                    forecast_steps = len(arima_results['forecast'])
                    forecast_x = list(range(len(arima_results['fitted_values']), 
                                          len(arima_results['fitted_values']) + forecast_steps))
                    fig.add_trace(go.Scatter(
                        x=forecast_x,
                        y=arima_results['forecast'],
                        mode='lines',
                        name='Forecast',
                        line=dict(color='red', dash='dash')
                    ))
                    
                    fig.update_layout(
                        title="ARIMA Forecast",
                        xaxis_title="Time",
                        yaxis_title="Yield",
                        template='plotly_white'
                    )
                    
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Model metrics
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("R²", f"{arima_results['r2']:.3f}")
                    with col2:
                        st.metric("RMSE", f"{arima_results['rmse']:.3f}")
                    with col3:
                        st.metric("MAE", f"{arima_results['mae']:.3f}")
            
            # Prophet forecast
            if 'prophet' in st.session_state.model_results:
                st.markdown("### Prophet Forecast")
                
                prophet_results = st.session_state.model_results['prophet']
                if 'error' not in prophet_results:
                    st.info("Prophet forecast visualization would be displayed here.")
                    
                    # Model metrics
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("R²", f"{prophet_results['r2']:.3f}")
                    with col2:
                        st.metric("RMSE", f"{prophet_results['rmse']:.3f}")
                    with col3:
                        st.metric("MAE", f"{prophet_results['mae']:.3f}")
        
        else:
            st.info("Run ML modeling to see forecasting results.")
    
    with tab4:
        st.markdown('<h2 class="sub-header">Model Comparison</h2>', unsafe_allow_html=True)
        
        if st.session_state.model_results and 'model_comparison' in st.session_state.model_results:
            comparison_df = st.session_state.model_results['model_comparison']
            
            # Model comparison table
            st.markdown("### Model Performance Comparison")
            st.dataframe(comparison_df, use_container_width=True)
            
            # Performance visualization
            fig = go.Figure()
            
            fig.add_trace(go.Bar(
                name='R² Score',
                x=comparison_df['Model'],
                y=comparison_df['R²'],
                marker_color='lightblue'
            ))
            
            fig.add_trace(go.Bar(
                name='RMSE',
                x=comparison_df['Model'],
                y=comparison_df['RMSE'],
                marker_color='lightcoral',
                yaxis='y2'
            ))
            
            fig.update_layout(
                title="Model Performance Comparison",
                xaxis_title="Model",
                yaxis=dict(title="R² Score", side="left"),
                yaxis2=dict(title="RMSE", side="right", overlaying="y"),
                template='plotly_white',
                height=500
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Feature importance
            if 'random_forest' in st.session_state.model_results:
                st.markdown("### Feature Importance (Random Forest)")
                
                rf_results = st.session_state.model_results['random_forest']
                if 'feature_importance' in rf_results:
                    importance_dict = rf_results['feature_importance']
                    
                    # Create feature importance plot
                    features = list(importance_dict.keys())
                    importance_values = list(importance_dict.values())
                    
                    fig = go.Figure(data=[
                        go.Bar(
                            x=importance_values,
                            y=features,
                            orientation='h',
                            marker=dict(color='#3b82f6')
                        )
                    ])
                    
                    fig.update_layout(
                        title="Feature Importance",
                        xaxis_title="Importance",
                        yaxis_title="Features",
                        template='plotly_white',
                        height=400
                    )
                    
                    st.plotly_chart(fig, use_container_width=True)
        
        else:
            st.info("Run ML modeling to see model comparison results.")
    
    with tab5:
        st.markdown('<h2 class="sub-header">Analysis Summary</h2>', unsafe_allow_html=True)
        
        # Data summary
        st.markdown("### Dataset Summary")
        
        summary_data = {
            'Metric': ['Total Records', 'States', 'Crops', 'Years', 'Features'],
            'Value': [
                len(st.session_state.processed_data),
                st.session_state.processed_data['State'].nunique(),
                st.session_state.processed_data['Crop'].nunique(),
                st.session_state.processed_data['Year'].nunique() if 'Year' in st.session_state.processed_data.columns else 'N/A',
                len(st.session_state.processed_data.columns)
            ]
        }
        
        summary_df = pd.DataFrame(summary_data)
        st.dataframe(summary_df, use_container_width=True)
        
        # Analysis insights
        if st.session_state.analysis_results:
            st.markdown("### Key Insights")
            
            # Get analysis summary
            analyzer = StatisticalAnalyzer(st.session_state.processed_data)
            insights = analyzer.get_analysis_summary()
            
            for key, value in insights.items():
                st.markdown(f"**{key.replace('_', ' ').title()}:** {value}")
        
        # Model insights
        if st.session_state.model_results:
            st.markdown("### Model Insights")
            
            # Get model summary
            modeler = MLModeler(st.session_state.processed_data)
            model_insights = modeler.get_model_summary()
            
            for key, value in model_insights.items():
                st.markdown(f"**{key.replace('_', ' ').title()}:** {value}")
        
        # Download section
        st.markdown("### Download Results")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            if st.button("Download Processed Data"):
                csv = st.session_state.processed_data.to_csv(index=False)
                st.download_button(
                    label="Download CSV",
                    data=csv,
                    file_name="processed_agriculture_data.csv",
                    mime="text/csv"
                )
        
        with col2:
            if st.button("Download Analysis Results"):
                if st.session_state.analysis_results:
                    # Convert results to JSON for download
                    import json
                    results_json = json.dumps(st.session_state.analysis_results, default=str)
                    st.download_button(
                        label="Download JSON",
                        data=results_json,
                        file_name="analysis_results.json",
                        mime="application/json"
                    )
        
        with col3:
            if st.button("Download Model Results"):
                if st.session_state.model_results:
                    import json
                    model_json = json.dumps(st.session_state.model_results, default=str)
                    st.download_button(
                        label="Download JSON",
                        data=model_json,
                        file_name="model_results.json",
                        mime="application/json"
                    )
    
    # Premium Footer
    st.markdown("---")
    st.markdown("""
    <div style="text-align: center; margin-top: 3rem; padding: 2rem; 
                background: linear-gradient(135deg, var(--secondary-50), rgba(255,255,255,0.8));
                border-radius: var(--radius-xl); border: 1px solid var(--secondary-200);">
        <div style="display: flex; justify-content: center; align-items: center; margin-bottom: 1rem;">
            <div style="width: 40px; height: 40px; background: linear-gradient(135deg, var(--primary-500), var(--primary-600)); 
                        border-radius: 50%; display: flex; align-items: center; justify-content: center; margin-right: 1rem;">
                <span style="color: white; font-size: 1.2rem;">📊</span>
            </div>
            <h3 style="color: var(--secondary-800); margin: 0; font-weight: 700;">AgriClimate Insight Portal</h3>
        </div>
        <p style="color: var(--secondary-600); font-size: 0.9rem; margin: 0.5rem 0;">
            Built with <span style="color: var(--primary-500); font-weight: 600;">Streamlit</span> & 
            <span style="color: var(--crop-500); font-weight: 600;">Python</span>
        </p>
        <p style="color: var(--secondary-500); font-size: 0.8rem; margin: 0;">
            Analyzing climate-agriculture relationships for sustainable farming 🌱
        </p>
        <div style="margin-top: 1rem; padding-top: 1rem; border-top: 1px solid var(--secondary-200);">
            <p style="color: var(--secondary-400); font-size: 0.75rem; margin: 0;">
                © 2024 AgriClimate Insight Portal | Premium Analytics Dashboard
            </p>
        </div>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
