#!/usr/bin/env python3
"""
Test script for AgriClimate Insight Portal
Verifies that all modules work correctly
"""

import sys
import os
import pandas as pd
import numpy as np

# Add utils to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'utils'))

def test_data_preprocessing():
    """Test data preprocessing module"""
    print("🧪 Testing Data Preprocessing...")
    
    try:
        from utils.data_preprocessing import DataPreprocessor
        
        preprocessor = DataPreprocessor()
        processed_data = preprocessor.preprocess_all()
        
        print(f"✅ Data preprocessing successful: {processed_data.shape}")
        return processed_data
        
    except Exception as e:
        print(f"❌ Data preprocessing failed: {e}")
        return None

def test_statistical_analysis(df):
    """Test statistical analysis module"""
    print("🧪 Testing Statistical Analysis...")
    
    try:
        from utils.analysis import StatisticalAnalyzer
        
        analyzer = StatisticalAnalyzer(df)
        results = analyzer.run_complete_analysis()
        
        print(f"✅ Statistical analysis successful: {len(results)} analyses completed")
        return results
        
    except Exception as e:
        print(f"❌ Statistical analysis failed: {e}")
        return None

def test_ml_modeling(df):
    """Test ML modeling module"""
    print("🧪 Testing ML Modeling...")
    
    try:
        from utils.modeling import MLModeler
        
        modeler = MLModeler(df)
        results = modeler.run_complete_modeling()
        
        print(f"✅ ML modeling successful: {len(results)} models trained")
        return results
        
    except Exception as e:
        print(f"❌ ML modeling failed: {e}")
        return None

def test_visualization(df):
    """Test visualization module"""
    print("🧪 Testing Visualization...")
    
    try:
        from utils.visualization import DataVisualizer
        
        visualizer = DataVisualizer(df)
        
        # Test different visualizations
        if 'Year' in df.columns and 'Yield' in df.columns:
            line_fig = visualizer.create_line_chart('Year', 'Yield', 'State')
            print("✅ Line chart created")
        
        numerical_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        if len(numerical_cols) > 1:
            corr_fig = visualizer.create_correlation_heatmap()
            print("✅ Correlation heatmap created")
        
        if 'Yield' in df.columns:
            hist_fig = visualizer.create_histogram('Yield')
            print("✅ Histogram created")
        
        print("✅ Visualization module working correctly")
        return True
        
    except Exception as e:
        print(f"❌ Visualization failed: {e}")
        return False

def test_helper_utilities():
    """Test helper utilities"""
    print("🧪 Testing Helper Utilities...")
    
    try:
        from utils.helpers import validate_data_format, get_data_quality_score, INDIAN_STATES, MAJOR_CROPS
        
        # Test with sample data
        test_df = pd.DataFrame({
            'State': ['Andhra Pradesh', 'Bihar', 'Gujarat'],
            'Crop': ['Rice', 'Wheat', 'Maize'],
            'Year': [2020, 2021, 2022],
            'Yield': [2500, 1800, 2200]
        })
        
        validation = validate_data_format(test_df)
        quality = get_data_quality_score(test_df)
        
        print(f"✅ Data validation: {validation['is_valid']}")
        print(f"✅ Data quality score: {quality['overall_score']:.2f}")
        print(f"✅ Constants loaded: {len(INDIAN_STATES)} states, {len(MAJOR_CROPS)} crops")
        
        return True
        
    except Exception as e:
        print(f"❌ Helper utilities failed: {e}")
        return False

def main():
    """Main test function"""
    print("🚀 Starting AgriClimate Insight Portal Tests...")
    print("=" * 60)
    
    # Test data preprocessing
    processed_data = test_data_preprocessing()
    if processed_data is None:
        print("❌ Cannot continue without processed data")
        return
    
    print()
    
    # Test statistical analysis
    analysis_results = test_statistical_analysis(processed_data)
    print()
    
    # Test ML modeling
    model_results = test_ml_modeling(processed_data)
    print()
    
    # Test visualization
    viz_success = test_visualization(processed_data)
    print()
    
    # Test helper utilities
    helpers_success = test_helper_utilities()
    print()
    
    # Summary
    print("=" * 60)
    print("📊 Test Summary:")
    print(f"   Data Preprocessing: {'✅' if processed_data is not None else '❌'}")
    print(f"   Statistical Analysis: {'✅' if analysis_results is not None else '❌'}")
    print(f"   ML Modeling: {'✅' if model_results is not None else '❌'}")
    print(f"   Visualization: {'✅' if viz_success else '❌'}")
    print(f"   Helper Utilities: {'✅' if helpers_success else '❌'}")
    
    all_tests_passed = all([
        processed_data is not None,
        analysis_results is not None,
        model_results is not None,
        viz_success,
        helpers_success
    ])
    
    if all_tests_passed:
        print("\n🎉 All tests passed! The AgriClimate Insight Portal is ready to use.")
        print("\n📝 Next steps:")
        print("   1. Run 'streamlit run app.py' to start the web application")
        print("   2. Place your CSV files in the data/ directory (optional)")
        print("   3. Explore the interactive dashboard!")
    else:
        print("\n⚠️ Some tests failed. Please check the error messages above.")
    
    print("\n" + "=" * 60)

if __name__ == "__main__":
    main()
