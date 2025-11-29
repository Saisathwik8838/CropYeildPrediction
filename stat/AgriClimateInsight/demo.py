#!/usr/bin/env python3
"""
AgriClimate Insight Portal - Quick Demo Script
Demonstrates key features of the portal
"""

import sys
import os
import pandas as pd
import numpy as np

# Add utils to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'utils'))

def run_demo():
    """Run a quick demo of the portal features"""
    print("🌾 AgriClimate Insight Portal - Quick Demo")
    print("=" * 50)
    
    # Import modules
    from utils.data_preprocessing import DataPreprocessor
    from utils.analysis import StatisticalAnalyzer
    from utils.modeling import MLModeler
    from utils.visualization import DataVisualizer
    from utils.helpers import validate_data_format, get_data_quality_score
    
    print("\n1️⃣ Data Preprocessing Demo")
    print("-" * 30)
    
    # Load and preprocess data
    preprocessor = DataPreprocessor()
    processed_data = preprocessor.preprocess_all()
    
    print(f"✅ Loaded dataset: {processed_data.shape}")
    print(f"   Columns: {list(processed_data.columns)}")
    print(f"   States: {processed_data['State'].nunique()}")
    print(f"   Crops: {processed_data['Crop'].nunique()}")
    
    # Data quality check
    quality = get_data_quality_score(processed_data)
    print(f"   Data Quality Score: {quality['overall_score']:.2f}")
    
    print("\n2️⃣ Statistical Analysis Demo")
    print("-" * 30)
    
    # Run statistical analysis
    analyzer = StatisticalAnalyzer(processed_data)
    analysis_results = analyzer.run_complete_analysis()
    
    # Show key insights
    summary = analyzer.get_analysis_summary()
    print("📊 Key Insights:")
    for key, value in summary.items():
        print(f"   • {key.replace('_', ' ').title()}: {value}")
    
    print("\n3️⃣ Machine Learning Demo")
    print("-" * 30)
    
    # Run ML modeling
    modeler = MLModeler(processed_data)
    model_results = modeler.run_complete_modeling()
    
    # Show model comparison
    if 'model_comparison' in model_results:
        comparison = model_results['model_comparison']
        print("🤖 Model Performance:")
        for _, row in comparison.iterrows():
            print(f"   • {row['Model']}: R² = {row['R²']:.3f}, RMSE = {row['RMSE']:.3f}")
    
    # Show feature importance
    if 'xgboost' in model_results and 'feature_importance' in model_results['xgboost']:
        importance = model_results['xgboost']['feature_importance']
        top_features = sorted(importance.items(), key=lambda x: x[1], reverse=True)[:3]
        print("🔍 Top Features:")
        for feature, imp in top_features:
            print(f"   • {feature}: {imp:.3f}")
    
    print("\n4️⃣ Visualization Demo")
    print("-" * 30)
    
    # Create visualizations
    visualizer = DataVisualizer(processed_data)
    
    # Show available visualizations
    viz_summary = visualizer.get_visualization_summary()
    print("📊 Available Visualizations:")
    for key, value in viz_summary.items():
        print(f"   • {key.replace('_', ' ').title()}: {value}")
    
    print("\n5️⃣ Sample Data Preview")
    print("-" * 30)
    
    # Show sample data
    print("📋 Sample Data (first 5 rows):")
    sample_cols = ['State', 'Crop', 'Year', 'Yield', 'Rainfall', 'Temperature']
    available_cols = [col for col in sample_cols if col in processed_data.columns]
    print(processed_data[available_cols].head().to_string(index=False))
    
    print("\n🎉 Demo Complete!")
    print("=" * 50)
    print("\n📝 Next Steps:")
    print("   1. Run 'streamlit run app.py' to start the web application")
    print("   2. Use the interactive dashboard to explore your data")
    print("   3. Upload your own CSV files to the data/ directory")
    print("   4. Customize the analysis parameters in the utils/ modules")
    
    print("\n🔗 Useful Commands:")
    print("   • Start app: streamlit run app.py")
    print("   • Run tests: python test_app.py")
    print("   • Quick start: ./start.sh")
    
    return processed_data, analysis_results, model_results

if __name__ == "__main__":
    try:
        processed_data, analysis_results, model_results = run_demo()
        print(f"\n✅ Demo completed successfully!")
        print(f"   Processed {len(processed_data)} records")
        print(f"   Completed {len(analysis_results)} analyses")
        print(f"   Trained {len(model_results)} models")
    except Exception as e:
        print(f"\n❌ Demo failed: {e}")
        print("   Please check your Python environment and dependencies")
        sys.exit(1)
