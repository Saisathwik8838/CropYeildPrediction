"""
Helper Utilities Module for AgriClimate Insight Portal
Contains utility functions and constants used across the application
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional
import warnings
warnings.filterwarnings('ignore')

# Constants
INDIAN_STATES = [
    'Andhra Pradesh', 'Arunachal Pradesh', 'Assam', 'Bihar', 'Chhattisgarh',
    'Goa', 'Gujarat', 'Haryana', 'Himachal Pradesh', 'Jharkhand',
    'Karnataka', 'Kerala', 'Madhya Pradesh', 'Maharashtra', 'Manipur',
    'Meghalaya', 'Mizoram', 'Nagaland', 'Odisha', 'Punjab',
    'Rajasthan', 'Sikkim', 'Tamil Nadu', 'Telangana', 'Tripura',
    'Uttar Pradesh', 'Uttarakhand', 'West Bengal', 'Delhi', 'Puducherry'
]

MAJOR_CROPS = [
    'Rice', 'Wheat', 'Maize', 'Sugarcane', 'Cotton', 'Soyabean',
    'Groundnuts', 'Barley', 'Potatoes', 'Tomatoes', 'Onions', 'Bananas',
    'Mangoes', 'Grapes', 'Apples', 'Oranges', 'Coffee', 'Tea',
    'Coconuts', 'Cashewnuts', 'Pepper', 'Ginger', 'Garlic', 'Chilli Dry',
    'Chilli Green', 'Eggplant', 'Cabbage', 'Cauliflower', 'Carrots',
    'Cucumber', 'Okra', 'Pumpkins', 'Watermelon', 'Melons', 'Papaya',
    'Pineapples', 'Peaches', 'Pears', 'Cherries', 'Apricots', 'Lemons',
    'Lettuce', 'Peas', 'Beans Dry', 'Beans Green', 'Lentils', 'Sesame',
    'Safflower', 'Areca Nuts', 'Nutmeg', 'Fennel Coriander'
]

CLIMATE_VARIABLES = ['Rainfall', 'Temperature', 'Humidity', 'Wind_Speed']

YIELD_UNITS = {
    'Rice': 'kg/ha', 'Wheat': 'kg/ha', 'Maize': 'kg/ha', 'Sugarcane': 'tonnes/ha',
    'Cotton': 'kg/ha', 'Soyabean': 'kg/ha', 'Groundnuts': 'kg/ha', 'Barley': 'kg/ha',
    'Potatoes': 'kg/ha', 'Tomatoes': 'kg/ha', 'Onions': 'kg/ha', 'Bananas': 'kg/ha',
    'Mangoes': 'kg/ha', 'Grapes': 'kg/ha', 'Apples': 'kg/ha', 'Oranges': 'kg/ha',
    'Coffee': 'kg/ha', 'Tea': 'kg/ha', 'Coconuts': 'nuts/ha', 'Cashewnuts': 'kg/ha',
    'Pepper': 'kg/ha', 'Ginger': 'kg/ha', 'Garlic': 'kg/ha', 'Chilli Dry': 'kg/ha',
    'Chilli Green': 'kg/ha', 'Eggplant': 'kg/ha', 'Cabbage': 'kg/ha', 'Cauliflower': 'kg/ha',
    'Carrots': 'kg/ha', 'Cucumber': 'kg/ha', 'Okra': 'kg/ha', 'Pumpkins': 'kg/ha',
    'Watermelon': 'kg/ha', 'Melons': 'kg/ha', 'Papaya': 'kg/ha', 'Pineapples': 'kg/ha',
    'Peaches': 'kg/ha', 'Pears': 'kg/ha', 'Cherries': 'kg/ha', 'Apricots': 'kg/ha',
    'Lemons': 'kg/ha', 'Lettuce': 'kg/ha', 'Peas': 'kg/ha', 'Beans Dry': 'kg/ha',
    'Beans Green': 'kg/ha', 'Lentils': 'kg/ha', 'Sesame': 'kg/ha', 'Safflower': 'kg/ha',
    'Areca Nuts': 'nuts/ha', 'Nutmeg': 'kg/ha', 'Fennel Coriander': 'kg/ha'
}

# Color schemes
COLOR_PALETTES = {
    'primary': ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd'],
    'climate': ['#2E8B57', '#228B22', '#32CD32', '#90EE90', '#98FB98'],
    'agriculture': ['#8B4513', '#D2691E', '#CD853F', '#DEB887', '#F5DEB3'],
    'correlation': ['#B22222', '#FF6347', '#FFA500', '#FFD700', '#90EE90', '#00CED1', '#4169E1']
}

def validate_data_format(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Validate that the DataFrame has the expected format
    
    Args:
        df: DataFrame to validate
        
    Returns:
        Dictionary with validation results
    """
    validation_results = {
        'is_valid': True,
        'errors': [],
        'warnings': [],
        'missing_columns': [],
        'data_types': {}
    }
    
    # Required columns
    required_columns = ['State', 'Crop', 'Year']
    optional_columns = ['Yield', 'Rainfall', 'Temperature', 'Area', 'Production']
    
    # Check required columns
    for col in required_columns:
        if col not in df.columns:
            validation_results['errors'].append(f"Missing required column: {col}")
            validation_results['missing_columns'].append(col)
            validation_results['is_valid'] = False
    
    # Check data types
    for col in df.columns:
        validation_results['data_types'][col] = str(df[col].dtype)
    
    # Check for reasonable data ranges
    if 'Year' in df.columns:
        years = df['Year'].dropna()
        if len(years) > 0:
            min_year, max_year = years.min(), years.max()
            if min_year < 1900 or max_year > 2030:
                validation_results['warnings'].append(f"Year range ({min_year}-{max_year}) seems unusual")
    
    if 'Yield' in df.columns:
        yields = df['Yield'].dropna()
        if len(yields) > 0:
            if yields.min() < 0 or yields.max() > 10000:
                validation_results['warnings'].append("Yield values seem outside normal range (0-10000 kg/ha)")
    
    return validation_results

def clean_state_names(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean and standardize state names
    
    Args:
        df: DataFrame with State column
        
    Returns:
        DataFrame with cleaned state names
    """
    df_clean = df.copy()
    
    if 'State' in df_clean.columns:
        # Common state name variations
        state_mappings = {
            'AP': 'Andhra Pradesh',
            'UP': 'Uttar Pradesh',
            'MP': 'Madhya Pradesh',
            'TN': 'Tamil Nadu',
            'WB': 'West Bengal',
            'MH': 'Maharashtra',
            'GJ': 'Gujarat',
            'RJ': 'Rajasthan',
            'KA': 'Karnataka',
            'KL': 'Kerala',
            'OR': 'Odisha',
            'AS': 'Assam',
            'BR': 'Bihar',
            'PB': 'Punjab',
            'HR': 'Haryana',
            'JK': 'Jammu and Kashmir',
            'HP': 'Himachal Pradesh',
            'UK': 'Uttarakhand',
            'JH': 'Jharkhand',
            'CT': 'Chhattisgarh',
            'GA': 'Goa',
            'DL': 'Delhi',
            'PY': 'Puducherry'
        }
        
        df_clean['State'] = df_clean['State'].replace(state_mappings)
        
        # Standardize case
        df_clean['State'] = df_clean['State'].str.title()
    
    return df_clean

def clean_crop_names(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean and standardize crop names
    
    Args:
        df: DataFrame with Crop column
        
    Returns:
        DataFrame with cleaned crop names
    """
    df_clean = df.copy()
    
    if 'Crop' in df_clean.columns:
        # Common crop name variations
        crop_mappings = {
            'Paddy': 'Rice',
            'Paddy Rice': 'Rice',
            'Wheat (Triticum aestivum)': 'Wheat',
            'Maize (Zea mays)': 'Maize',
            'Corn': 'Maize',
            'Sugarcane (Saccharum officinarum)': 'Sugarcane',
            'Cotton (Gossypium)': 'Cotton',
            'Soybean (Glycine max)': 'Soybean',
            'Groundnut (Arachis hypogaea)': 'Groundnut',
            'Peanut': 'Groundnut',
            'Rapeseed & Mustard': 'Rapeseed',
            'Mustard': 'Rapeseed',
            'Sunflower (Helianthus annuus)': 'Sunflower',
            'Jute (Corchorus)': 'Jute',
            'Tea (Camellia sinensis)': 'Tea',
            'Coffee (Coffea)': 'Coffee',
            'Rubber (Hevea brasiliensis)': 'Rubber'
        }
        
        df_clean['Crop'] = df_clean['Crop'].replace(crop_mappings)
        
        # Standardize case
        df_clean['Crop'] = df_clean['Crop'].str.title()
    
    return df_clean

def calculate_climate_indices(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate additional climate indices
    
    Args:
        df: DataFrame with climate data
        
    Returns:
        DataFrame with additional climate indices
    """
    df_indices = df.copy()
    
    # Temperature indices
    if 'Temperature' in df_indices.columns:
        df_indices['Temp_Anomaly'] = df_indices['Temperature'] - df_indices['Temperature'].mean()
        df_indices['Temp_Category'] = pd.cut(
            df_indices['Temperature'],
            bins=[-np.inf, 20, 25, 30, np.inf],
            labels=['Cold', 'Moderate', 'Warm', 'Hot']
        )
    
    # Rainfall indices
    if 'Rainfall' in df_indices.columns:
        df_indices['Rainfall_Anomaly'] = df_indices['Rainfall'] - df_indices['Rainfall'].mean()
        df_indices['Rainfall_Category'] = pd.cut(
            df_indices['Rainfall'],
            bins=[-np.inf, 500, 1000, 1500, np.inf],
            labels=['Dry', 'Moderate', 'Wet', 'Very Wet']
        )
    
    # Combined climate index
    if 'Temperature' in df_indices.columns and 'Rainfall' in df_indices.columns:
        # Normalize temperature and rainfall for combined index
        temp_norm = (df_indices['Temperature'] - df_indices['Temperature'].min()) / \
                   (df_indices['Temperature'].max() - df_indices['Temperature'].min())
        rain_norm = (df_indices['Rainfall'] - df_indices['Rainfall'].min()) / \
                   (df_indices['Rainfall'].max() - df_indices['Rainfall'].min())
        
        # Climate suitability index (higher is better for most crops)
        df_indices['Climate_Suitability'] = 0.6 * rain_norm + 0.4 * (1 - temp_norm)
    
    return df_indices

def detect_outliers(df: pd.DataFrame, columns: List[str], method: str = 'iqr') -> Dict[str, List[int]]:
    """
    Detect outliers in specified columns
    
    Args:
        df: DataFrame to analyze
        columns: List of columns to check for outliers
        method: Method for outlier detection ('iqr' or 'zscore')
        
    Returns:
        Dictionary with outlier indices for each column
    """
    outliers = {}
    
    for col in columns:
        if col in df.columns:
            data = df[col].dropna()
            
            if method == 'iqr':
                Q1 = data.quantile(0.25)
                Q3 = data.quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - 1.5 * IQR
                upper_bound = Q3 + 1.5 * IQR
                outlier_mask = (data < lower_bound) | (data > upper_bound)
            
            elif method == 'zscore':
                z_scores = np.abs((data - data.mean()) / data.std())
                outlier_mask = z_scores > 3
            
            else:
                raise ValueError("Method must be 'iqr' or 'zscore'")
            
            outliers[col] = data[outlier_mask].index.tolist()
    
    return outliers

def calculate_growth_rates(df: pd.DataFrame, value_col: str, time_col: str = 'Year') -> pd.DataFrame:
    """
    Calculate growth rates for time series data
    
    Args:
        df: DataFrame with time series data
        value_col: Column containing values to calculate growth for
        time_col: Column containing time information
        
    Returns:
        DataFrame with growth rate calculations
    """
    df_growth = df.copy()
    
    if value_col in df_growth.columns and time_col in df_growth.columns:
        # Sort by time
        df_growth = df_growth.sort_values(time_col)
        
        # Calculate year-over-year growth rate
        df_growth[f'{value_col}_Growth_Rate'] = df_growth[value_col].pct_change() * 100
        
        # Calculate compound annual growth rate (CAGR)
        if len(df_growth) > 1:
            first_value = df_growth[value_col].iloc[0]
            last_value = df_growth[value_col].iloc[-1]
            years = len(df_growth)
            
            if first_value > 0 and last_value > 0:
                cagr = ((last_value / first_value) ** (1 / years) - 1) * 100
                df_growth[f'{value_col}_CAGR'] = cagr
    
    return df_growth

def create_summary_table(df: pd.DataFrame, group_cols: List[str], 
                        value_cols: List[str]) -> pd.DataFrame:
    """
    Create summary statistics table
    
    Args:
        df: DataFrame to summarize
        group_cols: Columns to group by
        value_cols: Columns to calculate statistics for
        
    Returns:
        Summary DataFrame
    """
    summary_stats = []
    
    for group_col in group_cols:
        if group_col in df.columns:
            for value_col in value_cols:
                if value_col in df.columns:
                    group_stats = df.groupby(group_col)[value_col].agg([
                        'count', 'mean', 'std', 'min', 'max', 'median'
                    ]).round(2)
                    
                    group_stats['Group_Column'] = group_col
                    group_stats['Value_Column'] = value_col
                    group_stats = group_stats.reset_index()
                    
                    summary_stats.append(group_stats)
    
    if summary_stats:
        return pd.concat(summary_stats, ignore_index=True)
    else:
        return pd.DataFrame()

def export_results(results: Dict[str, Any], filename: str, format: str = 'json') -> bool:
    """
    Export analysis results to file
    
    Args:
        results: Dictionary containing results
        filename: Output filename
        format: Export format ('json', 'csv', 'pickle')
        
    Returns:
        Boolean indicating success
    """
    try:
        if format == 'json':
            import json
            with open(filename, 'w') as f:
                json.dump(results, f, default=str, indent=2)
        
        elif format == 'csv':
            # Convert results to DataFrame if possible
            if isinstance(results, dict):
                df = pd.DataFrame([results])
                df.to_csv(filename, index=False)
            else:
                return False
        
        elif format == 'pickle':
            import pickle
            with open(filename, 'wb') as f:
                pickle.dump(results, f)
        
        else:
            raise ValueError("Format must be 'json', 'csv', or 'pickle'")
        
        return True
    
    except Exception as e:
        print(f"Error exporting results: {e}")
        return False

def get_data_quality_score(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Calculate data quality score
    
    Args:
        df: DataFrame to evaluate
        
    Returns:
        Dictionary with quality metrics
    """
    quality_metrics = {
        'completeness': 0.0,
        'consistency': 0.0,
        'accuracy': 0.0,
        'overall_score': 0.0,
        'issues': []
    }
    
    # Completeness score
    total_cells = df.shape[0] * df.shape[1]
    missing_cells = df.isnull().sum().sum()
    completeness = (total_cells - missing_cells) / total_cells
    quality_metrics['completeness'] = completeness
    
    if completeness < 0.9:
        quality_metrics['issues'].append(f"Low completeness: {completeness:.2%}")
    
    # Consistency score (check for duplicates)
    duplicate_rows = df.duplicated().sum()
    consistency = 1 - (duplicate_rows / len(df))
    quality_metrics['consistency'] = consistency
    
    if consistency < 0.95:
        quality_metrics['issues'].append(f"Low consistency: {duplicate_rows} duplicate rows")
    
    # Accuracy score (check for reasonable ranges)
    accuracy_score = 1.0
    numerical_cols = df.select_dtypes(include=[np.number]).columns
    
    for col in numerical_cols:
        if col in ['Year']:
            years = df[col].dropna()
            if len(years) > 0:
                if years.min() < 1900 or years.max() > 2030:
                    accuracy_score -= 0.1
                    quality_metrics['issues'].append(f"Unusual year range in {col}")
        
        elif col in ['Yield']:
            yields = df[col].dropna()
            if len(yields) > 0:
                if yields.min() < 0 or yields.max() > 10000:
                    accuracy_score -= 0.1
                    quality_metrics['issues'].append(f"Unusual yield range in {col}")
    
    quality_metrics['accuracy'] = max(0, accuracy_score)
    
    # Overall score
    quality_metrics['overall_score'] = (
        quality_metrics['completeness'] * 0.4 +
        quality_metrics['consistency'] * 0.3 +
        quality_metrics['accuracy'] * 0.3
    )
    
    return quality_metrics

# Utility functions for Streamlit app
def format_number(value: float, decimals: int = 2) -> str:
    """Format number for display"""
    if abs(value) >= 1e6:
        return f"{value/1e6:.{decimals}f}M"
    elif abs(value) >= 1e3:
        return f"{value/1e3:.{decimals}f}K"
    else:
        return f"{value:.{decimals}f}"

def get_color_for_value(value: float, threshold: float = 0) -> str:
    """Get color based on value (green for positive, red for negative)"""
    if value > threshold:
        return "green"
    elif value < threshold:
        return "red"
    else:
        return "gray"

def create_metric_card(title: str, value: str, delta: str = None, delta_color: str = "normal") -> str:
    """Create HTML for metric card"""
    delta_html = ""
    if delta:
        delta_html = f'<div class="delta" style="color: {delta_color};">{delta}</div>'
    
    return f"""
    <div class="metric-card">
        <div class="metric-title">{title}</div>
        <div class="metric-value">{value}</div>
        {delta_html}
    </div>
    """

if __name__ == "__main__":
    # Test the helper functions
    print("🧪 Testing helper utilities...")
    
    # Test data validation
    test_df = pd.DataFrame({
        'State': ['Andhra Pradesh', 'Bihar', 'Gujarat'],
        'Crop': ['Rice', 'Wheat', 'Maize'],
        'Year': [2020, 2021, 2022],
        'Yield': [2500, 1800, 2200]
    })
    
    validation = validate_data_format(test_df)
    print(f"✅ Data validation: {validation['is_valid']}")
    
    # Test data quality score
    quality = get_data_quality_score(test_df)
    print(f"✅ Data quality score: {quality['overall_score']:.2f}")
    
    print("🎉 Helper utilities test completed!")
