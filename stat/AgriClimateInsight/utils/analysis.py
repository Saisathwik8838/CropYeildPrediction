"""
Statistical Analysis Module for AgriClimate Insight Portal
Performs comprehensive statistical analysis on climate and agriculture data
"""

import pandas as pd
import numpy as np
from scipy import stats
from statsmodels.tsa.stattools import grangercausalitytests
from statsmodels.stats.diagnostic import het_breuschpagan
from statsmodels.stats.stattools import durbin_watson
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import cross_val_score
from typing import Dict, List, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')


class StatisticalAnalyzer:
    """
    Main class for statistical analysis of climate and agriculture data
    """
    
    def __init__(self, df: pd.DataFrame):
        """
        Initialize the analyzer with a DataFrame
        
        Args:
            df: Preprocessed DataFrame
        """
        self.df = df.copy()
        self.results = {}
        
    def compute_summary_statistics(self) -> Dict[str, pd.DataFrame]:
        """
        Compute comprehensive summary statistics
        
        Returns:
            Dictionary containing various summary statistics
        """
        print("📊 Computing summary statistics...")
        
        # Basic descriptive statistics
        numerical_cols = self.df.select_dtypes(include=[np.number]).columns.tolist()
        basic_stats = self.df[numerical_cols].describe()
        
        # Additional statistics
        additional_stats = pd.DataFrame(index=['skewness', 'kurtosis', 'iqr', 'cv'])
        for col in numerical_cols:
            additional_stats[col] = [
                self.df[col].skew(),
                self.df[col].kurtosis(),
                self.df[col].quantile(0.75) - self.df[col].quantile(0.25),
                self.df[col].std() / self.df[col].mean() if self.df[col].mean() != 0 else np.nan
            ]
        
        # State-wise statistics
        state_stats = self.df.groupby('State')[numerical_cols].agg(['mean', 'std', 'min', 'max'])
        
        # Crop-wise statistics
        crop_stats = self.df.groupby('Crop')[numerical_cols].agg(['mean', 'std', 'min', 'max'])
        
        # Year-wise trends
        year_stats = self.df.groupby('Year')[numerical_cols].agg(['mean', 'std'])
        
        self.results['summary_stats'] = {
            'basic': basic_stats,
            'additional': additional_stats,
            'by_state': state_stats,
            'by_crop': crop_stats,
            'by_year': year_stats
        }
        
        print("✅ Summary statistics computed")
        return self.results['summary_stats']
    
    def identify_trends_and_anomalies(self, window: int = 5) -> Dict[str, pd.DataFrame]:
        """
        Identify trends and anomalies using rolling statistics
        
        Args:
            window: Rolling window size for trend analysis
            
        Returns:
            Dictionary containing trend and anomaly analysis results
        """
        print("📈 Identifying trends and anomalies...")
        
        # Key variables for trend analysis
        trend_vars = ['Yield', 'Rainfall', 'Temperature']
        trend_vars = [var for var in trend_vars if var in self.df.columns]
        
        trend_results = {}
        
        for var in trend_vars:
            # Calculate rolling statistics
            rolling_mean = self.df.groupby('Year')[var].mean().rolling(window=window, center=True).mean()
            rolling_std = self.df.groupby('Year')[var].mean().rolling(window=window, center=True).std()
            
            # Detect anomalies (values beyond 2 standard deviations)
            mean_val = self.df[var].mean()
            std_val = self.df[var].std()
            anomaly_threshold = 2 * std_val
            
            anomalies = self.df[abs(self.df[var] - mean_val) > anomaly_threshold]
            
            # Calculate trend using linear regression
            years = self.df['Year'].dt.year if hasattr(self.df['Year'].iloc[0], 'year') else self.df['Year']
            slope, intercept, r_value, p_value, std_err = stats.linregress(years, self.df[var])
            
            trend_results[var] = {
                'rolling_mean': rolling_mean,
                'rolling_std': rolling_std,
                'anomalies': anomalies,
                'trend_slope': slope,
                'trend_r_squared': r_value**2,
                'trend_p_value': p_value,
                'trend_significance': 'Significant' if p_value < 0.05 else 'Not Significant'
            }
        
        self.results['trends_anomalies'] = trend_results
        print("✅ Trends and anomalies identified")
        return trend_results
    
    def compute_correlations(self) -> Dict[str, pd.DataFrame]:
        """
        Compute correlation matrices and relationships
        
        Returns:
            Dictionary containing correlation analysis results
        """
        print("🔗 Computing correlations...")
        
        # Select numerical columns for correlation analysis
        numerical_cols = self.df.select_dtypes(include=[np.number]).columns.tolist()
        
        # Overall correlation matrix
        overall_corr = self.df[numerical_cols].corr()
        
        # State-wise correlations
        state_correlations = {}
        for state in self.df['State'].unique():
            state_data = self.df[self.df['State'] == state][numerical_cols]
            if len(state_data) > 1:
                state_correlations[state] = state_data.corr()
        
        # Crop-wise correlations
        crop_correlations = {}
        for crop in self.df['Crop'].unique():
            crop_data = self.df[self.df['Crop'] == crop][numerical_cols]
            if len(crop_data) > 1:
                crop_correlations[crop] = crop_data.corr()
        
        # Climate-Yield correlations
        climate_yield_corr = {}
        if 'Yield' in self.df.columns:
            climate_vars = ['Rainfall', 'Temperature']
            climate_vars = [var for var in climate_vars if var in self.df.columns]
            
            for var in climate_vars:
                climate_yield_corr[var] = self.df[['Yield', var]].corr().iloc[0, 1]
        
        self.results['correlations'] = {
            'overall': overall_corr,
            'by_state': state_correlations,
            'by_crop': crop_correlations,
            'climate_yield': climate_yield_corr
        }
        
        print("✅ Correlations computed")
        return self.results['correlations']
    
    def granger_causality_tests(self, max_lags: int = 4) -> Dict[str, Dict]:
        """
        Perform Granger causality tests between climate variables and yield
        
        Args:
            max_lags: Maximum number of lags to test
            
        Returns:
            Dictionary containing Granger causality test results
        """
        print("🔍 Performing Granger causality tests...")
        
        granger_results = {}
        
        # Prepare time series data
        if 'Yield' in self.df.columns:
            climate_vars = ['Rainfall', 'Temperature']
            climate_vars = [var for var in climate_vars if var in self.df.columns]
            
            for var in climate_vars:
                try:
                    # Create time series for Granger test
                    ts_data = self.df[['Yield', var]].dropna()
                    
                    if len(ts_data) > max_lags + 10:  # Ensure sufficient data
                        # Perform Granger causality test
                        gc_result = grangercausalitytests(ts_data, maxlag=max_lags, verbose=False)
                        
                        # Extract p-values for each lag
                        p_values = {}
                        for lag in range(1, max_lags + 1):
                            if lag in gc_result:
                                p_values[f'lag_{lag}'] = gc_result[lag][0]['ssr_ftest'][1]
                        
                        granger_results[f'{var}_to_Yield'] = {
                            'p_values': p_values,
                            'significant_lags': [lag for lag, p_val in p_values.items() if p_val < 0.05],
                            'min_p_value': min(p_values.values()) if p_values else 1.0
                        }
                        
                except Exception as e:
                    print(f"⚠️ Granger test failed for {var}: {e}")
                    granger_results[f'{var}_to_Yield'] = {'error': str(e)}
        
        self.results['granger_causality'] = granger_results
        print("✅ Granger causality tests completed")
        return granger_results
    
    def regression_analysis(self) -> Dict[str, Dict]:
        """
        Perform various regression analyses
        
        Returns:
            Dictionary containing regression analysis results
        """
        print("📈 Performing regression analysis...")
        
        regression_results = {}
        
        if 'Yield' in self.df.columns:
            # Prepare features and target
            feature_cols = ['Rainfall', 'Temperature', 'Area']
            feature_cols = [col for col in feature_cols if col in self.df.columns]
            
            if feature_cols:
                X = self.df[feature_cols].fillna(self.df[feature_cols].mean())
                y = self.df['Yield'].fillna(self.df['Yield'].mean())
                
                # Remove any remaining NaN values
                mask = ~(X.isna().any(axis=1) | y.isna())
                X = X[mask]
                y = y[mask]
                
                if len(X) > 0 and len(y) > 0:
                    # Linear Regression
                    lr_model = LinearRegression()
                    lr_model.fit(X, y)
                    lr_pred = lr_model.predict(X)
                    
                    regression_results['linear'] = {
                        'model': lr_model,
                        'coefficients': dict(zip(feature_cols, lr_model.coef_)),
                        'intercept': lr_model.intercept_,
                        'r_squared': r2_score(y, lr_pred),
                        'rmse': np.sqrt(mean_squared_error(y, lr_pred)),
                        'mae': mean_absolute_error(y, lr_pred)
                    }
                    
                    # Ridge Regression
                    ridge_model = Ridge(alpha=1.0)
                    ridge_model.fit(X, y)
                    ridge_pred = ridge_model.predict(X)
                    
                    regression_results['ridge'] = {
                        'model': ridge_model,
                        'coefficients': dict(zip(feature_cols, ridge_model.coef_)),
                        'intercept': ridge_model.intercept_,
                        'r_squared': r2_score(y, ridge_pred),
                        'rmse': np.sqrt(mean_squared_error(y, ridge_pred)),
                        'mae': mean_absolute_error(y, ridge_pred)
                    }
                    
                    # Lasso Regression
                    lasso_model = Lasso(alpha=0.1)
                    lasso_model.fit(X, y)
                    lasso_pred = lasso_model.predict(X)
                    
                    regression_results['lasso'] = {
                        'model': lasso_model,
                        'coefficients': dict(zip(feature_cols, lasso_model.coef_)),
                        'intercept': lasso_model.intercept_,
                        'r_squared': r2_score(y, lasso_pred),
                        'rmse': np.sqrt(mean_squared_error(y, lasso_pred)),
                        'mae': mean_absolute_error(y, lasso_pred)
                    }
                    
                    # Cross-validation scores
                    cv_scores = cross_val_score(lr_model, X, y, cv=5, scoring='r2')
                    regression_results['cross_validation'] = {
                        'mean_score': cv_scores.mean(),
                        'std_score': cv_scores.std(),
                        'scores': cv_scores.tolist()
                    }
        
        self.results['regression'] = regression_results
        print("✅ Regression analysis completed")
        return regression_results
    
    def statistical_tests(self) -> Dict[str, Dict]:
        """
        Perform various statistical tests
        
        Returns:
            Dictionary containing statistical test results
        """
        print("🧪 Performing statistical tests...")
        
        test_results = {}
        
        # Normality tests for key variables
        key_vars = ['Yield', 'Rainfall', 'Temperature']
        key_vars = [var for var in key_vars if var in self.df.columns]
        
        for var in key_vars:
            data = self.df[var].dropna()
            if len(data) > 3:
                # Shapiro-Wilk test for normality
                shapiro_stat, shapiro_p = stats.shapiro(data)
                
                # Kolmogorov-Smirnov test
                ks_stat, ks_p = stats.kstest(data, 'norm', args=(data.mean(), data.std()))
                
                test_results[f'{var}_normality'] = {
                    'shapiro_statistic': shapiro_stat,
                    'shapiro_p_value': shapiro_p,
                    'ks_statistic': ks_stat,
                    'ks_p_value': ks_p,
                    'is_normal': shapiro_p > 0.05
                }
        
        # ANOVA test for yield differences across states
        if 'Yield' in self.df.columns and 'State' in self.df.columns:
            state_groups = [group['Yield'].values for name, group in self.df.groupby('State') if len(group) > 1]
            if len(state_groups) > 1:
                f_stat, f_p = stats.f_oneway(*state_groups)
                test_results['yield_by_state_anova'] = {
                    'f_statistic': f_stat,
                    'p_value': f_p,
                    'significant': f_p < 0.05
                }
        
        # ANOVA test for yield differences across crops
        if 'Yield' in self.df.columns and 'Crop' in self.df.columns:
            crop_groups = [group['Yield'].values for name, group in self.df.groupby('Crop') if len(group) > 1]
            if len(crop_groups) > 1:
                f_stat, f_p = stats.f_oneway(*crop_groups)
                test_results['yield_by_crop_anova'] = {
                    'f_statistic': f_stat,
                    'p_value': f_p,
                    'significant': f_p < 0.05
                }
        
        self.results['statistical_tests'] = test_results
        print("✅ Statistical tests completed")
        return test_results
    
    def run_complete_analysis(self) -> Dict[str, Dict]:
        """
        Run complete statistical analysis pipeline
        
        Returns:
            Dictionary containing all analysis results
        """
        print("🚀 Starting complete statistical analysis...")
        
        # Run all analyses
        self.compute_summary_statistics()
        self.identify_trends_and_anomalies()
        self.compute_correlations()
        self.granger_causality_tests()
        self.regression_analysis()
        self.statistical_tests()
        
        print("✅ Complete statistical analysis finished!")
        return self.results
    
    def get_analysis_summary(self) -> Dict[str, str]:
        """
        Generate a summary of key findings
        
        Returns:
            Dictionary with key findings and insights
        """
        summary = {}
        
        # Key correlations
        if 'correlations' in self.results and 'climate_yield' in self.results['correlations']:
            climate_corr = self.results['correlations']['climate_yield']
            for var, corr in climate_corr.items():
                summary[f'{var}_yield_correlation'] = f"{corr:.3f} ({'Strong' if abs(corr) > 0.7 else 'Moderate' if abs(corr) > 0.3 else 'Weak'})"
        
        # Significant trends
        if 'trends_anomalies' in self.results:
            for var, results in self.results['trends_anomalies'].items():
                if results['trend_significance'] == 'Significant':
                    trend_direction = 'increasing' if results['trend_slope'] > 0 else 'decreasing'
                    summary[f'{var}_trend'] = f"Significant {trend_direction} trend (R² = {results['trend_r_squared']:.3f})"
        
        # Best regression model
        if 'regression' in self.results:
            models = ['linear', 'ridge', 'lasso']
            best_model = None
            best_r2 = -1
            
            for model in models:
                if model in self.results['regression']:
                    r2 = self.results['regression'][model]['r_squared']
                    if r2 > best_r2:
                        best_r2 = r2
                        best_model = model
            
            if best_model:
                summary['best_model'] = f"{best_model.title()} regression (R² = {best_r2:.3f})"
        
        return summary


if __name__ == "__main__":
    # Test the analyzer
    from data_preprocessing import DataPreprocessor
    
    # Load and preprocess data
    preprocessor = DataPreprocessor()
    processed_data = preprocessor.preprocess_all()
    
    # Run statistical analysis
    analyzer = StatisticalAnalyzer(processed_data)
    results = analyzer.run_complete_analysis()
    
    # Print summary
    summary = analyzer.get_analysis_summary()
    print("\n📊 Analysis Summary:")
    for key, value in summary.items():
        print(f"   {key}: {value}")
