"""
Machine Learning Modeling Module for AgriClimate Insight Portal
Implements various ML models for climate-agriculture prediction and analysis
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.preprocessing import StandardScaler
try:
    import xgboost as xgb
    XGBOOST_AVAILABLE = True
except (ImportError, Exception) as e:
    XGBOOST_AVAILABLE = False
    print(f"XGBoost not available: {e}")
    print("Install OpenMP runtime with: brew install libomp")

from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.stattools import adfuller
import joblib
from typing import Dict, List, Tuple, Optional, Any
import warnings
warnings.filterwarnings('ignore')

try:
    from prophet import Prophet
    PROPHET_AVAILABLE = True
except ImportError:
    PROPHET_AVAILABLE = False
    print("⚠️ Prophet not available. Install with: pip install prophet")


class MLModeler:
    """
    Main class for machine learning modeling and time series analysis
    """
    
    def __init__(self, df: pd.DataFrame):
        """
        Initialize the modeler with a DataFrame
        
        Args:
            df: Preprocessed DataFrame
        """
        self.df = df.copy()
        self.models = {}
        self.results = {}
        self.scaler = StandardScaler()
        
    def prepare_features(self, target_col: str = 'Yield') -> Tuple[pd.DataFrame, pd.Series]:
        """
        Prepare features and target for ML models
        
        Args:
            target_col: Name of the target column
            
        Returns:
            Tuple of (features, target)
        """
        print("🔧 Preparing features for ML models...")
        
        # Select feature columns
        feature_cols = ['Rainfall', 'Temperature', 'Area']
        feature_cols = [col for col in feature_cols if col in self.df.columns]
        
        # Add encoded categorical features
        encoded_cols = [col for col in self.df.columns if col.endswith('_encoded') or '_' in col and col not in feature_cols]
        feature_cols.extend(encoded_cols)
        
        # Prepare features and target
        X = self.df[feature_cols].fillna(self.df[feature_cols].mean())
        y = self.df[target_col].fillna(self.df[target_col].mean())
        
        # Remove any remaining NaN values
        mask = ~(X.isna().any(axis=1) | y.isna())
        X = X[mask]
        y = y[mask]
        
        print(f"✅ Features prepared: {X.shape[1]} features, {len(X)} samples")
        return X, y
    
    def train_random_forest(self, X: pd.DataFrame, y: pd.Series) -> Dict[str, Any]:
        """
        Train Random Forest model
        
        Args:
            X: Features
            y: Target
            
        Returns:
            Dictionary containing model results
        """
        print("🌲 Training Random Forest model...")
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # Train model
        rf_model = RandomForestRegressor(
            n_estimators=100,
            max_depth=10,
            min_samples_split=5,
            min_samples_leaf=2,
            random_state=42
        )
        
        rf_model.fit(X_train, y_train)
        
        # Make predictions
        y_pred_train = rf_model.predict(X_train)
        y_pred_test = rf_model.predict(X_test)
        
        # Calculate metrics
        train_r2 = r2_score(y_train, y_pred_train)
        test_r2 = r2_score(y_test, y_pred_test)
        train_rmse = np.sqrt(mean_squared_error(y_train, y_pred_train))
        test_rmse = np.sqrt(mean_squared_error(y_test, y_pred_test))
        train_mae = mean_absolute_error(y_train, y_pred_train)
        test_mae = mean_absolute_error(y_test, y_pred_test)
        
        # Feature importance
        feature_importance = dict(zip(X.columns, rf_model.feature_importances_))
        
        # Cross-validation
        cv_scores = cross_val_score(rf_model, X, y, cv=5, scoring='r2')
        
        results = {
            'model': rf_model,
            'train_r2': train_r2,
            'test_r2': test_r2,
            'train_rmse': train_rmse,
            'test_rmse': test_rmse,
            'train_mae': train_mae,
            'test_mae': test_mae,
            'feature_importance': feature_importance,
            'cv_mean': cv_scores.mean(),
            'cv_std': cv_scores.std(),
            'predictions': {
                'train': y_pred_train,
                'test': y_pred_test,
                'actual_train': y_train,
                'actual_test': y_test
            }
        }
        
        self.models['random_forest'] = rf_model
        self.results['random_forest'] = results
        
        print(f"✅ Random Forest trained - Test R²: {test_r2:.3f}, Test RMSE: {test_rmse:.3f}")
        return results
    
    def train_xgboost(self, X: pd.DataFrame, y: pd.Series) -> Dict[str, Any]:
        """
        Train XGBoost model
        
        Args:
            X: Features
            y: Target
            
        Returns:
            Dictionary containing model results
        """
        if not XGBOOST_AVAILABLE:
            print("XGBoost not available, skipping XGBoost training")
            return {'error': 'XGBoost not available'}
        
        print("Training XGBoost model...")
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # Train model
        xgb_model = xgb.XGBRegressor(
            n_estimators=100,
            max_depth=6,
            learning_rate=0.1,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42
        )
        
        xgb_model.fit(X_train, y_train)
        
        # Make predictions
        y_pred_train = xgb_model.predict(X_train)
        y_pred_test = xgb_model.predict(X_test)
        
        # Calculate metrics
        train_r2 = r2_score(y_train, y_pred_train)
        test_r2 = r2_score(y_test, y_pred_test)
        train_rmse = np.sqrt(mean_squared_error(y_train, y_pred_train))
        test_rmse = np.sqrt(mean_squared_error(y_test, y_pred_test))
        train_mae = mean_absolute_error(y_train, y_pred_train)
        test_mae = mean_absolute_error(y_test, y_pred_test)
        
        # Feature importance
        feature_importance = dict(zip(X.columns, xgb_model.feature_importances_))
        
        # Cross-validation
        cv_scores = cross_val_score(xgb_model, X, y, cv=5, scoring='r2')
        
        results = {
            'model': xgb_model,
            'train_r2': train_r2,
            'test_r2': test_r2,
            'train_rmse': train_rmse,
            'test_rmse': test_rmse,
            'train_mae': train_mae,
            'test_mae': test_mae,
            'feature_importance': feature_importance,
            'cv_mean': cv_scores.mean(),
            'cv_std': cv_scores.std(),
            'predictions': {
                'train': y_pred_train,
                'test': y_pred_test,
                'actual_train': y_train,
                'actual_test': y_test
            }
        }
        
        self.models['xgboost'] = xgb_model
        self.results['xgboost'] = results
        
        print(f"✅ XGBoost trained - Test R²: {test_r2:.3f}, Test RMSE: {test_rmse:.3f}")
        return results
    
    def train_arima(self, ts_data: pd.Series, order: Tuple[int, int, int] = (1, 1, 1)) -> Dict[str, Any]:
        """
        Train ARIMA model for time series forecasting
        
        Args:
            ts_data: Time series data
            order: ARIMA order (p, d, q)
            
        Returns:
            Dictionary containing model results
        """
        print("📈 Training ARIMA model...")
        
        try:
            # Check stationarity
            adf_result = adfuller(ts_data.dropna())
            is_stationary = adf_result[1] < 0.05
            
            # Fit ARIMA model
            arima_model = ARIMA(ts_data.dropna(), order=order)
            arima_fitted = arima_model.fit()
            
            # Make predictions
            forecast = arima_fitted.forecast(steps=5)
            fitted_values = arima_fitted.fittedvalues
            
            # Calculate metrics
            actual = ts_data.dropna()
            predicted = fitted_values
            r2 = r2_score(actual, predicted)
            rmse = np.sqrt(mean_squared_error(actual, predicted))
            mae = mean_absolute_error(actual, predicted)
            
            results = {
                'model': arima_fitted,
                'r2': r2,
                'rmse': rmse,
                'mae': mae,
                'forecast': forecast,
                'fitted_values': fitted_values,
                'is_stationary': is_stationary,
                'adf_p_value': adf_result[1],
                'aic': arima_fitted.aic,
                'bic': arima_fitted.bic,
                'params': arima_fitted.params
            }
            
            self.models['arima'] = arima_fitted
            self.results['arima'] = results
            
            print(f"✅ ARIMA trained - R²: {r2:.3f}, RMSE: {rmse:.3f}")
            return results
            
        except Exception as e:
            print(f"❌ ARIMA training failed: {e}")
            return {'error': str(e)}
    
    def train_prophet(self, df_prophet: pd.DataFrame) -> Dict[str, Any]:
        """
        Train Prophet model for time series forecasting
        
        Args:
            df_prophet: DataFrame with 'ds' (date) and 'y' (value) columns
            
        Returns:
            Dictionary containing model results
        """
        if not PROPHET_AVAILABLE:
            return {'error': 'Prophet not available'}
        
        print("🔮 Training Prophet model...")
        
        try:
            # Initialize and fit Prophet model
            prophet_model = Prophet(
                yearly_seasonality=True,
                weekly_seasonality=False,
                daily_seasonality=False,
                seasonality_mode='multiplicative'
            )
            
            prophet_model.fit(df_prophet)
            
            # Make future predictions
            future = prophet_model.make_future_dataframe(periods=12, freq='Y')
            forecast = prophet_model.predict(future)
            
            # Calculate metrics on historical data
            historical = forecast[forecast['ds'] <= df_prophet['ds'].max()]
            actual = df_prophet['y'].values
            predicted = historical['yhat'].values
            
            r2 = r2_score(actual, predicted)
            rmse = np.sqrt(mean_squared_error(actual, predicted))
            mae = mean_absolute_error(actual, predicted)
            
            results = {
                'model': prophet_model,
                'r2': r2,
                'rmse': rmse,
                'mae': mae,
                'forecast': forecast,
                'trend': forecast['trend'].values,
                'seasonal': forecast['yearly'].values
            }
            
            self.models['prophet'] = prophet_model
            self.results['prophet'] = results
            
            print(f"✅ Prophet trained - R²: {r2:.3f}, RMSE: {rmse:.3f}")
            return results
            
        except Exception as e:
            print(f"❌ Prophet training failed: {e}")
            return {'error': str(e)}
    
    def prepare_time_series_data(self, state: str = None, crop: str = None) -> pd.DataFrame:
        """
        Prepare time series data for ARIMA and Prophet models
        
        Args:
            state: Filter by specific state
            crop: Filter by specific crop
            
        Returns:
            DataFrame suitable for time series modeling
        """
        print("📅 Preparing time series data...")
        
        # Filter data
        ts_df = self.df.copy()
        if state:
            ts_df = ts_df[ts_df['State'] == state]
        if crop:
            ts_df = ts_df[ts_df['Crop'] == crop]
        
        # Aggregate by year
        yearly_data = ts_df.groupby('Year')['Yield'].mean().reset_index()
        
        # Convert to Prophet format
        prophet_df = pd.DataFrame({
            'ds': pd.to_datetime(yearly_data['Year'], format='%Y'),
            'y': yearly_data['Yield']
        })
        
        print(f"✅ Time series data prepared: {len(prophet_df)} data points")
        return prophet_df
    
    def model_comparison(self) -> pd.DataFrame:
        """
        Compare performance of all trained models
        
        Returns:
            DataFrame with model comparison results
        """
        print("📊 Comparing model performance...")
        
        comparison_data = []
        
        for model_name, results in self.results.items():
            if 'error' not in results:
                if model_name in ['arima', 'prophet']:
                    # Time series models
                    comparison_data.append({
                        'Model': model_name.title(),
                        'R²': results.get('r2', 0),
                        'RMSE': results.get('rmse', 0),
                        'MAE': results.get('mae', 0),
                        'Type': 'Time Series'
                    })
                else:
                    # ML models
                    comparison_data.append({
                        'Model': model_name.title(),
                        'R²': results.get('test_r2', 0),
                        'RMSE': results.get('test_rmse', 0),
                        'MAE': results.get('test_mae', 0),
                        'CV_Score': results.get('cv_mean', 0),
                        'Type': 'Machine Learning'
                    })
        
        comparison_df = pd.DataFrame(comparison_data)
        comparison_df = comparison_df.sort_values('R²', ascending=False)
        
        print("✅ Model comparison completed")
        return comparison_df
    
    def save_models(self, model_dir: str = "models/") -> None:
        """
        Save trained models using joblib
        
        Args:
            model_dir: Directory to save models
        """
        print("💾 Saving models...")
        
        import os
        os.makedirs(model_dir, exist_ok=True)
        
        for model_name, model in self.models.items():
            if model_name != 'prophet':  # Prophet models need special handling
                joblib.dump(model, f"{model_dir}{model_name}_model.pkl")
                print(f"   ✅ {model_name} model saved")
        
        # Save results
        joblib.dump(self.results, f"{model_dir}model_results.pkl")
        print("✅ All models and results saved")
    
    def load_models(self, model_dir: str = "models/") -> None:
        """
        Load trained models from disk
        
        Args:
            model_dir: Directory containing saved models
        """
        print("📂 Loading models...")
        
        import os
        if os.path.exists(f"{model_dir}model_results.pkl"):
            self.results = joblib.load(f"{model_dir}model_results.pkl")
            print("✅ Model results loaded")
        
        # Load individual models
        for model_name in ['random_forest', 'xgboost', 'arima']:
            model_path = f"{model_dir}{model_name}_model.pkl"
            if os.path.exists(model_path):
                self.models[model_name] = joblib.load(model_path)
                print(f"   ✅ {model_name} model loaded")
    
    def run_complete_modeling(self) -> Dict[str, Any]:
        """
        Run complete modeling pipeline
        
        Returns:
            Dictionary containing all modeling results
        """
        print("🚀 Starting complete modeling pipeline...")
        
        # Prepare features for ML models
        X, y = self.prepare_features()
        
        # Train ML models
        self.train_random_forest(X, y)
        if XGBOOST_AVAILABLE:
            self.train_xgboost(X, y)
        else:
            print("Skipping XGBoost training due to dependency issues")
        
        # Prepare time series data
        ts_data = self.prepare_time_series_data()
        
        # Train time series models
        if len(ts_data) > 10:  # Ensure sufficient data
            # ARIMA on yield time series
            yield_ts = ts_data.set_index('ds')['y']
            self.train_arima(yield_ts)
            
            # Prophet model
            self.train_prophet(ts_data)
        
        # Compare models
        comparison = self.model_comparison()
        self.results['model_comparison'] = comparison
        
        print("✅ Complete modeling pipeline finished!")
        return self.results
    
    def get_model_summary(self) -> Dict[str, str]:
        """
        Generate a summary of model performance
        
        Returns:
            Dictionary with key model insights
        """
        summary = {}
        
        if 'model_comparison' in self.results:
            comparison = self.results['model_comparison']
            if not comparison.empty:
                best_model = comparison.iloc[0]
                summary['best_model'] = f"{best_model['Model']} (R² = {best_model['R²']:.3f})"
        
        # Feature importance from best ML model
        for model_name in ['xgboost', 'random_forest']:
            if model_name in self.results and 'feature_importance' in self.results[model_name]:
                importance = self.results[model_name]['feature_importance']
                top_feature = max(importance, key=importance.get)
                summary[f'{model_name}_top_feature'] = f"{top_feature} ({importance[top_feature]:.3f})"
                break
        
        return summary


if __name__ == "__main__":
    # Test the modeler
    from data_preprocessing import DataPreprocessor
    
    # Load and preprocess data
    preprocessor = DataPreprocessor()
    processed_data = preprocessor.preprocess_all()
    
    # Run modeling
    modeler = MLModeler(processed_data)
    results = modeler.run_complete_modeling()
    
    # Print summary
    summary = modeler.get_model_summary()
    print("\n🤖 Model Summary:")
    for key, value in summary.items():
        print(f"   {key}: {value}")
