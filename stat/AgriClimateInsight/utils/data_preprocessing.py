"""
Data Preprocessing Module for AgriClimate Insight Portal
Handles loading, cleaning, and merging of climate and agriculture datasets
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
from sklearn.impute import KNNImputer, SimpleImputer
from typing import Tuple, Dict, List
import warnings
warnings.filterwarnings('ignore')


class DataPreprocessor:
    """
    Main class for preprocessing climate and agriculture data
    """
    
    def __init__(self, data_path: str = "data/"):
        """
        Initialize the preprocessor with data path
        
        Args:
            data_path: Path to the data directory
        """
        self.data_path = data_path
        self.master_df = None
        self.scaler = StandardScaler()
        self.label_encoders = {}
        self.onehot_encoders = {}
        
    def load_datasets(self) -> Dict[str, pd.DataFrame]:
        """
        Load all CSV datasets from the data directory
        
        Returns:
            Dictionary containing loaded datasets
        """
        datasets = {}
        
        try:
            # Load individual crop files
            crop_files = self._load_crop_files()
            datasets['crops_final'] = crop_files
            
            # Load climate datasets
            datasets['rainfall'] = pd.read_csv(f"{self.data_path}rainfall.csv")
            datasets['temperature'] = pd.read_csv(f"{self.data_path}temperature.csv")
            
            print("✅ Successfully loaded all datasets")
            for name, df in datasets.items():
                print(f"   {name}: {df.shape}")
                
        except FileNotFoundError as e:
            print(f"❌ Error loading datasets: {e}")
            print("📝 Creating sample datasets for demonstration...")
            datasets = self._create_sample_datasets()
            
        return datasets
    
    def _load_crop_files(self) -> pd.DataFrame:
        """
        Load and combine all individual crop CSV files
        
        Returns:
            Combined DataFrame with all crop data
        """
        import os
        import glob
        
        crop_files = []
        crops_dir = f"{self.data_path}Crops/"
        
        # Get all CSV files in the Crops directory
        csv_files = glob.glob(f"{crops_dir}*.csv")
        
        print(f"📁 Found {len(csv_files)} crop files")
        
        for file_path in csv_files:
            try:
                # Extract crop name from filename
                crop_name = os.path.basename(file_path).replace('.csv', '')
                
                # Load the CSV file
                df = pd.read_csv(file_path)
                
                # Standardize column names and add crop information
                df = self._standardize_crop_data(df, crop_name)
                
                if not df.empty:
                    crop_files.append(df)
                    print(f"   ✅ Loaded {crop_name}: {df.shape}")
                
            except Exception as e:
                print(f"   ⚠️ Error loading {file_path}: {e}")
                continue
        
        if crop_files:
            # Combine all crop data
            combined_df = pd.concat(crop_files, ignore_index=True)
            print(f"✅ Combined crop data: {combined_df.shape}")
            return combined_df
        else:
            print("❌ No crop files could be loaded")
            return pd.DataFrame()
    
    def _standardize_crop_data(self, df: pd.DataFrame, crop_name: str) -> pd.DataFrame:
        """
        Standardize crop data format from FAO structure
        
        Args:
            df: Raw crop DataFrame
            crop_name: Name of the crop
            
        Returns:
            Standardized DataFrame
        """
        try:
            # Create a standardized DataFrame
            standardized_df = pd.DataFrame()
            
            # Extract year and value columns
            if 'Year' in df.columns and 'Value' in df.columns and 'Element' in df.columns:
                # Pivot the data to get Area, Production, and Yield as separate columns
                pivot_df = df.pivot_table(
                    index=['Year'], 
                    columns='Element', 
                    values='Value', 
                    aggfunc='first'
                ).reset_index()
                
                # Flatten column names
                pivot_df.columns.name = None
                
                # Rename columns to standard format
                column_mapping = {
                    'Area harvested': 'Area',
                    'Production': 'Production',
                    'Yield': 'Yield'
                }
                
                # Map available columns
                for old_col, new_col in column_mapping.items():
                    if old_col in pivot_df.columns:
                        standardized_df[new_col] = pivot_df[old_col]
                
                # Add crop and year information
                standardized_df['Crop'] = crop_name.title()
                standardized_df['Year'] = pivot_df['Year']
                standardized_df['State'] = 'India'  # All data is for India
                
                # Calculate yield if not present but area and production are
                if 'Yield' not in standardized_df.columns and 'Area' in standardized_df.columns and 'Production' in standardized_df.columns:
                    standardized_df['Yield'] = standardized_df['Production'] / standardized_df['Area']
                
                return standardized_df
            
            else:
                print(f"   ⚠️ Unexpected format in {crop_name}")
                return pd.DataFrame()
                
        except Exception as e:
            print(f"   ⚠️ Error standardizing {crop_name}: {e}")
            return pd.DataFrame()
    
    def _create_sample_datasets(self) -> Dict[str, pd.DataFrame]:
        """
        Create sample datasets for demonstration purposes
        """
        np.random.seed(42)
        
        # Sample states and crops
        states = ['Andhra Pradesh', 'Bihar', 'Gujarat', 'Haryana', 'Karnataka', 
                 'Madhya Pradesh', 'Maharashtra', 'Punjab', 'Rajasthan', 'Tamil Nadu']
        crops = ['Rice', 'Wheat', 'Maize', 'Sugarcane', 'Cotton', 'Soybean']
        years = list(range(2000, 2023))
        
        # Create sample crop data
        crop_data = []
        for state in states:
            for crop in crops:
                for year in years:
                    crop_data.append({
                        'State': state,
                        'Crop': crop,
                        'Year': year,
                        'Yield': np.random.normal(2000, 500) + np.random.normal(0, 100),
                        'Area': np.random.normal(1000, 200),
                        'Production': np.random.normal(2000000, 500000)
                    })
        
        crops_final = pd.DataFrame(crop_data)
        
        # Create sample rainfall data
        rainfall_data = []
        for state in states:
            for year in years:
                rainfall_data.append({
                    'State': state,
                    'Year': year,
                    'Rainfall': np.random.normal(1000, 300),
                    'Rainfall_Anomaly': np.random.normal(0, 50)
                })
        
        rainfall_df = pd.DataFrame(rainfall_data)
        
        # Create sample temperature data
        temp_data = []
        for state in states:
            for year in years:
                temp_data.append({
                    'State': state,
                    'Year': year,
                    'Temperature': np.random.normal(25, 5),
                    'Temperature_Anomaly': np.random.normal(0, 1)
                })
        
        temp_df = pd.DataFrame(temp_data)
        
        # Create sample predictions
        predictions_data = []
        for state in states:
            for crop in crops:
                for year in years[-5:]:  # Last 5 years
                    predictions_data.append({
                        'State': state,
                        'Crop': crop,
                        'Year': year,
                        'Predicted_Yield': np.random.normal(2000, 500),
                        'Confidence_Interval': np.random.uniform(0.7, 0.95)
                    })
        
        crops_predictions = pd.DataFrame(predictions_data)
        
        return {
            'crops_final': crops_final,
            'crops_predictions': crops_predictions,
            'rainfall': rainfall_df,
            'temperature': temp_df
        }
    
    def merge_datasets(self, datasets: Dict[str, pd.DataFrame]) -> pd.DataFrame:
        """
        Merge all datasets into a master DataFrame
        
        Args:
            datasets: Dictionary of loaded datasets
            
        Returns:
            Merged master DataFrame
        """
        print("🔄 Merging datasets...")
        
        # Start with crop data as base
        master_df = datasets['crops_final'].copy()
        
        # Merge rainfall data
        rainfall_df = datasets['rainfall'].copy()
        # Rename YEAR column to Year for consistency
        if 'YEAR' in rainfall_df.columns:
            rainfall_df = rainfall_df.rename(columns={'YEAR': 'Year'})
        # Use annual rainfall (ANN column)
        if 'ANN' in rainfall_df.columns:
            rainfall_df = rainfall_df[['Year', 'ANN']].rename(columns={'ANN': 'Rainfall'})
        
        master_df = master_df.merge(
            rainfall_df, 
            on=['Year'], 
            how='left'
        )
        
        # Merge temperature data
        temp_df = datasets['temperature'].copy()
        # Rename YEAR column to Year for consistency
        if 'YEAR' in temp_df.columns:
            temp_df = temp_df.rename(columns={'YEAR': 'Year'})
        # Use annual temperature (ANNUAL column)
        if 'ANNUAL' in temp_df.columns:
            temp_df = temp_df[['Year', 'ANNUAL']].rename(columns={'ANNUAL': 'Temperature'})
        
        master_df = master_df.merge(
            temp_df, 
            on=['Year'], 
            how='left'
        )
        
        print(f"✅ Merged dataset shape: {master_df.shape}")
        self.master_df = master_df
        
        return master_df
    
    def handle_missing_values(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Handle missing values using appropriate imputation strategies
        
        Args:
            df: Input DataFrame
            
        Returns:
            DataFrame with missing values handled
        """
        print("🔄 Handling missing values...")
        
        df_clean = df.copy()
        
        # Separate numerical and categorical columns
        numerical_cols = df_clean.select_dtypes(include=[np.number]).columns.tolist()
        categorical_cols = df_clean.select_dtypes(include=['object']).columns.tolist()
        
        # Handle numerical missing values with KNN imputation
        if numerical_cols:
            knn_imputer = KNNImputer(n_neighbors=5)
            df_clean[numerical_cols] = knn_imputer.fit_transform(df_clean[numerical_cols])
        
        # Handle categorical missing values with mode imputation
        for col in categorical_cols:
            if df_clean[col].isnull().any():
                mode_value = df_clean[col].mode()[0] if not df_clean[col].mode().empty else 'Unknown'
                df_clean[col] = df_clean[col].fillna(mode_value)
        
        missing_count = df_clean.isnull().sum().sum()
        print(f"✅ Missing values handled. Remaining missing values: {missing_count}")
        
        return df_clean
    
    def encode_categorical_variables(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Encode categorical variables using LabelEncoder and OneHotEncoder
        
        Args:
            df: Input DataFrame
            
        Returns:
            DataFrame with encoded categorical variables
        """
        print("🔄 Encoding categorical variables...")
        
        df_encoded = df.copy()
        
        # Columns to encode
        categorical_cols = ['State', 'Crop']
        
        for col in categorical_cols:
            if col in df_encoded.columns:
                # Label encoding for ordinal-like categories
                if col == 'Crop':
                    le = LabelEncoder()
                    df_encoded[f'{col}_encoded'] = le.fit_transform(df_encoded[col])
                    self.label_encoders[col] = le
                
                # One-hot encoding for nominal categories
                ohe = OneHotEncoder(sparse_output=False, drop='first')
                encoded_cols = ohe.fit_transform(df_encoded[[col]])
                feature_names = [f'{col}_{cat}' for cat in ohe.categories_[0][1:]]
                df_encoded[feature_names] = encoded_cols
                self.onehot_encoders[col] = ohe
        
        print("✅ Categorical variables encoded")
        return df_encoded
    
    def normalize_features(self, df: pd.DataFrame, feature_cols: List[str] = None) -> pd.DataFrame:
        """
        Normalize numerical features using StandardScaler
        
        Args:
            df: Input DataFrame
            feature_cols: List of columns to normalize
            
        Returns:
            DataFrame with normalized features
        """
        print("🔄 Normalizing features...")
        
        df_normalized = df.copy()
        
        if feature_cols is None:
            # Default feature columns for normalization
            feature_cols = ['Yield', 'Area', 'Production', 'Rainfall', 'Temperature']
            feature_cols = [col for col in feature_cols if col in df_normalized.columns]
        
        # Normalize specified features
        if feature_cols:
            df_normalized[feature_cols] = self.scaler.fit_transform(df_normalized[feature_cols])
            print(f"✅ Normalized features: {feature_cols}")
        
        return df_normalized
    
    def convert_time_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Convert time-related columns to datetime format
        
        Args:
            df: Input DataFrame
            
        Returns:
            DataFrame with converted time columns
        """
        print("🔄 Converting time columns...")
        
        df_time = df.copy()
        
        # Convert Year to datetime if it exists
        if 'Year' in df_time.columns:
            df_time['Year'] = pd.to_datetime(df_time['Year'], format='%Y')
        
        print("✅ Time columns converted")
        return df_time
    
    def preprocess_all(self) -> pd.DataFrame:
        """
        Complete preprocessing pipeline
        
        Returns:
            Fully preprocessed master DataFrame
        """
        print("🚀 Starting complete preprocessing pipeline...")
        
        # Load datasets
        datasets = self.load_datasets()
        
        # Merge datasets
        master_df = self.merge_datasets(datasets)
        
        # Handle missing values
        master_df = self.handle_missing_values(master_df)
        
        # Encode categorical variables
        master_df = self.encode_categorical_variables(master_df)
        
        # Normalize features
        master_df = self.normalize_features(master_df)
        
        # Convert time columns
        master_df = self.convert_time_columns(master_df)
        
        print("✅ Complete preprocessing pipeline finished!")
        return master_df
    
    def get_summary_stats(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Generate summary statistics for the dataset
        
        Args:
            df: Input DataFrame
            
        Returns:
            DataFrame with summary statistics
        """
        numerical_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        summary_stats = df[numerical_cols].describe()
        
        # Add additional statistics
        summary_stats.loc['skewness'] = df[numerical_cols].skew()
        summary_stats.loc['kurtosis'] = df[numerical_cols].kurtosis()
        
        return summary_stats
    
    def get_data_info(self, df: pd.DataFrame) -> Dict:
        """
        Get comprehensive information about the dataset
        
        Args:
            df: Input DataFrame
            
        Returns:
            Dictionary with dataset information
        """
        info = {
            'shape': df.shape,
            'columns': df.columns.tolist(),
            'dtypes': df.dtypes.to_dict(),
            'missing_values': df.isnull().sum().to_dict(),
            'duplicate_rows': df.duplicated().sum(),
            'memory_usage': df.memory_usage(deep=True).sum() / 1024**2  # MB
        }
        
        return info


if __name__ == "__main__":
    # Test the preprocessor
    preprocessor = DataPreprocessor()
    processed_data = preprocessor.preprocess_all()
    
    print("\n📊 Dataset Summary:")
    print(f"Shape: {processed_data.shape}")
    print(f"Columns: {list(processed_data.columns)}")
    print(f"Memory usage: {processed_data.memory_usage(deep=True).sum() / 1024**2:.2f} MB")
