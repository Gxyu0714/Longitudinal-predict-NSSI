"""
Longitudinal Prediction Framework for N-Wave Cohort Survey Data
===============================================================

A flexible framework for predicting outcomes using machine learning approaches
on longitudinal data from multi-wave cohort studies.

This implementation supports:
- N-wave longitudinal data processing
- Multiple prediction models
- Time-aware cross-validation strategies
- Feature selection techniques
- Comprehensive evaluation metrics

Based on the methodology from:
Guo, X., Liu, S., Jiang, L., Xiong, Z., Wang, L., Lu, L., Li, X., Zhao, L., & Shek, D. T. L. (2025). 
Longitudinal machine learning prediction of non-suicidal self-injury among Chinese adolescents: 
A prospective multicenter Cohort study. Journal of Affective Disorders, 120110.

Author: Xinyu Guo
Date: August 2025
"""

import pandas as pd
import numpy as np
from sklearn.impute import KNNImputer
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import roc_auc_score, precision_recall_curve, auc
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import StratifiedKFold
import warnings
warnings.filterwarnings('ignore')

# ============================================================================
# CONFIGURATION
# ============================================================================

class Config:
    """Configuration parameters for the prediction framework."""
    
    # Data parameters
    WAVE_PREFIX = 'T'  # Prefix for time wave columns
    ID_COLUMN = 'ID'   # Identifier column name
    WAVE_COLUMN = 'wave'  # Wave indicator column name
    
    # Target variable
    TARGET_VAR = 'NSSI'  # Binary outcome variable
    
    # Model parameters
    N_FEATURES = 50  # Number of features to select
    RANDOM_STATE = 42  # Random seed for reproducibility
    
    # Cross-validation parameters
    CV_FOLDS = 5  # Number of cross-validation folds

# ============================================================================
# DATA PROCESSING PIPELINE
# ============================================================================

class LongitudinalDataProcessor:
    """
    Process and clean longitudinal cohort survey data.
    
    Handles:
    - Data transformation from wide to long format
    - Missing value imputation
    - Variable standardization
    - Quality control checks
    """
    
    def __init__(self, config=Config):
        self.config = config
        self.scaler = StandardScaler()
        self.imputer = KNNImputer(n_neighbors=5)
        self.feature_selector = SelectKBest(score_func=f_classif, k=config.N_FEATURES)
        
    def load_and_transform_data(self, file_path):
        """
        Load wide-format data and transform to long-format.
        
        Args:
            file_path (str): Path to the input CSV file
            
        Returns:
            pd.DataFrame: Transformed long-format data
        """
        print("Step 1: Loading and transforming data...")
        
        # Load data
        df = pd.read_csv(file_path)
        print(f"Loaded data with shape: {df.shape}")
        
        # Identify time-varying variables
        columns = df.columns.tolist()
        base_vars = set()
        for col in columns:
            if col != self.config.ID_COLUMN and col.startswith(self.config.WAVE_PREFIX):
                base_vars.add(col[3:])  # Remove Tx_ prefix
        
        print(f"Identified {len(base_vars)} base variables")
        
        # Transform to long format
        long_data = []
        for _, row in df.iterrows():
            for wave in range(1, 5):  # Assuming T1, T2, T3, T4
                time_prefix = f"{self.config.WAVE_PREFIX}{wave}_"
                new_row = {
                    self.config.ID_COLUMN: row[self.config.ID_COLUMN],
                    self.config.WAVE_COLUMN: wave
                }
                
                for base_var in base_vars:
                    original_var = f"{time_prefix}{base_var}"
                    if original_var in row:
                        new_row[base_var] = row[original_var]
                
                long_data.append(new_row)
        
        long_df = pd.DataFrame(long_data)
        
        # Reorder columns
        cols = long_df.columns.tolist()
        cols.remove(self.config.ID_COLUMN)
        cols.remove(self.config.WAVE_COLUMN)
        cols = [self.config.ID_COLUMN, self.config.WAVE_COLUMN] + sorted(cols)
        long_df = long_df[cols]
        
        print(f"Transformed data shape: {long_df.shape}")
        return long_df
    
    def clean_and_process_target(self, df):
        """
        Clean data and process target variable.
        
        Args:
            df (pd.DataFrame): Input long-format data
            
        Returns:
            pd.DataFrame: Cleaned data with processed target variable
        """
        print("\nStep 2: Cleaning data and processing target variable...")
        
        # Remove observations with missing target values
        initial_count = len(df)
        df_clean = df.dropna(subset=[self.config.TARGET_VAR])
        final_count = len(df_clean)
        
        print(f"Removed {initial_count - final_count} observations with missing target values")
        
        # Convert target to binary if needed
        if df_clean[self.config.TARGET_VAR].dtype != 'int64':
            df_clean[self.config.TARGET_VAR] = (
                df_clean[self.config.TARGET_VAR] > 0
            ).astype(int)
        
        print(f"Target variable distribution:")
        print(df_clean[self.config.TARGET_VAR].value_counts())
        
        return df_clean
    
    def fill_missing_values(self, df):
        """
        Fill missing values using appropriate methods.
        
        Args:
            df (pd.DataFrame): Input data with missing values
            
        Returns:
            pd.DataFrame: Data with filled missing values
        """
        print("\nStep 3: Filling missing values...")
        
        # Separate variable types
        numeric_vars = df.select_dtypes(include=[np.number]).columns.tolist()
        categorical_vars = df.select_dtypes(include=['object']).columns.tolist()
        
        # Remove ID and wave columns from processing
        numeric_vars = [var for var in numeric_vars 
                       if var not in [self.config.ID_COLUMN, self.config.WAVE_COLUMN]]
        categorical_vars = [var for var in categorical_vars 
                           if var not in [self.config.ID_COLUMN, self.config.WAVE_COLUMN]]
        
        print(f"Processing {len(numeric_vars)} numeric variables")
        print(f"Processing {len(categorical_vars)} categorical variables")
        
        # Fill missing values for numeric variables using KNN imputation
        if numeric_vars:
            df_numeric = df[numeric_vars].copy()
            df_numeric_imputed = pd.DataFrame(
                self.imputer.fit_transform(df_numeric),
                columns=numeric_vars,
                index=df.index
            )
            df[numeric_vars] = df_numeric_imputed
        
        # Fill missing values for categorical variables using mode imputation
        if categorical_vars:
            for var in categorical_vars:
                mode_value = df[var].mode()
                if len(mode_value) > 0:
                    df[var] = df[var].fillna(mode_value[0])
        
        # Report missing value status
        missing_after = df.isnull().sum().sum()
        print(f"Remaining missing values: {missing_after}")
        
        return df
    
    def standardize_features(self, df, fit_transform=True):
        """
        Standardize continuous features.
        
        Args:
            df (pd.DataFrame): Input data
            fit_transform (bool): Whether to fit the scaler or just transform
            
        Returns:
            pd.DataFrame: Data with standardized features
        """
        print("\nStep 4: Standardizing features...")
        
        # Identify continuous variables (excluding ID, wave, and target)
        exclude_vars = [self.config.ID_COLUMN, self.config.WAVE_COLUMN, self.config.TARGET_VAR]
        continuous_vars = [col for col in df.columns 
                         if col not in exclude_vars and df[col].dtype in ['float64', 'int64']]
        
        print(f"Standardizing {len(continuous_vars)} continuous variables")
        
        # Standardize features
        if fit_transform:
            df_continuous_scaled = pd.DataFrame(
                self.scaler.fit_transform(df[continuous_vars]),
                columns=continuous_vars,
                index=df.index
            )
        else:
            df_continuous_scaled = pd.DataFrame(
                self.scaler.transform(df[continuous_vars]),
                columns=continuous_vars,
                index=df.index
            )
        
        df[continuous_vars] = df_continuous_scaled
        
        return df

# ============================================================================
# FEATURE SELECTION
# ============================================================================

class FeatureSelector:
    """
    Select relevant features for prediction using statistical methods.
    """
    
    def __init__(self, config=Config):
        self.config = config
        self.selector = SelectKBest(score_func=f_classif, k=config.N_FEATURES)
        self.selected_features = None
    
    def select_features(self, X, y):
        """
        Select top K features based on ANOVA F-scores.
        
        Args:
            X (pd.DataFrame): Feature matrix
            y (pd.Series): Target variable
            
        Returns:
            pd.DataFrame: Selected features
        """
        print(f"\nSelecting top {self.config.N_FEATURES} features...")
        
        # Fit feature selector
        self.selector.fit(X, y)
        
        # Get selected feature indices
        selected_indices = self.selector.get_support(indices=True)
        self.selected_features = X.columns[selected_indices].tolist()
        
        # Transform features
        X_selected = self.selector.transform(X)
        X_selected_df = pd.DataFrame(
            X_selected,
            columns=self.selected_features,
            index=X.index
        )
        
        print(f"Selected features: {self.selected_features[:10]}...")
        
        return X_selected_df

# ============================================================================
# PREDICTION MODELS
# ============================================================================

class PredictionModels:
    """
    Collection of prediction models for longitudinal data.
    """
    
    def __init__(self, config=Config):
        self.config = config
        self.models = {
            'random_forest': RandomForestClassifier(
                n_estimators=100,
                random_state=config.RANDOM_STATE,
                n_jobs=-1
            ),
            'logistic_regression': LogisticRegression(
                random_state=config.RANDOM_STATE,
                max_iter=1000
            ),
            'svm': SVC(
                probability=True,
                random_state=config.RANDOM_STATE
            )
        }
        self.trained_models = {}
    
    def train_models(self, X, y):
        """
        Train all prediction models.
        
        Args:
            X (pd.DataFrame): Feature matrix
            y (pd.Series): Target variable
        """
        print("\nTraining prediction models...")
        
        for name, model in self.models.items():
            print(f"Training {name}...")
            model.fit(X, y)
            self.trained_models[name] = model
    
    def predict(self, X, model_name=None):
        """
        Make predictions using trained models.
        
        Args:
            X (pd.DataFrame): Feature matrix
            model_name (str): Specific model to use (None for all)
            
        Returns:
            dict: Predictions from each model
        """
        predictions = {}
        
        if model_name:
            models_to_use = {model_name: self.trained_models[model_name]}
        else:
            models_to_use = self.trained_models
        
        for name, model in models_to_use.items():
            predictions[f"{name}_pred"] = model.predict(X)
            predictions[f"{name}_prob"] = model.predict_proba(X)[:, 1]
        
        return predictions

# ============================================================================
# TIME-AWARE CROSS-VALIDATION
# ============================================================================

class TimeAwareCV:
    """
    Time-aware cross-validation strategies for longitudinal data.
    """
    
    def __init__(self, config=Config):
        self.config = config
    
    def fixed_split_validation(self, df):
        """
        Fixed split validation: Train on early waves, test on later waves.
        
        Args:
            df (pd.DataFrame): Longitudinal data
            
        Returns:
            dict: Validation results
        """
        print("\nPerforming fixed split validation...")
        
        # Split data
        train_data = df[df[self.config.WAVE_COLUMN].isin([1])]
        val_data = df[df[self.config.WAVE_COLUMN].isin([2])]
        test_data = df[df[self.config.WAVE_COLUMN].isin([3])]
        external_val_data = df[df[self.config.WAVE_COLUMN].isin([4])]
        
        results = {
            'train_waves': [1],
            'validation_waves': [2],
            'test_waves': [3],
            'external_validation_waves': [4]
        }
        
        print(f"Train waves: {results['train_waves']}")
        print(f"Validation waves: {results['validation_waves']}")
        print(f"Test waves: {results['test_waves']}")
        print(f"External validation waves: {results['external_validation_waves']}")
        
        return results
    
    def sliding_window_validation(self, df):
        """
        Sliding window validation: Train on expanding windows.
        
        Args:
            df (pd.DataFrame): Longitudinal data
            
        Returns:
            list: Validation results for each window
        """
        print("\nPerforming sliding window validation...")
        
        results = []
        max_wave = df[self.config.WAVE_COLUMN].max()
        
        for i in range(1, max_wave):
            train_waves = list(range(1, i + 1))
            test_wave = i + 1
            
            if test_wave <= max_wave:
                result = {
                    'window': i,
                    'train_waves': train_waves,
                    'test_wave': test_wave
                }
                results.append(result)
                print(f"Window {i}: Train on waves {train_waves}, Test on wave {test_wave}")
        
        return results
    
    def cumulative_learning_validation(self, df):
        """
        Cumulative learning validation: Accumulate data over time.
        
        Args:
            df (pd.DataFrame): Longitudinal data
            
        Returns:
            list: Validation results for each round
        """
        print("\nPerforming cumulative learning validation...")
        
        results = []
        max_wave = df[self.config.WAVE_COLUMN].max()
        
        for i in range(1, max_wave):
            train_waves = list(range(1, i + 1))
            test_wave = i + 1
            
            if test_wave <= max_wave:
                result = {
                    'round': i,
                    'train_waves': train_waves,
                    'test_wave': test_wave
                }
                results.append(result)
                print(f"Round {i}: Train on waves {train_waves}, Test on wave {test_wave}")
        
        return results

# ============================================================================
# MODEL EVALUATION
# ============================================================================

class ModelEvaluator:
    """
    Comprehensive model evaluation using multiple metrics.
    """
    
    def __init__(self, config=Config):
        self.config = config
        self.metrics = [
            'roc_auc', 'prc_auc', 'accuracy', 'precision', 'recall', 'f1_score'
        ]
    
    def evaluate_model(self, y_true, y_pred, y_pred_proba):
        """
        Evaluate model performance using multiple metrics.
        
        Args:
            y_true (array-like): True labels
            y_pred (array-like): Predicted labels
            y_pred_proba (array-like): Predicted probabilities
            
        Returns:
            dict: Evaluation metrics
        """
        metrics = {}
        
        # ROC AUC
        try:
            metrics['roc_auc'] = roc_auc_score(y_true, y_pred_proba)
        except:
            metrics['roc_auc'] = np.nan
        
        # Precision-Recall AUC
        try:
            precision, recall, _ = precision_recall_curve(y_true, y_pred_proba)
            metrics['prc_auc'] = auc(recall, precision)
        except:
            metrics['prc_auc'] = np.nan
        
        # Classification metrics
        metrics['accuracy'] = accuracy_score(y_true, y_pred)
        metrics['precision'] = precision_score(y_true, y_pred, zero_division=0)
        metrics['recall'] = recall_score(y_true, y_pred, zero_division=0)
        metrics['f1_score'] = f1_score(y_true, y_pred, zero_division=0)
        
        return metrics
    
    def print_evaluation_results(self, results, model_name="Model"):
        """
        Print evaluation results in a formatted way.
        
        Args:
            results (dict): Evaluation metrics
            model_name (str): Name of the model
        """
        print(f"\n{model_name} Results:")
        print("-" * 40)
        for metric, value in results.items():
            if not np.isnan(value):
                print(f"{metric.upper()}: {value:.4f}")
            else:
                print(f"{metric.upper()}: N/A")

# ============================================================================
# MAIN EXECUTION FRAMEWORK
# ============================================================================

class LongitudinalPredictor:
    """
    Main framework for longitudinal prediction in N-wave cohort studies.
    """
    
    def __init__(self, config=Config):
        self.config = config
        self.processor = LongitudinalDataProcessor(config)
        self.feature_selector = FeatureSelector(config)
        self.models = PredictionModels(config)
        self.cv = TimeAwareCV(config)
        self.evaluator = ModelEvaluator(config)
    
    def run_complete_pipeline(self, data_file_path=None):
        """
        Execute the complete prediction pipeline.
        
        Args:
            data_file_path (str): Path to input data file (optional)
        """
        print("=" * 80)
        print("LONGITUDINAL PREDICTION PIPELINE FOR N-WAVE COHORT STUDIES")
        print("=" * 80)
        
        try:
            # Step 1: Data Processing
            if data_file_path:
                df_long = self.processor.load_and_transform_data(data_file_path)
            else:
                print("No data file provided. Demonstrating pipeline structure.")
                df_long = self._create_sample_data()
            
            df_clean = self.processor.clean_and_process_target(df_long)
            df_imputed = self.processor.fill_missing_values(df_clean)
            df_standardized = self.processor.standardize_features(df_imputed)
            
            # Step 2: Feature Preparation
            X, y = self._prepare_features_and_target(df_standardized)
            
            # Step 3: Feature Selection
            X_selected = self.feature_selector.select_features(X, y)
            
            # Step 4: Model Training
            self.models.train_models(X_selected, y)
            
            # Step 5: Cross-Validation Strategies
            self._perform_cross_validation(df_standardized)
            
            # Step 6: Model Evaluation
            self._evaluate_models(X_selected, y)
            
            print("\n" + "=" * 80)
            print("PIPELINE COMPLETED SUCCESSFULLY")
            print("=" * 80)
            
        except Exception as e:
            print(f"\nError in pipeline execution: {str(e)}")
            print("Pipeline execution failed.")
    
    def _create_sample_data(self):
        """
        Create sample data for demonstration purposes.
        
        Returns:
            pd.DataFrame: Sample longitudinal data
        """
        # Create dummy data for demonstration
        np.random.seed(self.config.RANDOM_STATE)
        n_individuals = 1000
        n_waves = 4
        
        data = []
        for i in range(n_individuals):
            individual_id = f"ID_{i:04d}"
            for wave in range(1, n_waves + 1):
                row = {
                    self.config.ID_COLUMN: individual_id,
                    self.config.WAVE_COLUMN: wave,
                    'Age': np.random.randint(10, 18),
                    'Gender': np.random.choice([0, 1]),
                    'Grade': np.random.randint(3, 10),
                    'BMI': np.random.normal(22, 3),
                    'Family_Income': np.random.normal(50000, 20000),
                    self.config.TARGET_VAR: np.random.choice([0, 1], p=[0.7, 0.3])
                }
                data.append(row)
        
        return pd.DataFrame(data)
    
    def _prepare_features_and_target(self, df):
        """
        Prepare feature matrix and target vector.
        
        Args:
            df (pd.DataFrame): Processed data
            
        Returns:
            tuple: (X, y) Feature matrix and target vector
        """
        # Exclude ID, wave, and target from features
        exclude_cols = [self.config.ID_COLUMN, self.config.WAVE_COLUMN, self.config.TARGET_VAR]
        feature_cols = [col for col in df.columns if col not in exclude_cols]
        
        X = df[feature_cols]
        y = df[self.config.TARGET_VAR]
        
        print(f"\nFeature matrix shape: {X.shape}")
        print(f"Target vector shape: {y.shape}")
        print(f"Target distribution: {y.value_counts().to_dict()}")
        
        return X, y
    
    def _perform_cross_validation(self, df):
        """
        Perform all cross-validation strategies.
        
        Args:
            df (pd.DataFrame): Longitudinal data
        """
        print("\n" + "=" * 60)
        print("CROSS-VALIDATION STRATEGIES")
        print("=" * 60)
        
        # Fixed split validation
        self.cv.fixed_split_validation(df)
        
        # Sliding window validation
        self.cv.sliding_window_validation(df)
        
        # Cumulative learning validation
        self.cv.cumulative_learning_validation(df)
    
    def _evaluate_models(self, X, y):
        """
        Evaluate all trained models.
        
        Args:
            X (pd.DataFrame): Feature matrix
            y (pd.Series): Target vector
        """
        print("\n" + "=" * 60)
        print("MODEL EVALUATION")
        print("=" * 60)
        
        # Make predictions with all models
        predictions = self.models.predict(X)
        
        # Evaluate each model
        for model_name in self.models.trained_models.keys():
            pred_key = f"{model_name}_pred"
            prob_key = f"{model_name}_prob"
            
            if pred_key in predictions and prob_key in predictions:
                results = self.evaluator.evaluate_model(
                    y, predictions[pred_key], predictions[prob_key]
                )
                self.evaluator.print_evaluation_results(results, model_name)

# ============================================================================
# MAIN EXECUTION
# ============================================================================

def main():
    """
    Main execution function.
    """
    # Initialize predictor
    predictor = LongitudinalPredictor()
    
    # Run complete pipeline
    predictor.run_complete_pipeline()

if __name__ == "__main__":
    main()