"""
Machine Learning Models for Bug Duration Prediction
"""

import pandas as pd
import numpy as np
import pickle
import os
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import streamlit as st

from utils.feature_engineering import (
    extract_features_from_log,
    encode_categorical_features,
    get_feature_columns
)


class BugDurationPredictor:
    """
    Machine Learning model for predicting bug resolution duration.
    """
    
    def __init__(self, model_type='random_forest'):
        """
        Initialize the predictor.
        
        Args:
            model_type: Type of model ('random_forest', 'gradient_boosting', 'linear')
        """
        self.model_type = model_type
        self.model = None
        self.encoders = None
        self.feature_columns = get_feature_columns()
        self.is_trained = False
        self.metrics = {}
        self.feature_importance = None
        
        # Initialize model
        if model_type == 'random_forest':
            self.model = RandomForestRegressor(
                n_estimators=100,
                max_depth=10,
                min_samples_split=5,
                min_samples_leaf=2,
                random_state=42,
                n_jobs=-1
            )
        elif model_type == 'gradient_boosting':
            self.model = GradientBoostingRegressor(
                n_estimators=100,
                max_depth=5,
                learning_rate=0.1,
                random_state=42
            )
        elif model_type == 'linear':
            self.model = LinearRegression()
        else:
            raise ValueError(f"Unknown model type: {model_type}")
    
    def train(self, df, test_size=0.2):
        """
        Train the model on historical data.
        
        Args:
            df: Event log dataframe
            test_size: Proportion of data for testing
            
        Returns:
            dict: Training metrics
        """
        # Extract features
        features_df = extract_features_from_log(df)
        
        if features_df.empty or len(features_df) < 10:
            raise ValueError("Not enough data to train model (minimum 10 cases required)")
        
        # Encode categorical features
        features_encoded, self.encoders = encode_categorical_features(features_df, fit=True)
        
        # Prepare X and y
        X = features_encoded[self.feature_columns]
        y = features_encoded['duration_hours']
        
        # Handle missing columns
        for col in self.feature_columns:
            if col not in X.columns:
                X[col] = 0
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=42
        )
        
        # Train model
        self.model.fit(X_train, y_train)
        self.is_trained = True
        
        # Evaluate
        y_pred_train = self.model.predict(X_train)
        y_pred_test = self.model.predict(X_test)
        
        self.metrics = {
            'train_mae': mean_absolute_error(y_train, y_pred_train),
            'train_rmse': np.sqrt(mean_squared_error(y_train, y_pred_train)),
            'train_r2': r2_score(y_train, y_pred_train),
            'test_mae': mean_absolute_error(y_test, y_pred_test),
            'test_rmse': np.sqrt(mean_squared_error(y_test, y_pred_test)),
            'test_r2': r2_score(y_test, y_pred_test),
            'n_samples': len(features_df),
            'n_features': len(self.feature_columns)
        }
        
        # Feature importance (for tree-based models)
        if hasattr(self.model, 'feature_importances_'):
            self.feature_importance = pd.DataFrame({
                'feature': self.feature_columns,
                'importance': self.model.feature_importances_
            }).sort_values('importance', ascending=False)
        
        return self.metrics
    
    def predict(self, features_df):
        """
        Predict duration for new bugs.
        
        Args:
            features_df: DataFrame with features
            
        Returns:
            np.array: Predicted durations in hours
        """
        if not self.is_trained:
            raise ValueError("Model is not trained. Call train() first.")
        
        # Encode features
        features_encoded, _ = encode_categorical_features(
            features_df, 
            fit=False, 
            encoders=self.encoders
        )
        
        # Prepare X
        X = features_encoded[self.feature_columns]
        
        # Handle missing columns
        for col in self.feature_columns:
            if col not in X.columns:
                X[col] = 0
        
        # Predict
        predictions = self.model.predict(X)
        
        # Ensure non-negative predictions
        predictions = np.maximum(predictions, 0)
        
        return predictions
    
    def cross_validate(self, df, cv=5):
        """
        Perform cross-validation.
        
        Args:
            df: Event log dataframe
            cv: Number of folds
            
        Returns:
            dict: Cross-validation scores
        """
        # Extract features
        features_df = extract_features_from_log(df)
        
        if features_df.empty:
            return {}
        
        # Encode features
        features_encoded, self.encoders = encode_categorical_features(features_df, fit=True)
        
        # Prepare X and y
        X = features_encoded[self.feature_columns]
        y = features_encoded['duration_hours']
        
        # Handle missing columns
        for col in self.feature_columns:
            if col not in X.columns:
                X[col] = 0
        
        # Cross-validate
        cv_scores = cross_val_score(
            self.model, X, y, 
            cv=cv, 
            scoring='neg_mean_absolute_error'
        )
        
        return {
            'cv_mae_mean': -cv_scores.mean(),
            'cv_mae_std': cv_scores.std(),
            'cv_scores': -cv_scores
        }
    
    def save_model(self, filepath='models/bug_predictor.pkl'):
        """
        Save trained model to disk.
        
        Args:
            filepath: Path to save model
        """
        if not self.is_trained:
            raise ValueError("Model is not trained. Cannot save.")
        
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        
        model_data = {
            'model': self.model,
            'encoders': self.encoders,
            'feature_columns': self.feature_columns,
            'model_type': self.model_type,
            'metrics': self.metrics,
            'feature_importance': self.feature_importance
        }
        
        with open(filepath, 'wb') as f:
            pickle.dump(model_data, f)
    
    @classmethod
    def load_model(cls, filepath='models/bug_predictor.pkl'):
        """
        Load trained model from disk.
        
        Args:
            filepath: Path to model file
            
        Returns:
            BugDurationPredictor: Loaded predictor
        """
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Model file not found: {filepath}")
        
        with open(filepath, 'rb') as f:
            model_data = pickle.load(f)
        
        predictor = cls(model_type=model_data['model_type'])
        predictor.model = model_data['model']
        predictor.encoders = model_data['encoders']
        predictor.feature_columns = model_data['feature_columns']
        predictor.metrics = model_data.get('metrics', {})
        predictor.feature_importance = model_data.get('feature_importance', None)
        predictor.is_trained = True
        
        return predictor
    
    def get_prediction_interval(self, features_df, confidence=0.9):
        """
        Get prediction interval (for tree-based models).
        
        Args:
            features_df: DataFrame with features
            confidence: Confidence level
            
        Returns:
            tuple: (lower_bound, upper_bound)
        """
        if not self.is_trained:
            raise ValueError("Model is not trained.")
        
        prediction = self.predict(features_df)
        
        # For tree-based models, use predictions from all trees
        if hasattr(self.model, 'estimators_'):
            # Get predictions from all trees
            tree_predictions = np.array([
                tree.predict(features_df[self.feature_columns]) 
                for tree in self.model.estimators_
            ])
            
            # Calculate percentiles
            lower_percentile = (1 - confidence) / 2
            upper_percentile = 1 - lower_percentile
            
            lower_bound = np.percentile(tree_predictions, lower_percentile * 100, axis=0)
            upper_bound = np.percentile(tree_predictions, upper_percentile * 100, axis=0)
        else:
            # For other models, use simple standard deviation
            std = self.metrics.get('test_rmse', prediction * 0.3)
            z_score = 1.96 if confidence == 0.95 else 1.645  # 95% or 90% confidence
            lower_bound = prediction - z_score * std
            upper_bound = prediction + z_score * std
        
        return np.maximum(lower_bound, 0), upper_bound


@st.cache_resource
def train_model_cached(df, model_type='random_forest'):
    """
    Train model with caching for Streamlit.
    
    Args:
        df: Event log dataframe
        model_type: Type of model
        
    Returns:
        BugDurationPredictor: Trained predictor
    """
    predictor = BugDurationPredictor(model_type=model_type)
    predictor.train(df)
    return predictor


def compare_models(df):
    """
    Compare different model types.
    
    Args:
        df: Event log dataframe
        
    Returns:
        pd.DataFrame: Comparison of model metrics
    """
    model_types = ['random_forest', 'gradient_boosting', 'linear']
    results = []
    
    for model_type in model_types:
        try:
            predictor = BugDurationPredictor(model_type=model_type)
            metrics = predictor.train(df)
            
            results.append({
                'model': model_type,
                'test_mae': metrics['test_mae'],
                'test_rmse': metrics['test_rmse'],
                'test_r2': metrics['test_r2'],
                'train_mae': metrics['train_mae'],
                'train_r2': metrics['train_r2']
            })
        except Exception as e:
            results.append({
                'model': model_type,
                'test_mae': np.nan,
                'test_rmse': np.nan,
                'test_r2': np.nan,
                'train_mae': np.nan,
                'train_r2': np.nan,
                'error': str(e)
            })
    
    return pd.DataFrame(results)

