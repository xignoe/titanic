"""
Model training module for the Titanic Survival Predictor.

This module provides the ModelTrainer class that handles training multiple
machine learning algorithms, validation splits, and cross-validation evaluation.
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report
from typing import Dict, Tuple, Any, Optional
import logging
import joblib
import os

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ModelTrainer:
    """
    Handles training multiple machine learning algorithms for survival prediction.
    
    This class provides functionality to:
    - Split data into training and validation sets
    - Train multiple algorithms (Random Forest, Logistic Regression, SVM)
    - Perform cross-validation evaluation
    - Compare model performance
    """
    
    def __init__(self, random_state: int = 42):
        """
        Initialize the ModelTrainer.
        
        Args:
            random_state: Random seed for reproducible results
        """
        self.random_state = random_state
        self.models = {}
        self.model_scores = {}
        self.tuned_models = {}
        self.best_params = {}
        
    def split_data(self, X: pd.DataFrame, y: pd.Series, 
                   test_size: float = 0.2, stratify: bool = True) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
        """
        Split data into training and validation sets.
        
        Args:
            X: Feature matrix
            y: Target variable
            test_size: Proportion of data to use for validation
            stratify: Whether to stratify the split based on target variable
            
        Returns:
            Tuple of (X_train, X_val, y_train, y_val)
        """
        logger.info(f"Splitting data with test_size={test_size}, stratify={stratify}")
        
        stratify_param = y if stratify else None
        
        X_train, X_val, y_train, y_val = train_test_split(
            X, y, 
            test_size=test_size, 
            random_state=self.random_state,
            stratify=stratify_param
        )
        
        logger.info(f"Training set size: {len(X_train)}, Validation set size: {len(X_val)}")
        return X_train, X_val, y_train, y_val
    
    def _get_default_models(self) -> Dict[str, Any]:
        """
        Get default model configurations for training.
        
        Returns:
            Dictionary of model name to model instance
        """
        return {
            'RandomForest': RandomForestClassifier(
                n_estimators=100,
                max_depth=10,
                min_samples_split=5,
                min_samples_leaf=2,
                random_state=self.random_state
            ),
            'LogisticRegression': LogisticRegression(
                max_iter=1000,
                random_state=self.random_state
            ),
            'SVM': SVC(
                kernel='rbf',
                C=1.0,
                gamma='scale',
                random_state=self.random_state
            )
        }
    
    def train_multiple_models(self, X_train: pd.DataFrame, y_train: pd.Series,
                            models: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Train multiple machine learning algorithms.
        
        Args:
            X_train: Training feature matrix
            y_train: Training target variable
            models: Optional dictionary of models to train. If None, uses default models.
            
        Returns:
            Dictionary of trained models
        """
        if models is None:
            models = self._get_default_models()
            
        logger.info(f"Training {len(models)} models: {list(models.keys())}")
        
        trained_models = {}
        
        for name, model in models.items():
            logger.info(f"Training {name}...")
            try:
                model.fit(X_train, y_train)
                trained_models[name] = model
                logger.info(f"Successfully trained {name}")
            except Exception as e:
                logger.error(f"Failed to train {name}: {str(e)}")
                continue
                
        self.models = trained_models
        return trained_models
    
    def perform_cross_validation(self, model: Any, X: pd.DataFrame, y: pd.Series,
                               cv_folds: int = 5) -> Dict[str, float]:
        """
        Perform cross-validation evaluation for model reliability.
        
        Args:
            model: Trained model to evaluate
            X: Feature matrix
            y: Target variable
            cv_folds: Number of cross-validation folds
            
        Returns:
            Dictionary with cross-validation results
        """
        logger.info(f"Performing {cv_folds}-fold cross-validation")
        
        # Use StratifiedKFold to maintain class distribution
        cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=self.random_state)
        
        # Perform cross-validation
        cv_scores = cross_val_score(model, X, y, cv=cv, scoring='accuracy')
        
        results = {
            'cv_scores': cv_scores,
            'mean_score': cv_scores.mean(),
            'std_score': cv_scores.std(),
            'min_score': cv_scores.min(),
            'max_score': cv_scores.max()
        }
        
        logger.info(f"Cross-validation results: Mean={results['mean_score']:.4f} (+/- {results['std_score']*2:.4f})")
        
        return results
    
    def evaluate_models(self, X_val: pd.DataFrame, y_val: pd.Series) -> Dict[str, Dict[str, float]]:
        """
        Evaluate all trained models on validation data.
        
        Args:
            X_val: Validation feature matrix
            y_val: Validation target variable
            
        Returns:
            Dictionary of model evaluation results
        """
        if not self.models:
            raise ValueError("No models have been trained yet. Call train_multiple_models first.")
            
        logger.info("Evaluating models on validation data")
        
        evaluation_results = {}
        
        for name, model in self.models.items():
            logger.info(f"Evaluating {name}...")
            
            # Make predictions
            y_pred = model.predict(X_val)
            
            # Calculate accuracy
            accuracy = accuracy_score(y_val, y_pred)
            
            # Perform cross-validation on full training data
            cv_results = self.perform_cross_validation(model, X_val, y_val)
            
            evaluation_results[name] = {
                'validation_accuracy': accuracy,
                'cv_mean_score': cv_results['mean_score'],
                'cv_std_score': cv_results['std_score']
            }
            
            logger.info(f"{name} - Validation Accuracy: {accuracy:.4f}")
            
        self.model_scores = evaluation_results
        return evaluation_results
    
    def get_best_model(self) -> Tuple[str, Any, float]:
        """
        Get the best performing model based on validation accuracy.
        
        Returns:
            Tuple of (model_name, model_instance, accuracy_score)
        """
        if not self.model_scores:
            raise ValueError("No model evaluations available. Call evaluate_models first.")
            
        best_model_name = max(self.model_scores.keys(), 
                            key=lambda x: self.model_scores[x]['validation_accuracy'])
        
        best_model = self.models[best_model_name]
        best_score = self.model_scores[best_model_name]['validation_accuracy']
        
        logger.info(f"Best model: {best_model_name} with accuracy: {best_score:.4f}")
        
        return best_model_name, best_model, best_score
    
    def get_model_summary(self) -> pd.DataFrame:
        """
        Get a summary of all model performances.
        
        Returns:
            DataFrame with model performance metrics
        """
        if not self.model_scores:
            raise ValueError("No model evaluations available. Call evaluate_models first.")
            
        summary_data = []
        for name, scores in self.model_scores.items():
            summary_data.append({
                'Model': name,
                'Validation_Accuracy': scores['validation_accuracy'],
                'CV_Mean_Score': scores['cv_mean_score'],
                'CV_Std_Score': scores['cv_std_score']
            })
            
        return pd.DataFrame(summary_data).sort_values('Validation_Accuracy', ascending=False)
    
    def _get_hyperparameter_grids(self) -> Dict[str, Dict[str, list]]:
        """
        Get hyperparameter grids for grid search optimization.
        
        Returns:
            Dictionary of model name to parameter grid
        """
        return {
            'RandomForest': {
                'n_estimators': [50, 100, 200],
                'max_depth': [5, 10, 15, None],
                'min_samples_split': [2, 5, 10],
                'min_samples_leaf': [1, 2, 4]
            },
            'LogisticRegression': {
                'C': [0.1, 1.0, 10.0, 100.0],
                'penalty': ['l1', 'l2'],
                'solver': ['liblinear', 'saga']
            },
            'SVM': {
                'C': [0.1, 1.0, 10.0],
                'kernel': ['rbf', 'linear'],
                'gamma': ['scale', 'auto', 0.001, 0.01]
            }
        }
    
    def tune_hyperparameters(self, X_train: pd.DataFrame, y_train: pd.Series,
                           param_grids: Dict[str, Dict[str, list]] = None,
                           cv_folds: int = 5, scoring: str = 'accuracy') -> Dict[str, Any]:
        """
        Perform hyperparameter tuning using grid search.
        
        Args:
            X_train: Training feature matrix
            y_train: Training target variable
            param_grids: Optional parameter grids. If None, uses default grids.
            cv_folds: Number of cross-validation folds for grid search
            scoring: Scoring metric for optimization
            
        Returns:
            Dictionary of tuned models
        """
        if not self.models:
            raise ValueError("No models have been trained yet. Call train_multiple_models first.")
            
        if param_grids is None:
            param_grids = self._get_hyperparameter_grids()
            
        logger.info(f"Starting hyperparameter tuning with {cv_folds}-fold CV")
        
        tuned_models = {}
        best_params = {}
        
        for name, model in self.models.items():
            if name not in param_grids:
                logger.warning(f"No parameter grid found for {name}, skipping tuning")
                tuned_models[name] = model
                continue
                
            logger.info(f"Tuning hyperparameters for {name}...")
            
            try:
                # Create a fresh model instance for tuning
                base_model = type(model)(random_state=self.random_state)
                
                # Set up grid search
                grid_search = GridSearchCV(
                    estimator=base_model,
                    param_grid=param_grids[name],
                    cv=cv_folds,
                    scoring=scoring,
                    n_jobs=-1,
                    verbose=0
                )
                
                # Perform grid search
                grid_search.fit(X_train, y_train)
                
                # Store results
                tuned_models[name] = grid_search.best_estimator_
                best_params[name] = grid_search.best_params_
                
                logger.info(f"Best parameters for {name}: {grid_search.best_params_}")
                logger.info(f"Best CV score for {name}: {grid_search.best_score_:.4f}")
                
            except Exception as e:
                logger.error(f"Failed to tune {name}: {str(e)}")
                tuned_models[name] = model  # Fall back to original model
                
        self.tuned_models = tuned_models
        self.best_params = best_params
        
        return tuned_models
    
    def compare_models(self, X_val: pd.DataFrame, y_val: pd.Series,
                      use_tuned: bool = True) -> pd.DataFrame:
        """
        Compare model performance and select the best model.
        
        Args:
            X_val: Validation feature matrix
            y_val: Validation target variable
            use_tuned: Whether to use tuned models for comparison
            
        Returns:
            DataFrame with model comparison results
        """
        models_to_compare = self.tuned_models if use_tuned and self.tuned_models else self.models
        
        if not models_to_compare:
            raise ValueError("No models available for comparison.")
            
        logger.info(f"Comparing {'tuned' if use_tuned else 'base'} models")
        
        comparison_results = []
        
        for name, model in models_to_compare.items():
            # Make predictions
            y_pred = model.predict(X_val)
            accuracy = accuracy_score(y_val, y_pred)
            
            # Perform cross-validation
            cv_results = self.perform_cross_validation(model, X_val, y_val)
            
            comparison_results.append({
                'Model': name,
                'Validation_Accuracy': accuracy,
                'CV_Mean_Score': cv_results['mean_score'],
                'CV_Std_Score': cv_results['std_score'],
                'Tuned': use_tuned and name in self.tuned_models,
                'Best_Params': self.best_params.get(name, {}) if use_tuned else {}
            })
            
        comparison_df = pd.DataFrame(comparison_results)
        comparison_df = comparison_df.sort_values('Validation_Accuracy', ascending=False)
        
        return comparison_df
    
    def select_best_model(self, X_val: pd.DataFrame, y_val: pd.Series,
                         min_accuracy: float = 0.8, use_tuned: bool = True) -> Tuple[str, Any, float, bool]:
        """
        Select the best model based on validation performance and accuracy threshold.
        
        Args:
            X_val: Validation feature matrix
            y_val: Validation target variable
            min_accuracy: Minimum accuracy threshold (default 0.8 as per requirements)
            use_tuned: Whether to prefer tuned models
            
        Returns:
            Tuple of (model_name, model_instance, accuracy_score, meets_threshold)
        """
        comparison_df = self.compare_models(X_val, y_val, use_tuned)
        
        # Get the best performing model
        best_row = comparison_df.iloc[0]
        best_name = best_row['Model']
        best_accuracy = best_row['Validation_Accuracy']
        
        # Get the model instance
        models_to_use = self.tuned_models if use_tuned and self.tuned_models else self.models
        best_model = models_to_use[best_name]
        
        # Check if it meets the accuracy threshold
        meets_threshold = best_accuracy >= min_accuracy
        
        logger.info(f"Selected best model: {best_name}")
        logger.info(f"Accuracy: {best_accuracy:.4f}")
        logger.info(f"Meets {min_accuracy:.1%} threshold: {meets_threshold}")
        
        if not meets_threshold:
            logger.warning(f"Best model accuracy ({best_accuracy:.4f}) is below threshold ({min_accuracy:.4f})")
            logger.warning("Consider feature engineering improvements or trying different algorithms")
        
        return best_name, best_model, best_accuracy, meets_threshold
    
    def save_model(self, model: Any, model_name: str, filepath: str = None) -> str:
        """
        Save a trained model to disk.
        
        Args:
            model: Trained model to save
            model_name: Name of the model
            filepath: Optional custom filepath. If None, uses default naming.
            
        Returns:
            Path where the model was saved
        """
        if filepath is None:
            # Create outputs/models directory if it doesn't exist
            os.makedirs('outputs/models', exist_ok=True)
            filepath = f'outputs/models/{model_name}_model.joblib'
        
        try:
            joblib.dump(model, filepath)
            logger.info(f"Model {model_name} saved to {filepath}")
            return filepath
        except Exception as e:
            logger.error(f"Failed to save model {model_name}: {str(e)}")
            raise
    
    def load_model(self, filepath: str) -> Any:
        """
        Load a trained model from disk.
        
        Args:
            filepath: Path to the saved model file
            
        Returns:
            Loaded model instance
        """
        try:
            model = joblib.load(filepath)
            logger.info(f"Model loaded from {filepath}")
            return model
        except Exception as e:
            logger.error(f"Failed to load model from {filepath}: {str(e)}")
            raise
    
    def save_best_model(self, X_val: pd.DataFrame, y_val: pd.Series,
                       min_accuracy: float = 0.8, use_tuned: bool = True) -> Tuple[str, str]:
        """
        Select and save the best performing model.
        
        Args:
            X_val: Validation feature matrix
            y_val: Validation target variable
            min_accuracy: Minimum accuracy threshold
            use_tuned: Whether to prefer tuned models
            
        Returns:
            Tuple of (model_name, saved_filepath)
        """
        best_name, best_model, best_accuracy, meets_threshold = self.select_best_model(
            X_val, y_val, min_accuracy, use_tuned
        )
        
        # Save the best model
        filepath = self.save_model(best_model, f"best_{best_name}")
        
        # Also save model metadata
        metadata = {
            'model_name': best_name,
            'accuracy': best_accuracy,
            'meets_threshold': meets_threshold,
            'threshold': min_accuracy,
            'tuned': use_tuned and best_name in self.tuned_models,
            'best_params': self.best_params.get(best_name, {}) if use_tuned else {}
        }
        
        metadata_path = filepath.replace('.joblib', '_metadata.joblib')
        joblib.dump(metadata, metadata_path)
        
        logger.info(f"Best model metadata saved to {metadata_path}")
        
        return best_name, filepath