"""
Model evaluation and performance analysis utilities.

This module provides comprehensive model evaluation capabilities including
performance metrics calculation, confusion matrix generation, and feature
importance analysis.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report, roc_curve, auc
)
from typing import Dict, List, Tuple, Any, Optional
import logging

logger = logging.getLogger(__name__)


class ModelEvaluator:
    """
    Comprehensive model evaluation and analysis class.
    
    Provides methods to calculate performance metrics, generate confusion matrices,
    analyze feature importance, and create visualizations for model assessment.
    """
    
    def __init__(self):
        """Initialize the ModelEvaluator."""
        self.metrics_history = []
        self.feature_importance_cache = {}
    
    def calculate_metrics(self, y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
        """
        Calculate comprehensive performance metrics.
        
        Args:
            y_true: True labels
            y_pred: Predicted labels
            
        Returns:
            Dictionary containing accuracy, precision, recall, and F1-score
            
        Raises:
            ValueError: If input arrays have different lengths or invalid values
        """
        if len(y_true) != len(y_pred):
            raise ValueError("y_true and y_pred must have the same length")
        
        if not all(val in [0, 1] for val in np.unique(np.concatenate([y_true, y_pred]))):
            raise ValueError("Labels must be binary (0 or 1)")
        
        try:
            metrics = {
                'accuracy': accuracy_score(y_true, y_pred),
                'precision': precision_score(y_true, y_pred, zero_division=0),
                'recall': recall_score(y_true, y_pred, zero_division=0),
                'f1_score': f1_score(y_true, y_pred, zero_division=0)
            }
            
            # Store metrics for history tracking
            self.metrics_history.append(metrics.copy())
            
            logger.info(f"Calculated metrics: {metrics}")
            return metrics
            
        except Exception as e:
            logger.error(f"Error calculating metrics: {str(e)}")
            raise
    
    def generate_confusion_matrix(self, y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
        """
        Generate confusion matrix for binary classification.
        
        Args:
            y_true: True labels
            y_pred: Predicted labels
            
        Returns:
            2x2 confusion matrix as numpy array
            
        Raises:
            ValueError: If input arrays have different lengths
        """
        if len(y_true) != len(y_pred):
            raise ValueError("y_true and y_pred must have the same length")
        
        try:
            cm = confusion_matrix(y_true, y_pred)
            logger.info(f"Generated confusion matrix:\n{cm}")
            return cm
            
        except Exception as e:
            logger.error(f"Error generating confusion matrix: {str(e)}")
            raise
    
    def visualize_confusion_matrix(self, y_true: np.ndarray, y_pred: np.ndarray, 
                                 save_path: Optional[str] = None) -> plt.Figure:
        """
        Create a visualization of the confusion matrix.
        
        Args:
            y_true: True labels
            y_pred: Predicted labels
            save_path: Optional path to save the plot
            
        Returns:
            Matplotlib figure object
        """
        cm = self.generate_confusion_matrix(y_true, y_pred)
        
        fig, ax = plt.subplots(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                   xticklabels=['Died', 'Survived'],
                   yticklabels=['Died', 'Survived'],
                   ax=ax)
        
        ax.set_title('Confusion Matrix')
        ax.set_xlabel('Predicted')
        ax.set_ylabel('Actual')
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Confusion matrix saved to {save_path}")
        
        return fig
    
    def analyze_feature_importance(self, model: Any, feature_names: List[str]) -> Dict[str, float]:
        """
        Analyze and rank feature importance from trained model.
        
        Args:
            model: Trained scikit-learn model with feature_importances_ attribute
            feature_names: List of feature names corresponding to model features
            
        Returns:
            Dictionary mapping feature names to importance scores, sorted by importance
            
        Raises:
            ValueError: If model doesn't have feature importance or mismatched feature count
        """
        if not hasattr(model, 'feature_importances_'):
            raise ValueError("Model does not have feature_importances_ attribute")
        
        if len(feature_names) != len(model.feature_importances_):
            raise ValueError("Number of feature names must match model features")
        
        try:
            # Get feature importances and create sorted dictionary
            importances = model.feature_importances_
            feature_importance = dict(zip(feature_names, importances))
            
            # Sort by importance (descending)
            sorted_importance = dict(sorted(feature_importance.items(), 
                                          key=lambda x: x[1], reverse=True))
            
            # Cache for later use
            model_name = type(model).__name__
            self.feature_importance_cache[model_name] = sorted_importance
            
            logger.info(f"Analyzed feature importance for {model_name}")
            logger.info(f"Top 5 features: {list(sorted_importance.keys())[:5]}")
            
            return sorted_importance
            
        except Exception as e:
            logger.error(f"Error analyzing feature importance: {str(e)}")
            raise
    
    def get_top_features(self, feature_importance: Dict[str, float], n: int = 5) -> List[str]:
        """
        Get the top N most important features.
        
        Args:
            feature_importance: Dictionary of feature importance scores
            n: Number of top features to return
            
        Returns:
            List of top N feature names
        """
        return list(feature_importance.keys())[:n]
    
    def generate_classification_report(self, y_true: np.ndarray, y_pred: np.ndarray) -> str:
        """
        Generate detailed classification report.
        
        Args:
            y_true: True labels
            y_pred: Predicted labels
            
        Returns:
            Formatted classification report string
        """
        try:
            report = classification_report(y_true, y_pred, 
                                         target_names=['Died', 'Survived'])
            logger.info("Generated classification report")
            return report
            
        except Exception as e:
            logger.error(f"Error generating classification report: {str(e)}")
            raise
    
    def calculate_roc_metrics(self, y_true: np.ndarray, y_prob: np.ndarray) -> Dict[str, Any]:
        """
        Calculate ROC curve metrics and AUC score.
        
        Args:
            y_true: True labels
            y_prob: Predicted probabilities for positive class
            
        Returns:
            Dictionary containing fpr, tpr, thresholds, and auc score
        """
        try:
            fpr, tpr, thresholds = roc_curve(y_true, y_prob)
            auc_score = auc(fpr, tpr)
            
            roc_metrics = {
                'fpr': fpr,
                'tpr': tpr,
                'thresholds': thresholds,
                'auc': auc_score
            }
            
            logger.info(f"Calculated ROC AUC: {auc_score:.4f}")
            return roc_metrics
            
        except Exception as e:
            logger.error(f"Error calculating ROC metrics: {str(e)}")
            raise
    
    def compare_models(self, model_results: Dict[str, Dict[str, float]]) -> pd.DataFrame:
        """
        Compare multiple models' performance metrics.
        
        Args:
            model_results: Dictionary mapping model names to their metrics
            
        Returns:
            DataFrame with models as rows and metrics as columns
        """
        try:
            comparison_df = pd.DataFrame(model_results).T
            comparison_df = comparison_df.round(4)
            
            # Sort by accuracy (or F1 score if accuracy is tied)
            comparison_df = comparison_df.sort_values(['accuracy', 'f1_score'], 
                                                    ascending=False)
            
            logger.info("Generated model comparison table")
            return comparison_df
            
        except Exception as e:
            logger.error(f"Error comparing models: {str(e)}")
            raise
    
    def get_metrics_summary(self) -> Dict[str, Any]:
        """
        Get summary of all calculated metrics.
        
        Returns:
            Dictionary containing metrics history and statistics
        """
        if not self.metrics_history:
            return {"message": "No metrics calculated yet"}
        
        summary = {
            "total_evaluations": len(self.metrics_history),
            "latest_metrics": self.metrics_history[-1],
            "cached_feature_importance": list(self.feature_importance_cache.keys())
        }
        
        return summary