"""
Unit tests for ModelEvaluator class.

Tests all evaluation methods including metrics calculation, confusion matrix generation,
feature importance analysis, and error handling.
"""

import pytest
import numpy as np
import pandas as pd
from unittest.mock import Mock, patch
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression

from src.models.evaluator import ModelEvaluator


class TestModelEvaluator:
    """Test cases for ModelEvaluator class."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.evaluator = ModelEvaluator()
        
        # Sample binary classification data
        self.y_true = np.array([0, 1, 1, 0, 1, 0, 1, 1, 0, 0])
        self.y_pred = np.array([0, 1, 0, 0, 1, 0, 1, 1, 0, 1])
        self.y_prob = np.array([0.1, 0.9, 0.4, 0.2, 0.8, 0.3, 0.7, 0.9, 0.1, 0.6])
        
        # Feature names for testing
        self.feature_names = ['age', 'fare', 'pclass', 'sex_encoded']
    
    def test_calculate_metrics_valid_input(self):
        """Test metrics calculation with valid binary input."""
        metrics = self.evaluator.calculate_metrics(self.y_true, self.y_pred)
        
        # Check that all required metrics are present
        required_metrics = ['accuracy', 'precision', 'recall', 'f1_score']
        assert all(metric in metrics for metric in required_metrics)
        
        # Check that all metrics are between 0 and 1
        assert all(0 <= value <= 1 for value in metrics.values())
        
        # Check specific values for our test data
        assert metrics['accuracy'] == 0.8  # 8 correct out of 10
        assert len(self.evaluator.metrics_history) == 1
    
    def test_calculate_metrics_perfect_prediction(self):
        """Test metrics calculation with perfect predictions."""
        y_true = np.array([0, 1, 1, 0])
        y_pred = np.array([0, 1, 1, 0])
        
        metrics = self.evaluator.calculate_metrics(y_true, y_pred)
        
        assert metrics['accuracy'] == 1.0
        assert metrics['precision'] == 1.0
        assert metrics['recall'] == 1.0
        assert metrics['f1_score'] == 1.0
    
    def test_calculate_metrics_all_wrong(self):
        """Test metrics calculation with all wrong predictions."""
        y_true = np.array([0, 1, 1, 0])
        y_pred = np.array([1, 0, 0, 1])
        
        metrics = self.evaluator.calculate_metrics(y_true, y_pred)
        
        assert metrics['accuracy'] == 0.0
        assert metrics['precision'] == 0.0
        assert metrics['recall'] == 0.0
        assert metrics['f1_score'] == 0.0
    
    def test_calculate_metrics_mismatched_length(self):
        """Test error handling for mismatched array lengths."""
        y_true = np.array([0, 1, 1])
        y_pred = np.array([0, 1])
        
        with pytest.raises(ValueError, match="must have the same length"):
            self.evaluator.calculate_metrics(y_true, y_pred)
    
    def test_calculate_metrics_invalid_labels(self):
        """Test error handling for non-binary labels."""
        y_true = np.array([0, 1, 2, 0])
        y_pred = np.array([0, 1, 1, 0])
        
        with pytest.raises(ValueError, match="Labels must be binary"):
            self.evaluator.calculate_metrics(y_true, y_pred)
    
    def test_generate_confusion_matrix_valid_input(self):
        """Test confusion matrix generation with valid input."""
        cm = self.evaluator.generate_confusion_matrix(self.y_true, self.y_pred)
        
        # Check shape and type
        assert cm.shape == (2, 2)
        assert isinstance(cm, np.ndarray)
        
        # Check that values sum to total predictions
        assert cm.sum() == len(self.y_true)
        
        # Check specific values for our test data
        # True negatives: positions where both y_true and y_pred are 0
        # True positives: positions where both y_true and y_pred are 1
        expected_tn = sum((self.y_true == 0) & (self.y_pred == 0))
        expected_tp = sum((self.y_true == 1) & (self.y_pred == 1))
        
        assert cm[0, 0] == expected_tn  # True negatives
        assert cm[1, 1] == expected_tp  # True positives
    
    def test_generate_confusion_matrix_mismatched_length(self):
        """Test error handling for mismatched array lengths in confusion matrix."""
        y_true = np.array([0, 1, 1])
        y_pred = np.array([0, 1])
        
        with pytest.raises(ValueError, match="must have the same length"):
            self.evaluator.generate_confusion_matrix(y_true, y_pred)
    
    @patch('matplotlib.pyplot.savefig')
    def test_visualize_confusion_matrix(self, mock_savefig):
        """Test confusion matrix visualization."""
        fig = self.evaluator.visualize_confusion_matrix(self.y_true, self.y_pred)
        
        # Check that a figure is returned
        assert fig is not None
        
        # Test with save path
        save_path = "test_cm.png"
        fig = self.evaluator.visualize_confusion_matrix(
            self.y_true, self.y_pred, save_path=save_path
        )
        
        mock_savefig.assert_called_once_with(save_path, dpi=300, bbox_inches='tight')
    
    def test_analyze_feature_importance_random_forest(self):
        """Test feature importance analysis with RandomForest model."""
        # Create a mock RandomForest model
        model = Mock(spec=RandomForestClassifier)
        model.feature_importances_ = np.array([0.4, 0.3, 0.2, 0.1])
        
        importance = self.evaluator.analyze_feature_importance(model, self.feature_names)
        
        # Check that importance is sorted in descending order
        importance_values = list(importance.values())
        assert importance_values == sorted(importance_values, reverse=True)
        
        # Check that all features are included
        assert len(importance) == len(self.feature_names)
        assert all(name in importance for name in self.feature_names)
        
        # Check that the most important feature is first
        assert list(importance.keys())[0] == 'age'  # 0.4 is highest
    
    def test_analyze_feature_importance_no_attribute(self):
        """Test error handling for model without feature importance."""
        model = Mock(spec=LogisticRegression)
        # LogisticRegression mock without feature_importances_
        
        with pytest.raises(ValueError, match="does not have feature_importances_"):
            self.evaluator.analyze_feature_importance(model, self.feature_names)
    
    def test_analyze_feature_importance_mismatched_features(self):
        """Test error handling for mismatched feature count."""
        model = Mock(spec=RandomForestClassifier)
        model.feature_importances_ = np.array([0.4, 0.3, 0.2])  # Only 3 features
        
        with pytest.raises(ValueError, match="Number of feature names must match"):
            self.evaluator.analyze_feature_importance(model, self.feature_names)  # 4 features
    
    def test_get_top_features(self):
        """Test getting top N features from importance dictionary."""
        importance = {'age': 0.4, 'fare': 0.3, 'pclass': 0.2, 'sex_encoded': 0.1}
        
        top_3 = self.evaluator.get_top_features(importance, n=3)
        
        assert len(top_3) == 3
        assert top_3 == ['age', 'fare', 'pclass']
    
    def test_generate_classification_report(self):
        """Test classification report generation."""
        report = self.evaluator.generate_classification_report(self.y_true, self.y_pred)
        
        assert isinstance(report, str)
        assert 'precision' in report
        assert 'recall' in report
        assert 'f1-score' in report
        assert 'Died' in report
        assert 'Survived' in report
    
    def test_calculate_roc_metrics(self):
        """Test ROC curve metrics calculation."""
        roc_metrics = self.evaluator.calculate_roc_metrics(self.y_true, self.y_prob)
        
        required_keys = ['fpr', 'tpr', 'thresholds', 'auc']
        assert all(key in roc_metrics for key in required_keys)
        
        # Check AUC is between 0 and 1
        assert 0 <= roc_metrics['auc'] <= 1
        
        # Check array lengths
        assert len(roc_metrics['fpr']) == len(roc_metrics['tpr'])
        assert len(roc_metrics['fpr']) == len(roc_metrics['thresholds'])
    
    def test_compare_models(self):
        """Test model comparison functionality."""
        model_results = {
            'RandomForest': {'accuracy': 0.85, 'precision': 0.82, 'recall': 0.88, 'f1_score': 0.85},
            'LogisticRegression': {'accuracy': 0.80, 'precision': 0.78, 'recall': 0.82, 'f1_score': 0.80},
            'SVM': {'accuracy': 0.83, 'precision': 0.81, 'recall': 0.85, 'f1_score': 0.83}
        }
        
        comparison_df = self.evaluator.compare_models(model_results)
        
        # Check that it's a DataFrame
        assert isinstance(comparison_df, pd.DataFrame)
        
        # Check that models are sorted by accuracy (descending)
        assert comparison_df.index[0] == 'RandomForest'  # Highest accuracy
        assert comparison_df.index[-1] == 'LogisticRegression'  # Lowest accuracy
        
        # Check all metrics are present
        expected_columns = ['accuracy', 'precision', 'recall', 'f1_score']
        assert all(col in comparison_df.columns for col in expected_columns)
    
    def test_get_metrics_summary_empty(self):
        """Test metrics summary when no metrics calculated."""
        summary = self.evaluator.get_metrics_summary()
        
        assert "message" in summary
        assert "No metrics calculated yet" in summary["message"]
    
    def test_get_metrics_summary_with_data(self):
        """Test metrics summary with calculated metrics."""
        # Calculate some metrics first
        self.evaluator.calculate_metrics(self.y_true, self.y_pred)
        
        # Add some feature importance
        model = Mock(spec=RandomForestClassifier)
        model.feature_importances_ = np.array([0.4, 0.3, 0.2, 0.1])
        self.evaluator.analyze_feature_importance(model, self.feature_names)
        
        summary = self.evaluator.get_metrics_summary()
        
        assert summary["total_evaluations"] == 1
        assert "latest_metrics" in summary
        assert "cached_feature_importance" in summary
        assert len(summary["cached_feature_importance"]) == 1
    
    def test_metrics_history_tracking(self):
        """Test that metrics history is properly tracked."""
        # Calculate metrics multiple times
        self.evaluator.calculate_metrics(self.y_true, self.y_pred)
        self.evaluator.calculate_metrics(self.y_true[:5], self.y_pred[:5])
        
        assert len(self.evaluator.metrics_history) == 2
        
        # Check that both metric sets are stored
        assert all('accuracy' in metrics for metrics in self.evaluator.metrics_history)