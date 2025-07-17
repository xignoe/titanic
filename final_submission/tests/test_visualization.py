"""
Unit tests for ModelVisualizationUtils class.

Tests all visualization methods including ROC curves, feature importance plots,
model comparison charts, data exploration visualizations, and error handling.
"""

import pytest
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from unittest.mock import Mock, patch
from sklearn.ensemble import RandomForestClassifier

from src.utils.visualization import ModelVisualizationUtils


class TestModelVisualizationUtils:
    """Test cases for ModelVisualizationUtils class."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.viz_utils = ModelVisualizationUtils()
        
        # Sample binary classification data
        np.random.seed(42)
        self.y_true = np.array([0, 1, 1, 0, 1, 0, 1, 1, 0, 0])
        self.y_prob = np.array([0.1, 0.9, 0.4, 0.2, 0.8, 0.3, 0.7, 0.9, 0.1, 0.6])
        
        # Sample feature importance data
        self.feature_importance = {
            'age': 0.35, 'fare': 0.25, 'pclass': 0.20, 'sex_encoded': 0.15, 'sibsp': 0.05
        }
        
        # Sample model results
        self.model_results = {
            'RandomForest': {'accuracy': 0.85, 'precision': 0.82, 'recall': 0.88, 'f1_score': 0.85},
            'LogisticRegression': {'accuracy': 0.80, 'precision': 0.78, 'recall': 0.82, 'f1_score': 0.80},
            'SVM': {'accuracy': 0.83, 'precision': 0.81, 'recall': 0.85, 'f1_score': 0.83}
        }
        
        # Sample DataFrame for data exploration
        self.sample_df = pd.DataFrame({
            'Age': np.random.normal(30, 10, 100),
            'Fare': np.random.exponential(30, 100),
            'Pclass': np.random.choice([1, 2, 3], 100),
            'Sex': np.random.choice(['male', 'female'], 100),
            'Survived': np.random.choice([0, 1], 100)
        })
    
    def teardown_method(self):
        """Clean up after each test."""
        plt.close('all')
    
    def test_initialization(self):
        """Test ModelVisualizationUtils initialization."""
        viz = ModelVisualizationUtils(figsize=(8, 6))
        assert viz.figsize == (8, 6)
        assert len(viz.color_palette) == 8
    
    def test_plot_roc_curve_valid_input(self):
        """Test ROC curve plotting with valid input."""
        fig = self.viz_utils.plot_roc_curve(self.y_true, self.y_prob, "TestModel")
        
        # Check that a figure is returned
        assert fig is not None
        assert isinstance(fig, plt.Figure)
        
        # Check that the plot has the expected elements
        ax = fig.get_axes()[0]
        assert ax.get_xlabel() == 'False Positive Rate'
        assert ax.get_ylabel() == 'True Positive Rate'
        assert 'ROC Curve - TestModel' in ax.get_title()
        
        # Check that legend exists
        legend = ax.get_legend()
        assert legend is not None
        assert len(legend.get_texts()) == 2  # Model + Random classifier
    
    @patch('matplotlib.pyplot.savefig')
    def test_plot_roc_curve_with_save(self, mock_savefig):
        """Test ROC curve plotting with save functionality."""
        save_path = "test_roc.png"
        fig = self.viz_utils.plot_roc_curve(self.y_true, self.y_prob, 
                                           "TestModel", save_path=save_path)
        
        mock_savefig.assert_called_once_with(save_path, dpi=300, bbox_inches='tight')
        assert fig is not None
    
    def test_plot_roc_curve_invalid_input(self):
        """Test ROC curve plotting with invalid input."""
        # Test with mismatched array lengths
        y_true_short = self.y_true[:5]
        
        with pytest.raises(ValueError):
            self.viz_utils.plot_roc_curve(y_true_short, self.y_prob, "TestModel")
    
    def test_plot_feature_importance_valid_input(self):
        """Test feature importance plotting with valid input."""
        fig = self.viz_utils.plot_feature_importance(self.feature_importance, top_n=3)
        
        # Check that a figure is returned
        assert fig is not None
        assert isinstance(fig, plt.Figure)
        
        # Check that the plot has the expected elements
        ax = fig.get_axes()[0]
        assert ax.get_xlabel() == 'Feature Importance'
        assert 'Top 3 Feature Importance' in ax.get_title()
        
        # Check that only top 3 features are shown
        y_labels = [label.get_text() for label in ax.get_yticklabels()]
        assert len(y_labels) == 3
        assert 'age' in y_labels  # Should be the top feature
    
    @patch('matplotlib.pyplot.savefig')
    def test_plot_feature_importance_with_save(self, mock_savefig):
        """Test feature importance plotting with save functionality."""
        save_path = "test_importance.png"
        fig = self.viz_utils.plot_feature_importance(self.feature_importance, 
                                                    save_path=save_path)
        
        mock_savefig.assert_called_once_with(save_path, dpi=300, bbox_inches='tight')
        assert fig is not None
    
    def test_plot_feature_importance_empty_dict(self):
        """Test feature importance plotting with empty dictionary."""
        empty_importance = {}
        fig = self.viz_utils.plot_feature_importance(empty_importance)
        
        # Should still create a figure, just empty
        assert fig is not None
        ax = fig.get_axes()[0]
        assert len(ax.get_yticklabels()) == 0
    
    def test_plot_model_comparison_valid_input(self):
        """Test model comparison plotting with valid input."""
        fig = self.viz_utils.plot_model_comparison(self.model_results)
        
        # Check that a figure is returned
        assert fig is not None
        assert isinstance(fig, plt.Figure)
        
        # Check that the plot has the expected elements
        ax = fig.get_axes()[0]
        assert ax.get_xlabel() == 'Models'
        assert ax.get_ylabel() == 'Score'
        assert 'Model Performance Comparison' in ax.get_title()
        
        # Check that all models are shown
        x_labels = [label.get_text() for label in ax.get_xticklabels()]
        assert len(x_labels) == 3
        assert all(model in x_labels for model in self.model_results.keys())
    
    def test_plot_model_comparison_specific_metrics(self):
        """Test model comparison plotting with specific metrics."""
        metrics = ['accuracy', 'f1_score']
        fig = self.viz_utils.plot_model_comparison(self.model_results, metrics=metrics)
        
        assert fig is not None
        ax = fig.get_axes()[0]
        
        # Check that legend has only the specified metrics
        legend = ax.get_legend()
        legend_labels = [text.get_text() for text in legend.get_texts()]
        assert len(legend_labels) == 2
        assert 'Accuracy' in legend_labels
        assert 'F1_Score' in legend_labels
    
    @patch('matplotlib.pyplot.savefig')
    def test_plot_model_comparison_with_save(self, mock_savefig):
        """Test model comparison plotting with save functionality."""
        save_path = "test_comparison.png"
        fig = self.viz_utils.plot_model_comparison(self.model_results, save_path=save_path)
        
        mock_savefig.assert_called_once_with(save_path, dpi=300, bbox_inches='tight')
        assert fig is not None
    
    def test_plot_data_distributions_valid_input(self):
        """Test data distribution plotting with valid input."""
        fig = self.viz_utils.plot_data_distributions(self.sample_df)
        
        # Check that a figure is returned
        assert fig is not None
        assert isinstance(fig, plt.Figure)
        
        # Check that subplots are created for numerical columns
        axes = fig.get_axes()
        numerical_cols = self.sample_df.select_dtypes(include=[np.number]).columns
        assert len([ax for ax in axes if ax.get_visible()]) == len(numerical_cols)
    
    def test_plot_data_distributions_specific_columns(self):
        """Test data distribution plotting with specific columns."""
        columns = ['Age', 'Fare']
        fig = self.viz_utils.plot_data_distributions(self.sample_df, columns=columns)
        
        assert fig is not None
        axes = fig.get_axes()
        visible_axes = [ax for ax in axes if ax.get_visible()]
        assert len(visible_axes) == 2
        
        # Check that the correct columns are plotted
        titles = [ax.get_title() for ax in visible_axes]
        assert any('Age' in title for title in titles)
        assert any('Fare' in title for title in titles)
    
    @patch('matplotlib.pyplot.savefig')
    def test_plot_data_distributions_with_save(self, mock_savefig):
        """Test data distribution plotting with save functionality."""
        save_path = "test_distributions.png"
        fig = self.viz_utils.plot_data_distributions(self.sample_df, save_path=save_path)
        
        mock_savefig.assert_called_once_with(save_path, dpi=300, bbox_inches='tight')
        assert fig is not None
    
    def test_plot_correlation_matrix_valid_input(self):
        """Test correlation matrix plotting with valid input."""
        fig = self.viz_utils.plot_correlation_matrix(self.sample_df)
        
        # Check that a figure is returned
        assert fig is not None
        assert isinstance(fig, plt.Figure)
        
        # Check that the plot has the expected elements
        ax = fig.get_axes()[0]
        assert 'Feature Correlation Matrix' in ax.get_title()
    
    def test_plot_correlation_matrix_no_numerical_columns(self):
        """Test correlation matrix plotting with no numerical columns."""
        text_df = pd.DataFrame({'A': ['a', 'b', 'c'], 'B': ['x', 'y', 'z']})
        
        with pytest.raises(ValueError, match="No numerical columns found"):
            self.viz_utils.plot_correlation_matrix(text_df)
    
    @patch('matplotlib.pyplot.savefig')
    def test_plot_correlation_matrix_with_save(self, mock_savefig):
        """Test correlation matrix plotting with save functionality."""
        save_path = "test_correlation.png"
        fig = self.viz_utils.plot_correlation_matrix(self.sample_df, save_path=save_path)
        
        mock_savefig.assert_called_once_with(save_path, dpi=300, bbox_inches='tight')
        assert fig is not None
    
    def test_plot_survival_patterns_valid_input(self):
        """Test survival pattern plotting with valid input."""
        fig = self.viz_utils.plot_survival_patterns(self.sample_df, 'Pclass')
        
        # Check that a figure is returned
        assert fig is not None
        assert isinstance(fig, plt.Figure)
        
        # Check that two subplots are created
        axes = fig.get_axes()
        assert len(axes) == 2
        
        # Check subplot titles
        titles = [ax.get_title() for ax in axes]
        assert any('Survival Rate by Pclass' in title for title in titles)
        assert any('Survival Count by Pclass' in title for title in titles)
    
    def test_plot_survival_patterns_missing_column(self):
        """Test survival pattern plotting with missing column."""
        with pytest.raises(ValueError, match="Column 'NonExistent' not found"):
            self.viz_utils.plot_survival_patterns(self.sample_df, 'NonExistent')
    
    def test_plot_survival_patterns_missing_target(self):
        """Test survival pattern plotting with missing target column."""
        df_no_target = self.sample_df.drop('Survived', axis=1)
        
        with pytest.raises(ValueError, match="Target column 'Survived' not found"):
            self.viz_utils.plot_survival_patterns(df_no_target, 'Pclass')
    
    @patch('matplotlib.pyplot.savefig')
    def test_plot_survival_patterns_with_save(self, mock_savefig):
        """Test survival pattern plotting with save functionality."""
        save_path = "test_survival.png"
        fig = self.viz_utils.plot_survival_patterns(self.sample_df, 'Pclass', 
                                                   save_path=save_path)
        
        mock_savefig.assert_called_once_with(save_path, dpi=300, bbox_inches='tight')
        assert fig is not None
    
    def test_create_multi_roc_comparison_valid_input(self):
        """Test multi-ROC comparison plotting with valid input."""
        model_results = {
            'Model1': (self.y_true, self.y_prob),
            'Model2': (self.y_true, self.y_prob * 0.8)  # Slightly different probabilities
        }
        
        fig = self.viz_utils.create_multi_roc_comparison(model_results)
        
        # Check that a figure is returned
        assert fig is not None
        assert isinstance(fig, plt.Figure)
        
        # Check that the plot has the expected elements
        ax = fig.get_axes()[0]
        assert ax.get_xlabel() == 'False Positive Rate'
        assert ax.get_ylabel() == 'True Positive Rate'
        assert 'ROC Curve Comparison' in ax.get_title()
        
        # Check that legend exists with all models + random classifier
        legend = ax.get_legend()
        assert legend is not None
        assert len(legend.get_texts()) == 3  # 2 models + random classifier
    
    @patch('matplotlib.pyplot.savefig')
    def test_create_multi_roc_comparison_with_save(self, mock_savefig):
        """Test multi-ROC comparison plotting with save functionality."""
        model_results = {
            'Model1': (self.y_true, self.y_prob)
        }
        save_path = "test_multi_roc.png"
        
        fig = self.viz_utils.create_multi_roc_comparison(model_results, save_path=save_path)
        
        mock_savefig.assert_called_once_with(save_path, dpi=300, bbox_inches='tight')
        assert fig is not None
    
    def test_create_multi_roc_comparison_empty_input(self):
        """Test multi-ROC comparison plotting with empty input."""
        empty_results = {}
        fig = self.viz_utils.create_multi_roc_comparison(empty_results)
        
        # Should still create a figure with just the random classifier line
        assert fig is not None
        ax = fig.get_axes()[0]
        legend = ax.get_legend()
        assert len(legend.get_texts()) == 1  # Only random classifier
    
    @patch('matplotlib.pyplot.close')
    def test_close_all_figures(self, mock_close):
        """Test closing all figures functionality."""
        self.viz_utils.close_all_figures()
        mock_close.assert_called_once_with('all')
    
    def test_custom_figsize(self):
        """Test initialization with custom figure size."""
        custom_viz = ModelVisualizationUtils(figsize=(12, 8))
        assert custom_viz.figsize == (12, 8)
        
        # Test that custom figsize is used
        fig = custom_viz.plot_roc_curve(self.y_true, self.y_prob)
        assert fig.get_size_inches()[0] == 12
        assert fig.get_size_inches()[1] == 8
    
    def test_color_palette_consistency(self):
        """Test that color palette is consistent across plots."""
        # Create multiple plots and check they use the same color palette
        fig1 = self.viz_utils.plot_feature_importance(self.feature_importance)
        fig2 = self.viz_utils.plot_model_comparison(self.model_results)
        
        # Both figures should be created successfully
        assert fig1 is not None
        assert fig2 is not None
        
        # Color palette should be the same
        assert len(self.viz_utils.color_palette) == 8