"""
Visualization utilities for model analysis and data exploration.

This module provides comprehensive visualization capabilities including
performance plots, feature importance charts, data exploration visualizations,
and model comparison charts.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import roc_curve, auc
from typing import Dict, List, Tuple, Any, Optional, Union
import logging

logger = logging.getLogger(__name__)

# Set style for consistent plots
plt.style.use('default')
sns.set_palette("husl")


class ModelVisualizationUtils:
    """
    Comprehensive visualization utilities for model analysis and data exploration.
    
    Provides methods to create performance plots, feature importance charts,
    data exploration visualizations, and model comparison charts.
    """
    
    def __init__(self, figsize: Tuple[int, int] = (10, 6)):
        """
        Initialize the visualization utilities.
        
        Args:
            figsize: Default figure size for plots
        """
        self.figsize = figsize
        self.color_palette = sns.color_palette("husl", 8)
    
    def plot_roc_curve(self, y_true: np.ndarray, y_prob: np.ndarray, 
                      model_name: str = "Model", save_path: Optional[str] = None) -> plt.Figure:
        """
        Create ROC curve visualization.
        
        Args:
            y_true: True labels
            y_prob: Predicted probabilities for positive class
            model_name: Name of the model for the plot title
            save_path: Optional path to save the plot
            
        Returns:
            Matplotlib figure object
        """
        try:
            fpr, tpr, _ = roc_curve(y_true, y_prob)
            auc_score = auc(fpr, tpr)
            
            fig, ax = plt.subplots(figsize=self.figsize)
            
            # Plot ROC curve
            ax.plot(fpr, tpr, color=self.color_palette[0], lw=2,
                   label=f'{model_name} (AUC = {auc_score:.3f})')
            
            # Plot diagonal line (random classifier)
            ax.plot([0, 1], [0, 1], color='gray', lw=1, linestyle='--',
                   label='Random Classifier (AUC = 0.500)')
            
            ax.set_xlim([0.0, 1.0])
            ax.set_ylim([0.0, 1.05])
            ax.set_xlabel('False Positive Rate')
            ax.set_ylabel('True Positive Rate')
            ax.set_title(f'ROC Curve - {model_name}')
            ax.legend(loc="lower right")
            ax.grid(True, alpha=0.3)
            
            plt.tight_layout()
            
            if save_path:
                plt.savefig(save_path, dpi=300, bbox_inches='tight')
                logger.info(f"ROC curve saved to {save_path}")
            
            logger.info(f"Created ROC curve for {model_name} with AUC: {auc_score:.3f}")
            return fig
            
        except Exception as e:
            logger.error(f"Error creating ROC curve: {str(e)}")
            raise
    
    def plot_feature_importance(self, feature_importance: Dict[str, float], 
                               top_n: int = 10, save_path: Optional[str] = None) -> plt.Figure:
        """
        Create feature importance bar chart.
        
        Args:
            feature_importance: Dictionary mapping feature names to importance scores
            top_n: Number of top features to display
            save_path: Optional path to save the plot
            
        Returns:
            Matplotlib figure object
        """
        try:
            # Get top N features
            top_features = dict(list(feature_importance.items())[:top_n])
            
            fig, ax = plt.subplots(figsize=(12, 8))
            
            features = list(top_features.keys())
            importances = list(top_features.values())
            
            # Create horizontal bar chart
            bars = ax.barh(range(len(features)), importances, 
                          color=self.color_palette[:len(features)])
            
            ax.set_yticks(range(len(features)))
            ax.set_yticklabels(features)
            ax.set_xlabel('Feature Importance')
            ax.set_title(f'Top {len(features)} Feature Importance')
            ax.grid(True, alpha=0.3, axis='x')
            
            # Add value labels on bars
            for i, (bar, importance) in enumerate(zip(bars, importances)):
                ax.text(importance + 0.001, i, f'{importance:.3f}', 
                       va='center', ha='left', fontsize=9)
            
            plt.tight_layout()
            
            if save_path:
                plt.savefig(save_path, dpi=300, bbox_inches='tight')
                logger.info(f"Feature importance plot saved to {save_path}")
            
            logger.info(f"Created feature importance plot with {len(features)} features")
            return fig
            
        except Exception as e:
            logger.error(f"Error creating feature importance plot: {str(e)}")
            raise
    
    def plot_model_comparison(self, model_results: Dict[str, Dict[str, float]], 
                             metrics: List[str] = None, save_path: Optional[str] = None) -> plt.Figure:
        """
        Create model comparison chart.
        
        Args:
            model_results: Dictionary mapping model names to their metrics
            metrics: List of metrics to compare (default: all available)
            save_path: Optional path to save the plot
            
        Returns:
            Matplotlib figure object
        """
        try:
            df = pd.DataFrame(model_results).T
            
            if metrics is None:
                metrics = df.columns.tolist()
            
            # Filter to requested metrics
            df = df[metrics]
            
            fig, ax = plt.subplots(figsize=(12, 8))
            
            # Create grouped bar chart
            x = np.arange(len(df.index))
            width = 0.8 / len(metrics)
            
            for i, metric in enumerate(metrics):
                offset = (i - len(metrics)/2 + 0.5) * width
                bars = ax.bar(x + offset, df[metric], width, 
                             label=metric.title(), color=self.color_palette[i])
                
                # Add value labels on bars
                for bar in bars:
                    height = bar.get_height()
                    ax.text(bar.get_x() + bar.get_width()/2., height + 0.005,
                           f'{height:.3f}', ha='center', va='bottom', fontsize=8)
            
            ax.set_xlabel('Models')
            ax.set_ylabel('Score')
            ax.set_title('Model Performance Comparison')
            ax.set_xticks(x)
            ax.set_xticklabels(df.index, rotation=45, ha='right')
            ax.legend()
            ax.grid(True, alpha=0.3, axis='y')
            ax.set_ylim(0, 1.1)
            
            plt.tight_layout()
            
            if save_path:
                plt.savefig(save_path, dpi=300, bbox_inches='tight')
                logger.info(f"Model comparison plot saved to {save_path}")
            
            logger.info(f"Created model comparison plot for {len(df)} models")
            return fig
            
        except Exception as e:
            logger.error(f"Error creating model comparison plot: {str(e)}")
            raise
    
    def plot_data_distributions(self, df: pd.DataFrame, columns: List[str] = None, 
                               save_path: Optional[str] = None) -> plt.Figure:
        """
        Create distribution plots for numerical features.
        
        Args:
            df: DataFrame containing the data
            columns: List of columns to plot (default: all numerical columns)
            save_path: Optional path to save the plot
            
        Returns:
            Matplotlib figure object
        """
        try:
            if columns is None:
                columns = df.select_dtypes(include=[np.number]).columns.tolist()
            
            n_cols = min(3, len(columns))
            n_rows = (len(columns) + n_cols - 1) // n_cols
            
            fig, axes = plt.subplots(n_rows, n_cols, figsize=(5*n_cols, 4*n_rows))
            if n_rows == 1 and n_cols == 1:
                axes = [axes]
            elif n_rows == 1 or n_cols == 1:
                axes = axes.flatten()
            else:
                axes = axes.flatten()
            
            for i, col in enumerate(columns):
                if i < len(axes):
                    ax = axes[i]
                    
                    # Create histogram with KDE
                    ax.hist(df[col].dropna(), bins=30, alpha=0.7, 
                           color=self.color_palette[i % len(self.color_palette)], 
                           density=True, edgecolor='black', linewidth=0.5)
                    
                    # Add KDE curve
                    try:
                        df[col].dropna().plot.kde(ax=ax, color='red', linewidth=2)
                    except:
                        pass  # Skip KDE if not possible
                    
                    ax.set_title(f'Distribution of {col}')
                    ax.set_xlabel(col)
                    ax.set_ylabel('Density')
                    ax.grid(True, alpha=0.3)
            
            # Hide unused subplots
            for i in range(len(columns), len(axes)):
                axes[i].set_visible(False)
            
            plt.tight_layout()
            
            if save_path:
                plt.savefig(save_path, dpi=300, bbox_inches='tight')
                logger.info(f"Distribution plots saved to {save_path}")
            
            logger.info(f"Created distribution plots for {len(columns)} features")
            return fig
            
        except Exception as e:
            logger.error(f"Error creating distribution plots: {str(e)}")
            raise
    
    def plot_correlation_matrix(self, df: pd.DataFrame, save_path: Optional[str] = None) -> plt.Figure:
        """
        Create correlation matrix heatmap.
        
        Args:
            df: DataFrame containing the data
            save_path: Optional path to save the plot
            
        Returns:
            Matplotlib figure object
        """
        try:
            # Select only numerical columns
            numerical_df = df.select_dtypes(include=[np.number])
            
            if numerical_df.empty:
                raise ValueError("No numerical columns found for correlation analysis")
            
            correlation_matrix = numerical_df.corr()
            
            fig, ax = plt.subplots(figsize=(12, 10))
            
            # Create heatmap
            mask = np.triu(np.ones_like(correlation_matrix, dtype=bool))
            sns.heatmap(correlation_matrix, mask=mask, annot=True, cmap='coolwarm',
                       center=0, square=True, linewidths=0.5, cbar_kws={"shrink": .8},
                       ax=ax, fmt='.2f')
            
            ax.set_title('Feature Correlation Matrix')
            plt.tight_layout()
            
            if save_path:
                plt.savefig(save_path, dpi=300, bbox_inches='tight')
                logger.info(f"Correlation matrix saved to {save_path}")
            
            logger.info(f"Created correlation matrix for {len(numerical_df.columns)} features")
            return fig
            
        except Exception as e:
            logger.error(f"Error creating correlation matrix: {str(e)}")
            raise
    
    def plot_survival_patterns(self, df: pd.DataFrame, feature_col: str, 
                              target_col: str = 'Survived', save_path: Optional[str] = None) -> plt.Figure:
        """
        Create survival pattern visualization by feature groups.
        
        Args:
            df: DataFrame containing the data
            feature_col: Column name to group by
            target_col: Target column name (default: 'Survived')
            save_path: Optional path to save the plot
            
        Returns:
            Matplotlib figure object
        """
        try:
            if feature_col not in df.columns:
                raise ValueError(f"Column '{feature_col}' not found in DataFrame")
            
            if target_col not in df.columns:
                raise ValueError(f"Target column '{target_col}' not found in DataFrame")
            
            # Calculate survival rates by feature groups
            survival_rates = df.groupby(feature_col)[target_col].agg(['count', 'sum', 'mean']).reset_index()
            survival_rates.columns = [feature_col, 'Total', 'Survived', 'Survival_Rate']
            
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
            
            # Plot 1: Survival rates by group
            bars1 = ax1.bar(survival_rates[feature_col], survival_rates['Survival_Rate'],
                           color=self.color_palette[:len(survival_rates)])
            
            ax1.set_title(f'Survival Rate by {feature_col}')
            ax1.set_xlabel(feature_col)
            ax1.set_ylabel('Survival Rate')
            ax1.set_ylim(0, 1)
            ax1.grid(True, alpha=0.3, axis='y')
            
            # Add value labels on bars
            for bar, rate in zip(bars1, survival_rates['Survival_Rate']):
                ax1.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.01,
                        f'{rate:.2%}', ha='center', va='bottom')
            
            # Plot 2: Count of survivors vs non-survivors
            x_pos = np.arange(len(survival_rates))
            width = 0.35
            
            bars2 = ax2.bar(x_pos - width/2, survival_rates['Total'] - survival_rates['Survived'],
                           width, label='Did not survive', color=self.color_palette[0], alpha=0.7)
            bars3 = ax2.bar(x_pos + width/2, survival_rates['Survived'],
                           width, label='Survived', color=self.color_palette[1], alpha=0.7)
            
            ax2.set_title(f'Survival Count by {feature_col}')
            ax2.set_xlabel(feature_col)
            ax2.set_ylabel('Count')
            ax2.set_xticks(x_pos)
            ax2.set_xticklabels(survival_rates[feature_col])
            ax2.legend()
            ax2.grid(True, alpha=0.3, axis='y')
            
            plt.tight_layout()
            
            if save_path:
                plt.savefig(save_path, dpi=300, bbox_inches='tight')
                logger.info(f"Survival pattern plot saved to {save_path}")
            
            logger.info(f"Created survival pattern plot for {feature_col}")
            return fig
            
        except Exception as e:
            logger.error(f"Error creating survival pattern plot: {str(e)}")
            raise
    
    def create_multi_roc_comparison(self, model_results: Dict[str, Tuple[np.ndarray, np.ndarray]], 
                                   save_path: Optional[str] = None) -> plt.Figure:
        """
        Create ROC curve comparison for multiple models.
        
        Args:
            model_results: Dictionary mapping model names to (y_true, y_prob) tuples
            save_path: Optional path to save the plot
            
        Returns:
            Matplotlib figure object
        """
        try:
            fig, ax = plt.subplots(figsize=self.figsize)
            
            for i, (model_name, (y_true, y_prob)) in enumerate(model_results.items()):
                fpr, tpr, _ = roc_curve(y_true, y_prob)
                auc_score = auc(fpr, tpr)
                
                ax.plot(fpr, tpr, color=self.color_palette[i % len(self.color_palette)], 
                       lw=2, label=f'{model_name} (AUC = {auc_score:.3f})')
            
            # Plot diagonal line (random classifier)
            ax.plot([0, 1], [0, 1], color='gray', lw=1, linestyle='--',
                   label='Random Classifier (AUC = 0.500)')
            
            ax.set_xlim([0.0, 1.0])
            ax.set_ylim([0.0, 1.05])
            ax.set_xlabel('False Positive Rate')
            ax.set_ylabel('True Positive Rate')
            ax.set_title('ROC Curve Comparison')
            ax.legend(loc="lower right")
            ax.grid(True, alpha=0.3)
            
            plt.tight_layout()
            
            if save_path:
                plt.savefig(save_path, dpi=300, bbox_inches='tight')
                logger.info(f"Multi-ROC comparison saved to {save_path}")
            
            logger.info(f"Created ROC comparison for {len(model_results)} models")
            return fig
            
        except Exception as e:
            logger.error(f"Error creating multi-ROC comparison: {str(e)}")
            raise
    
    def close_all_figures(self):
        """Close all matplotlib figures to free memory."""
        plt.close('all')
        logger.info("Closed all matplotlib figures")