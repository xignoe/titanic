"""
Data exploration utilities for the Titanic Survival Predictor.

This module provides the DataExplorer class for analyzing and exploring
the Titanic dataset to understand patterns and relationships in the data.
"""

import pandas as pd
import numpy as np
from typing import Dict, Any, List, Optional
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DataExplorer:
    """
    Handles exploration and analysis of the Titanic dataset.
    
    This class provides methods to generate statistical summaries,
    analyze missing values, and examine survival patterns by different
    feature groups.
    """
    
    def __init__(self):
        """Initialize DataExplorer."""
        pass
    
    def generate_summary_statistics(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Generate comprehensive statistical summary for all features.
        
        Args:
            df: DataFrame to analyze
            
        Returns:
            dict: Statistical summary including descriptive stats for numeric
                  and categorical features
        """
        logger.info("Generating summary statistics")
        
        summary = {
            'dataset_info': {
                'total_records': len(df),
                'total_features': len(df.columns),
                'memory_usage_mb': df.memory_usage(deep=True).sum() / (1024 * 1024)
            },
            'numeric_features': {},
            'categorical_features': {}
        }
        
        # Analyze numeric features
        numeric_columns = df.select_dtypes(include=[np.number]).columns
        for col in numeric_columns:
            summary['numeric_features'][col] = {
                'count': df[col].count(),
                'mean': df[col].mean() if df[col].count() > 0 else None,
                'std': df[col].std() if df[col].count() > 0 else None,
                'min': df[col].min() if df[col].count() > 0 else None,
                'max': df[col].max() if df[col].count() > 0 else None,
                'median': df[col].median() if df[col].count() > 0 else None,
                'q25': df[col].quantile(0.25) if df[col].count() > 0 else None,
                'q75': df[col].quantile(0.75) if df[col].count() > 0 else None,
                'missing_count': df[col].isnull().sum(),
                'missing_percentage': (df[col].isnull().sum() / len(df)) * 100
            }
        
        # Analyze categorical features
        categorical_columns = df.select_dtypes(include=['object']).columns
        for col in categorical_columns:
            value_counts = df[col].value_counts()
            summary['categorical_features'][col] = {
                'count': df[col].count(),
                'unique_values': df[col].nunique(),
                'most_frequent': value_counts.index[0] if len(value_counts) > 0 else None,
                'most_frequent_count': value_counts.iloc[0] if len(value_counts) > 0 else 0,
                'value_counts': value_counts.to_dict(),
                'missing_count': df[col].isnull().sum(),
                'missing_percentage': (df[col].isnull().sum() / len(df)) * 100
            }
        
        logger.info(f"Generated statistics for {len(numeric_columns)} numeric and {len(categorical_columns)} categorical features")
        return summary
    
    def analyze_missing_values(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Analyze missing value patterns with counts and percentages.
        
        Args:
            df: DataFrame to analyze
            
        Returns:
            dict: Missing value analysis including counts, percentages,
                  and patterns
        """
        logger.info("Analyzing missing values")
        
        missing_analysis = {
            'total_missing': df.isnull().sum().sum(),
            'total_cells': df.size,
            'overall_missing_percentage': (df.isnull().sum().sum() / df.size) * 100,
            'by_column': {},
            'missing_patterns': {}
        }
        
        # Analyze missing values by column
        for col in df.columns:
            missing_count = df[col].isnull().sum()
            missing_analysis['by_column'][col] = {
                'missing_count': missing_count,
                'missing_percentage': (missing_count / len(df)) * 100,
                'present_count': len(df) - missing_count,
                'present_percentage': ((len(df) - missing_count) / len(df)) * 100
            }
        
        # Identify columns with missing values
        columns_with_missing = [col for col in df.columns if df[col].isnull().sum() > 0]
        
        # Analyze missing value patterns (combinations of missing columns)
        if columns_with_missing:
            # Create a DataFrame showing missing patterns
            missing_pattern_df = df[columns_with_missing].isnull()
            pattern_counts = missing_pattern_df.value_counts()
            
            missing_analysis['missing_patterns'] = {
                'total_patterns': len(pattern_counts),
                'patterns': {}
            }
            
            for pattern, count in pattern_counts.items():
                pattern_key = '_'.join([f"{col}:{val}" for col, val in zip(columns_with_missing, pattern)])
                missing_analysis['missing_patterns']['patterns'][pattern_key] = {
                    'count': count,
                    'percentage': (count / len(df)) * 100
                }
        
        # Summary statistics
        missing_analysis['summary'] = {
            'columns_with_missing': len(columns_with_missing),
            'columns_without_missing': len(df.columns) - len(columns_with_missing),
            'rows_with_any_missing': df.isnull().any(axis=1).sum(),
            'rows_without_missing': (~df.isnull().any(axis=1)).sum(),
            'most_missing_column': df.isnull().sum().idxmax() if df.isnull().sum().sum() > 0 else None,
            'most_missing_count': df.isnull().sum().max() if df.isnull().sum().sum() > 0 else 0
        }
        
        logger.info(f"Found missing values in {len(columns_with_missing)} columns")
        return missing_analysis
    
    def analyze_survival_patterns(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Analyze survival patterns by feature groups (sex, class, etc.).
        
        Args:
            df: DataFrame to analyze (must contain 'Survived' column)
            
        Returns:
            dict: Survival pattern analysis by different feature groups
            
        Raises:
            ValueError: If 'Survived' column is not present
        """
        if 'Survived' not in df.columns:
            raise ValueError("DataFrame must contain 'Survived' column for survival analysis")
        
        logger.info("Analyzing survival patterns")
        
        survival_analysis = {
            'overall_survival': {
                'total_passengers': len(df),
                'survivors': df['Survived'].sum(),
                'deaths': len(df) - df['Survived'].sum(),
                'survival_rate': df['Survived'].mean() * 100
            },
            'by_feature': {}
        }
        
        # Analyze survival by categorical features
        categorical_features = ['Sex', 'Pclass', 'Embarked']
        
        for feature in categorical_features:
            if feature in df.columns:
                survival_analysis['by_feature'][feature] = self._analyze_survival_by_category(df, feature)
        
        # Analyze survival by age groups
        if 'Age' in df.columns:
            survival_analysis['by_feature']['Age_Groups'] = self._analyze_survival_by_age_groups(df)
        
        # Analyze survival by fare groups
        if 'Fare' in df.columns:
            survival_analysis['by_feature']['Fare_Groups'] = self._analyze_survival_by_fare_groups(df)
        
        # Analyze survival by family size
        if 'SibSp' in df.columns and 'Parch' in df.columns:
            survival_analysis['by_feature']['Family_Size'] = self._analyze_survival_by_family_size(df)
        
        logger.info(f"Analyzed survival patterns across {len(survival_analysis['by_feature'])} feature groups")
        return survival_analysis
    
    def _analyze_survival_by_category(self, df: pd.DataFrame, feature: str) -> Dict[str, Any]:
        """
        Analyze survival rates by categorical feature.
        
        Args:
            df: DataFrame to analyze
            feature: Categorical feature to analyze
            
        Returns:
            dict: Survival analysis for the categorical feature
        """
        # Remove rows where the feature is missing
        feature_df = df.dropna(subset=[feature])
        
        analysis = {
            'total_with_data': len(feature_df),
            'missing_data': len(df) - len(feature_df),
            'categories': {}
        }
        
        for category in feature_df[feature].unique():
            category_data = feature_df[feature_df[feature] == category]
            survivors = category_data['Survived'].sum()
            total = len(category_data)
            
            analysis['categories'][str(category)] = {
                'total': total,
                'survivors': survivors,
                'deaths': total - survivors,
                'survival_rate': (survivors / total * 100) if total > 0 else 0,
                'percentage_of_total': (total / len(feature_df) * 100) if len(feature_df) > 0 else 0
            }
        
        return analysis
    
    def _analyze_survival_by_age_groups(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Analyze survival rates by age groups.
        
        Args:
            df: DataFrame to analyze
            
        Returns:
            dict: Survival analysis by age groups
        """
        # Remove rows where age is missing
        age_df = df.dropna(subset=['Age'])
        
        # Define age groups
        age_bins = [0, 12, 18, 35, 60, 100]
        age_labels = ['Child (0-12)', 'Teen (13-18)', 'Young Adult (19-35)', 'Adult (36-60)', 'Senior (60+)']
        
        age_df = age_df.copy()
        age_df['Age_Group'] = pd.cut(age_df['Age'], bins=age_bins, labels=age_labels, right=False)
        
        return self._analyze_survival_by_category(age_df, 'Age_Group')
    
    def _analyze_survival_by_fare_groups(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Analyze survival rates by fare groups.
        
        Args:
            df: DataFrame to analyze
            
        Returns:
            dict: Survival analysis by fare groups
        """
        # Remove rows where fare is missing
        fare_df = df.dropna(subset=['Fare'])
        
        # Define fare groups based on quartiles
        fare_quartiles = fare_df['Fare'].quantile([0, 0.25, 0.5, 0.75, 1.0])
        fare_bins = [fare_quartiles.iloc[i] for i in range(len(fare_quartiles))]
        fare_labels = ['Low Fare (Q1)', 'Medium-Low Fare (Q2)', 'Medium-High Fare (Q3)', 'High Fare (Q4)']
        
        fare_df = fare_df.copy()
        fare_df['Fare_Group'] = pd.cut(fare_df['Fare'], bins=fare_bins, labels=fare_labels, include_lowest=True)
        
        return self._analyze_survival_by_category(fare_df, 'Fare_Group')
    
    def _analyze_survival_by_family_size(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Analyze survival rates by family size.
        
        Args:
            df: DataFrame to analyze
            
        Returns:
            dict: Survival analysis by family size
        """
        # Calculate family size
        family_df = df.copy()
        family_df['Family_Size'] = family_df['SibSp'] + family_df['Parch'] + 1
        
        # Create family size categories
        def categorize_family_size(size):
            if size == 1:
                return 'Alone'
            elif size <= 3:
                return 'Small Family (2-3)'
            elif size <= 6:
                return 'Medium Family (4-6)'
            else:
                return 'Large Family (7+)'
        
        family_df['Family_Size_Category'] = family_df['Family_Size'].apply(categorize_family_size)
        
        return self._analyze_survival_by_category(family_df, 'Family_Size_Category')
    
    def create_exploration_report(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Create a comprehensive exploration report combining all analyses.
        
        Args:
            df: DataFrame to analyze
            
        Returns:
            dict: Complete exploration report
        """
        logger.info("Creating comprehensive exploration report")
        
        report = {
            'summary_statistics': self.generate_summary_statistics(df),
            'missing_values': self.analyze_missing_values(df)
        }
        
        # Add survival analysis if this is training data
        if 'Survived' in df.columns:
            report['survival_patterns'] = self.analyze_survival_patterns(df)
        
        logger.info("Exploration report completed")
        return report