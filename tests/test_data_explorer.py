"""
Unit tests for the DataExplorer class.

This module contains comprehensive tests for data exploration functionality
including statistical summaries, missing value analysis, and survival patterns.
"""

import unittest
import pandas as pd
import numpy as np
from unittest.mock import patch, MagicMock
import sys
import os

# Add src directory to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from data.explorer import DataExplorer


class TestDataExplorer(unittest.TestCase):
    """Test cases for DataExplorer class."""
    
    def setUp(self):
        """Set up test fixtures before each test method."""
        self.explorer = DataExplorer()
        
        # Create sample training data
        self.sample_train_data = pd.DataFrame({
            'PassengerId': [1, 2, 3, 4, 5],
            'Survived': [0, 1, 1, 0, 1],
            'Pclass': [3, 1, 3, 1, 2],
            'Name': ['Smith, Mr. John', 'Johnson, Mrs. Mary', 'Brown, Miss. Jane', 'Davis, Mr. Bob', 'Wilson, Mrs. Sue'],
            'Sex': ['male', 'female', 'female', 'male', 'female'],
            'Age': [22.0, 38.0, np.nan, 35.0, 29.0],
            'SibSp': [1, 1, 0, 1, 0],
            'Parch': [0, 0, 0, 0, 0],
            'Ticket': ['A123', 'B456', 'C789', 'D012', 'E345'],
            'Fare': [7.25, 71.28, 7.92, np.nan, 16.10],
            'Cabin': [np.nan, 'C85', np.nan, 'B96', np.nan],
            'Embarked': ['S', 'C', 'S', 'S', 'Q']
        })
        
        # Create sample test data (no Survived column)
        self.sample_test_data = self.sample_train_data.drop('Survived', axis=1)
    
    def test_init(self):
        """Test DataExplorer initialization."""
        explorer = DataExplorer()
        self.assertIsInstance(explorer, DataExplorer)
    
    def test_generate_summary_statistics_basic(self):
        """Test basic summary statistics generation."""
        result = self.explorer.generate_summary_statistics(self.sample_train_data)
        
        # Check structure
        self.assertIn('dataset_info', result)
        self.assertIn('numeric_features', result)
        self.assertIn('categorical_features', result)
        
        # Check dataset info
        self.assertEqual(result['dataset_info']['total_records'], 5)
        self.assertEqual(result['dataset_info']['total_features'], 12)
        self.assertIsInstance(result['dataset_info']['memory_usage_mb'], float)
    
    def test_generate_summary_statistics_numeric_features(self):
        """Test summary statistics for numeric features."""
        result = self.explorer.generate_summary_statistics(self.sample_train_data)
        
        # Check numeric features are present
        numeric_features = result['numeric_features']
        expected_numeric = ['PassengerId', 'Survived', 'Pclass', 'Age', 'SibSp', 'Parch', 'Fare']
        
        for feature in expected_numeric:
            self.assertIn(feature, numeric_features)
        
        # Check Age statistics (has missing values)
        age_stats = numeric_features['Age']
        self.assertEqual(age_stats['count'], 4)  # 4 non-null values
        self.assertEqual(age_stats['missing_count'], 1)
        self.assertEqual(age_stats['missing_percentage'], 20.0)
        self.assertAlmostEqual(age_stats['mean'], 31.0, places=1)
    
    def test_generate_summary_statistics_categorical_features(self):
        """Test summary statistics for categorical features."""
        result = self.explorer.generate_summary_statistics(self.sample_train_data)
        
        # Check categorical features are present
        categorical_features = result['categorical_features']
        expected_categorical = ['Name', 'Sex', 'Ticket', 'Cabin', 'Embarked']
        
        for feature in expected_categorical:
            self.assertIn(feature, categorical_features)
        
        # Check Sex statistics
        sex_stats = categorical_features['Sex']
        self.assertEqual(sex_stats['count'], 5)
        self.assertEqual(sex_stats['unique_values'], 2)
        self.assertEqual(sex_stats['most_frequent'], 'female')
        self.assertEqual(sex_stats['most_frequent_count'], 3)
    
    def test_analyze_missing_values_basic(self):
        """Test basic missing value analysis."""
        result = self.explorer.analyze_missing_values(self.sample_train_data)
        
        # Check structure
        self.assertIn('total_missing', result)
        self.assertIn('by_column', result)
        self.assertIn('missing_patterns', result)
        self.assertIn('summary', result)
        
        # Check totals
        self.assertEqual(result['total_missing'], 5)  # Age(1) + Fare(1) + Cabin(3) = 5
        self.assertEqual(result['total_cells'], 60)  # 5 rows * 12 columns
    
    def test_analyze_missing_values_by_column(self):
        """Test missing value analysis by column."""
        result = self.explorer.analyze_missing_values(self.sample_train_data)
        
        by_column = result['by_column']
        
        # Check Age column (1 missing)
        self.assertEqual(by_column['Age']['missing_count'], 1)
        self.assertEqual(by_column['Age']['missing_percentage'], 20.0)
        self.assertEqual(by_column['Age']['present_count'], 4)
        
        # Check Cabin column (3 missing)
        self.assertEqual(by_column['Cabin']['missing_count'], 3)
        self.assertEqual(by_column['Cabin']['missing_percentage'], 60.0)
        
        # Check PassengerId column (0 missing)
        self.assertEqual(by_column['PassengerId']['missing_count'], 0)
        self.assertEqual(by_column['PassengerId']['missing_percentage'], 0.0)
    
    def test_analyze_missing_values_summary(self):
        """Test missing value summary statistics."""
        result = self.explorer.analyze_missing_values(self.sample_train_data)
        
        summary = result['summary']
        self.assertEqual(summary['columns_with_missing'], 3)  # Age, Fare, Cabin
        self.assertEqual(summary['columns_without_missing'], 9)
        self.assertEqual(summary['most_missing_column'], 'Cabin')
        self.assertEqual(summary['most_missing_count'], 3)
    
    def test_analyze_survival_patterns_basic(self):
        """Test basic survival pattern analysis."""
        result = self.explorer.analyze_survival_patterns(self.sample_train_data)
        
        # Check structure
        self.assertIn('overall_survival', result)
        self.assertIn('by_feature', result)
        
        # Check overall survival
        overall = result['overall_survival']
        self.assertEqual(overall['total_passengers'], 5)
        self.assertEqual(overall['survivors'], 3)
        self.assertEqual(overall['deaths'], 2)
        self.assertEqual(overall['survival_rate'], 60.0)
    
    def test_analyze_survival_patterns_by_sex(self):
        """Test survival pattern analysis by sex."""
        result = self.explorer.analyze_survival_patterns(self.sample_train_data)
        
        sex_analysis = result['by_feature']['Sex']
        
        # Check structure
        self.assertIn('categories', sex_analysis)
        self.assertIn('female', sex_analysis['categories'])
        self.assertIn('male', sex_analysis['categories'])
        
        # Check female survival (3 females, all survived in our sample)
        female_stats = sex_analysis['categories']['female']
        self.assertEqual(female_stats['total'], 3)
        self.assertEqual(female_stats['survivors'], 3)
        self.assertEqual(female_stats['survival_rate'], 100.0)
        
        # Check male survival (2 males, 0 survived in our sample)
        male_stats = sex_analysis['categories']['male']
        self.assertEqual(male_stats['total'], 2)
        self.assertEqual(male_stats['survivors'], 0)
        self.assertEqual(male_stats['survival_rate'], 0.0)
    
    def test_analyze_survival_patterns_by_pclass(self):
        """Test survival pattern analysis by passenger class."""
        result = self.explorer.analyze_survival_patterns(self.sample_train_data)
        
        pclass_analysis = result['by_feature']['Pclass']
        
        # Check that all classes are represented
        self.assertIn('1', pclass_analysis['categories'])
        self.assertIn('2', pclass_analysis['categories'])
        self.assertIn('3', pclass_analysis['categories'])
    
    def test_analyze_survival_patterns_no_survived_column(self):
        """Test that survival analysis raises error without Survived column."""
        with self.assertRaises(ValueError) as context:
            self.explorer.analyze_survival_patterns(self.sample_test_data)
        
        self.assertIn("must contain 'Survived' column", str(context.exception))
    
    def test_analyze_survival_by_age_groups(self):
        """Test survival analysis by age groups."""
        result = self.explorer.analyze_survival_patterns(self.sample_train_data)
        
        # Check that age groups analysis is present
        self.assertIn('Age_Groups', result['by_feature'])
        age_groups = result['by_feature']['Age_Groups']
        
        # Should have some age group categories
        self.assertIn('categories', age_groups)
        self.assertGreater(len(age_groups['categories']), 0)
    
    def test_analyze_survival_by_family_size(self):
        """Test survival analysis by family size."""
        result = self.explorer.analyze_survival_patterns(self.sample_train_data)
        
        # Check that family size analysis is present
        self.assertIn('Family_Size', result['by_feature'])
        family_analysis = result['by_feature']['Family_Size']
        
        # Should have family size categories
        self.assertIn('categories', family_analysis)
        self.assertGreater(len(family_analysis['categories']), 0)
    
    def test_create_exploration_report_with_survival(self):
        """Test comprehensive exploration report creation with survival data."""
        result = self.explorer.create_exploration_report(self.sample_train_data)
        
        # Check all sections are present
        self.assertIn('summary_statistics', result)
        self.assertIn('missing_values', result)
        self.assertIn('survival_patterns', result)
        
        # Verify each section has expected structure
        self.assertIn('dataset_info', result['summary_statistics'])
        self.assertIn('total_missing', result['missing_values'])
        self.assertIn('overall_survival', result['survival_patterns'])
    
    def test_create_exploration_report_without_survival(self):
        """Test comprehensive exploration report creation without survival data."""
        result = self.explorer.create_exploration_report(self.sample_test_data)
        
        # Check sections are present (no survival patterns)
        self.assertIn('summary_statistics', result)
        self.assertIn('missing_values', result)
        self.assertNotIn('survival_patterns', result)
    
    def test_empty_dataframe(self):
        """Test handling of empty DataFrame."""
        empty_df = pd.DataFrame()
        
        # Should handle empty DataFrame gracefully
        result = self.explorer.generate_summary_statistics(empty_df)
        self.assertEqual(result['dataset_info']['total_records'], 0)
        self.assertEqual(result['dataset_info']['total_features'], 0)
    
    def test_dataframe_with_all_missing_values(self):
        """Test handling of DataFrame with all missing values."""
        all_missing_df = pd.DataFrame({
            'col1': [np.nan, np.nan, np.nan],
            'col2': [np.nan, np.nan, np.nan]
        })
        
        result = self.explorer.analyze_missing_values(all_missing_df)
        self.assertEqual(result['total_missing'], 6)  # 3 rows * 2 columns
        self.assertEqual(result['overall_missing_percentage'], 100.0)
    
    @patch('data.explorer.logger')
    def test_logging_calls(self, mock_logger):
        """Test that appropriate logging calls are made."""
        self.explorer.generate_summary_statistics(self.sample_train_data)
        
        # Verify logging calls were made
        mock_logger.info.assert_called()
        
        # Check specific log messages
        log_calls = [call[0][0] for call in mock_logger.info.call_args_list]
        self.assertTrue(any("Generating summary statistics" in call for call in log_calls))


if __name__ == '__main__':
    unittest.main()