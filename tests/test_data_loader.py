"""
Unit tests for the DataLoader class.
"""

import unittest
import pandas as pd
import os
import tempfile
import shutil
from unittest.mock import patch, MagicMock
import sys

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from data.loader import DataLoader


class TestDataLoader(unittest.TestCase):
    """Test cases for DataLoader class."""
    
    def setUp(self):
        """Set up test fixtures."""
        # Create temporary directory for test files
        self.test_dir = tempfile.mkdtemp()
        self.loader = DataLoader(data_dir=self.test_dir)
        
        # Sample valid training data
        self.valid_train_data = {
            'PassengerId': [1, 2, 3],
            'Survived': [0, 1, 1],
            'Pclass': [3, 1, 3],
            'Name': ['Braund, Mr. Owen Harris', 'Cumings, Mrs. John Bradley', 'Heikkinen, Miss. Laina'],
            'Sex': ['male', 'female', 'female'],
            'Age': [22.0, 38.0, 26.0],
            'SibSp': [1, 1, 0],
            'Parch': [0, 0, 0],
            'Ticket': ['A/5 21171', 'PC 17599', 'STON/O2. 3101282'],
            'Fare': [7.25, 71.2833, 7.925],
            'Cabin': [None, 'C85', None],
            'Embarked': ['S', 'C', 'S']
        }
        
        # Sample valid test data (no Survived column)
        self.valid_test_data = {
            'PassengerId': [892, 893, 894],
            'Pclass': [3, 3, 2],
            'Name': ['Kelly, Mr. James', 'Wilkes, Mrs. James', 'Myles, Mr. Thomas Francis'],
            'Sex': ['male', 'female', 'male'],
            'Age': [34.5, 47.0, 62.0],
            'SibSp': [0, 1, 0],
            'Parch': [0, 0, 0],
            'Ticket': ['330911', '363272', '240276'],
            'Fare': [7.8292, 7.0, 9.6875],
            'Cabin': [None, None, None],
            'Embarked': ['Q', 'S', 'Q']
        }
    
    def tearDown(self):
        """Clean up test fixtures."""
        shutil.rmtree(self.test_dir)
    
    def create_test_csv(self, filename, data):
        """Helper method to create test CSV files."""
        df = pd.DataFrame(data)
        filepath = os.path.join(self.test_dir, filename)
        df.to_csv(filepath, index=False)
        return filepath
    
    def test_init(self):
        """Test DataLoader initialization."""
        loader = DataLoader("test_dir")
        self.assertEqual(loader.data_dir, "test_dir")
        self.assertEqual(loader.train_file, os.path.join("test_dir", "train.csv"))
        self.assertEqual(loader.test_file, os.path.join("test_dir", "test.csv"))
    
    def test_load_train_data_success(self):
        """Test successful loading of training data."""
        # Create valid training file
        self.create_test_csv("train.csv", self.valid_train_data)
        
        # Load data
        df = self.loader.load_train_data()
        
        # Verify data
        self.assertEqual(len(df), 3)
        self.assertIn('Survived', df.columns)
        self.assertEqual(list(df['PassengerId']), [1, 2, 3])
        self.assertEqual(list(df['Survived']), [0, 1, 1])
    
    def test_load_test_data_success(self):
        """Test successful loading of test data."""
        # Create valid test file
        self.create_test_csv("test.csv", self.valid_test_data)
        
        # Load data
        df = self.loader.load_test_data()
        
        # Verify data
        self.assertEqual(len(df), 3)
        self.assertNotIn('Survived', df.columns)
        self.assertEqual(list(df['PassengerId']), [892, 893, 894])
    
    def test_load_train_data_file_not_found(self):
        """Test error handling when training file doesn't exist."""
        with self.assertRaises(FileNotFoundError) as context:
            self.loader.load_train_data()
        
        self.assertIn("Training file not found", str(context.exception))
    
    def test_load_test_data_file_not_found(self):
        """Test error handling when test file doesn't exist."""
        with self.assertRaises(FileNotFoundError) as context:
            self.loader.load_test_data()
        
        self.assertIn("Test file not found", str(context.exception))
    
    def test_load_empty_file(self):
        """Test error handling for empty CSV files."""
        # Create empty file
        empty_file = os.path.join(self.test_dir, "train.csv")
        with open(empty_file, 'w') as f:
            f.write("")
        
        with self.assertRaises(ValueError) as context:
            self.loader.load_train_data()
        
        self.assertIn("empty", str(context.exception).lower())
    
    def test_validate_data_format_training_success(self):
        """Test successful validation of training data format."""
        df = pd.DataFrame(self.valid_train_data)
        result = self.loader.validate_data_format(df, is_training=True)
        self.assertTrue(result)
    
    def test_validate_data_format_test_success(self):
        """Test successful validation of test data format."""
        df = pd.DataFrame(self.valid_test_data)
        result = self.loader.validate_data_format(df, is_training=False)
        self.assertTrue(result)
    
    def test_validate_data_format_missing_columns(self):
        """Test validation failure for missing required columns."""
        # Remove required column
        invalid_data = self.valid_train_data.copy()
        del invalid_data['PassengerId']
        df = pd.DataFrame(invalid_data)
        
        with self.assertRaises(ValueError) as context:
            self.loader.validate_data_format(df, is_training=True)
        
        self.assertIn("Missing required columns", str(context.exception))
        self.assertIn("PassengerId", str(context.exception))
    
    def test_validate_data_format_empty_dataframe(self):
        """Test validation failure for empty DataFrame."""
        df = pd.DataFrame()
        
        with self.assertRaises(ValueError) as context:
            self.loader.validate_data_format(df)
        
        self.assertIn("DataFrame is empty", str(context.exception))
    
    def test_validate_data_format_invalid_survived_values(self):
        """Test validation failure for invalid Survived values."""
        invalid_data = self.valid_train_data.copy()
        invalid_data['Survived'] = [0, 1, 2]  # Invalid value: 2
        df = pd.DataFrame(invalid_data)
        
        with self.assertRaises(ValueError) as context:
            self.loader.validate_data_format(df, is_training=True)
        
        self.assertIn("Survived column must contain only 0 or 1", str(context.exception))
    
    def test_validate_data_format_invalid_pclass_values(self):
        """Test validation failure for invalid Pclass values."""
        invalid_data = self.valid_train_data.copy()
        invalid_data['Pclass'] = [1, 2, 4]  # Invalid value: 4
        df = pd.DataFrame(invalid_data)
        
        with self.assertRaises(ValueError) as context:
            self.loader.validate_data_format(df, is_training=True)
        
        self.assertIn("Pclass must contain only values 1, 2, or 3", str(context.exception))
    
    def test_validate_data_format_invalid_sex_values(self):
        """Test validation failure for invalid Sex values."""
        invalid_data = self.valid_train_data.copy()
        invalid_data['Sex'] = ['male', 'female', 'other']  # Invalid value: 'other'
        df = pd.DataFrame(invalid_data)
        
        with self.assertRaises(ValueError) as context:
            self.loader.validate_data_format(df, is_training=True)
        
        self.assertIn("Sex column must contain only 'male' or 'female'", str(context.exception))
    
    def test_validate_data_format_duplicate_passenger_ids(self):
        """Test validation failure for duplicate PassengerId values."""
        invalid_data = self.valid_train_data.copy()
        invalid_data['PassengerId'] = [1, 1, 3]  # Duplicate ID
        df = pd.DataFrame(invalid_data)
        
        with self.assertRaises(ValueError) as context:
            self.loader.validate_data_format(df, is_training=True)
        
        self.assertIn("PassengerId values must be unique", str(context.exception))
    
    def test_validate_data_format_non_numeric_age(self):
        """Test validation failure for non-numeric Age values."""
        invalid_data = self.valid_train_data.copy()
        invalid_data['Age'] = [22.0, 'thirty-eight', 26.0]  # Invalid: string
        df = pd.DataFrame(invalid_data)
        
        with self.assertRaises(ValueError) as context:
            self.loader.validate_data_format(df, is_training=True)
        
        self.assertIn("Column Age contains non-numeric values", str(context.exception))
    
    def test_get_data_info(self):
        """Test getting data information."""
        df = pd.DataFrame(self.valid_train_data)
        info = self.loader.get_data_info(df)
        
        self.assertEqual(info['shape'], (3, 12))
        self.assertIn('PassengerId', info['columns'])
        self.assertIn('Survived', info['columns'])
        self.assertIsInstance(info['dtypes'], dict)
        self.assertIsInstance(info['missing_values'], dict)
        self.assertTrue(pd.api.types.is_integer(info['memory_usage']))
    
    @patch('data.loader.logger')
    def test_logging_on_successful_load(self, mock_logger):
        """Test that appropriate log messages are generated on successful load."""
        # Create valid training file
        self.create_test_csv("train.csv", self.valid_train_data)
        
        # Load data
        self.loader.load_train_data()
        
        # Verify logging calls
        mock_logger.info.assert_called()
        log_calls = [call[0][0] for call in mock_logger.info.call_args_list]
        self.assertTrue(any("Loading training data" in call for call in log_calls))
        self.assertTrue(any("Successfully loaded" in call for call in log_calls))


if __name__ == '__main__':
    unittest.main()